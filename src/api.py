import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from llm import create_llm
from loader import DATA_DIR, load_pdf, load_pdfs
from rag import build_context, generate_answer, retrieve
from splitter import split_documents
from vectorstore import (
    INDEX_PATH,
    build_vectorstore,
    load_vectorstore,
    save_vectorstore,
)


DEFAULT_TOP_K = 5
MAX_TOP_K = 10
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
INVALID_FILENAME_PATTERN = re.compile(r'[<>:"/\\|?*]')


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str):
        value = value.strip()
        if not value:
            raise ValueError("问题不能为空")
        return value


class SourceItem(BaseModel):
    number: int
    file_name: str
    page_number: int
    chunk_id: int
    score: float
    preview: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str
    vector_count: int
    vector_dimension: int
    llm_model: str


class DocumentItem(BaseModel):
    file_name: str
    size_bytes: int


class IndexResponse(BaseModel):
    document_pages: int
    chunk_count: int
    vector_count: int


class UploadResponse(IndexResponse):
    file_name: str
    uploaded_pages: int


def format_sources(results):
    sources = []

    for source_number, (document, score) in enumerate(results, start=1):
        sources.append(
            SourceItem(
                number=source_number,
                file_name=document.metadata.get("file_name", "未知文件"),
                page_number=document.metadata.get("page_number", -1),
                chunk_id=document.metadata.get("chunk_id", -1),
                score=float(score),
                preview=document.page_content.replace("\n", " ")[:240],
            )
        )

    return sources


def validate_filename(raw_filename: str | None):
    filename = Path(raw_filename or "").name.strip()

    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="文件名无效")

    if INVALID_FILENAME_PATTERN.search(filename):
        raise HTTPException(status_code=400, detail="文件名包含非法字符")

    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="只允许上传 PDF 文件")

    return filename


def rebuild_index(app: FastAPI):
    """完整重建索引，成功后再切换内存中的向量库。"""
    with app.state.reindex_lock:
        documents = load_pdfs(DATA_DIR)

        if not documents:
            raise ValueError("data 目录中没有可索引的 PDF")

        chunks = split_documents(documents)

        if not chunks:
            raise ValueError("PDF 没有产生可索引的文本块")

        new_vectorstore = build_vectorstore(chunks)
        save_vectorstore(new_vectorstore, INDEX_PATH)
        app.state.vectorstore = new_vectorstore

        return IndexResponse(
            document_pages=len(documents),
            chunk_count=len(chunks),
            vector_count=new_vectorstore.index.ntotal,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时只加载一次 FAISS 和大模型客户端。"""
    app.state.vectorstore = load_vectorstore(INDEX_PATH)
    app.state.llm = create_llm()
    app.state.reindex_lock = threading.Lock()
    yield


app = FastAPI(
    title="论文 RAG API",
    description="基于 FAISS 与通义千问的多文档问答接口",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    vectorstore = request.app.state.vectorstore
    return HealthResponse(
        status="ok",
        vector_count=vectorstore.index.ntotal,
        vector_dimension=vectorstore.index.d,
        llm_model=os.getenv("LLM_MODEL", "unknown"),
    )


@app.get("/documents", response_model=list[DocumentItem])
def list_documents():
    return [
        DocumentItem(file_name=path.name, size_bytes=path.stat().st_size)
        for path in sorted(DATA_DIR.glob("*.pdf"))
    ]


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
):
    filename = validate_filename(file.filename)
    destination = DATA_DIR / filename

    if destination.exists():
        raise HTTPException(
            status_code=409,
            detail="同名 PDF 已存在，请先更换文件名",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="PDF 不能超过 20 MB",
        )

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="文件内容不是有效的 PDF",
        )

    temp_dir = DATA_DIR / ".uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{uuid4().hex}.pdf"
    temp_path.write_bytes(content)

    try:
        uploaded_documents = await run_in_threadpool(load_pdf, temp_path)

        if not uploaded_documents:
            raise ValueError("PDF 中没有可读取的页面")

        temp_path.replace(destination)

        try:
            index_result = await run_in_threadpool(
                rebuild_index,
                request.app,
            )
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"PDF 处理失败：{error}",
        ) from error
    finally:
        temp_path.unlink(missing_ok=True)

    return UploadResponse(
        file_name=filename,
        uploaded_pages=len(uploaded_documents),
        document_pages=index_result.document_pages,
        chunk_count=index_result.chunk_count,
        vector_count=index_result.vector_count,
    )


@app.post("/reindex", response_model=IndexResponse)
def reindex(request: Request):
    try:
        return rebuild_index(request.app)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"索引重建失败：{error}",
        ) from error


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request):
    started_at = perf_counter()

    try:
        results = retrieve(
            vectorstore=request.app.state.vectorstore,
            query=payload.question,
            k=payload.top_k,
        )
        context = build_context(results)
        answer = generate_answer(
            llm=request.app.state.llm,
            query=payload.question,
            context=context,
        )
        sources = format_sources(results)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"RAG 处理失败：{error}",
        ) from error

    return AskResponse(
        question=payload.question,
        answer=answer,
        sources=sources,
        elapsed_ms=(perf_counter() - started_at) * 1000,
    )
