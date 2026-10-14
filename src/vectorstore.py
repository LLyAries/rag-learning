from pathlib import Path

from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from loader import DATA_DIR, load_pdfs
from splitter import split_documents


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# embedding 模型
MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# FAISS 索引保存位置
INDEX_PATH = Path.home() / "rag-learning-faiss-index"


def create_embeddings():
    """
    创建 embedding 模型。

    文档和问题必须使用相同的模型及相同的归一化设置。
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    return embeddings


def build_vectorstore(chunks):
    """
    输入：chunk Document 列表
    输出：内存中的 FAISS 向量库
    """
    embeddings = create_embeddings()

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vectorstore


def save_vectorstore(vectorstore, index_path):
    """
    把内存中的 FAISS 向量库保存到磁盘。
    """
    vectorstore.save_local(str(index_path))


def load_vectorstore(index_path):
    """
    从磁盘加载 FAISS 向量库。

    加载后可以直接检索，无需重新读取和向量化 PDF。
    """
    embeddings = create_embeddings()

    vectorstore = FAISS.load_local(
        folder_path=str(index_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore


if __name__ == "__main__":
    # 1. 读取 PDF
    documents = load_pdfs(DATA_DIR)

    # 2. 切分 chunks
    chunks = split_documents(documents)

    # 3. 建立 FAISS
    vectorstore = build_vectorstore(chunks)

    # 4. 保存 FAISS
    save_vectorstore(vectorstore, INDEX_PATH)

    print(f"PDF 页数：{len(documents)}")
    print(f"chunk 数量：{len(chunks)}")
    print(f"FAISS 中的向量数量：{vectorstore.index.ntotal}")
    print(f"每个向量的维度：{vectorstore.index.d}")
    print(f"索引已保存到：{INDEX_PATH}")
