import os

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv(override=True)

API_BASE_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")
REQUEST_TIMEOUT_SECONDS = 120
UPLOAD_TIMEOUT_SECONDS = 600


st.set_page_config(
    page_title="论文 RAG 助手",
    page_icon="📚",
    layout="centered",
)


def get_health():
    response = requests.get(f"{API_BASE_URL}/health", timeout=5)
    response.raise_for_status()
    return response.json()


def get_documents():
    response = requests.get(f"{API_BASE_URL}/documents", timeout=5)
    response.raise_for_status()
    return response.json()


def upload_pdf(uploaded_file):
    response = requests.post(
        f"{API_BASE_URL}/documents/upload",
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf",
            )
        },
        timeout=UPLOAD_TIMEOUT_SECONDS,
    )

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(detail)

    return response.json()


def ask_api(question, top_k):
    response = requests.post(
        f"{API_BASE_URL}/ask",
        json={"question": question, "top_k": top_k},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(detail)

    return response.json()


def display_sources(sources):
    with st.expander("查看检索来源", expanded=False):
        for source in sources:
            st.markdown(
                f"**[来源{source['number']}] "
                f"{source['file_name']}，"
                f"第 {source['page_number']} 页**"
            )
            st.caption(
                f"Chunk {source['chunk_id']} · "
                f"FAISS 距离 {source['score']:.4f}"
            )
            st.write(source["preview"])
            st.divider()


st.title("📚 论文 RAG 助手")
st.caption(
    "Streamlit 前端 · FastAPI 后端 · "
    "FAISS 检索 · 通义千问生成"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("系统设置")

    top_k = st.slider(
        "Top K",
        min_value=1,
        max_value=10,
        value=5,
        help="发送给大模型的检索片段数量",
    )

    try:
        health_data = get_health()
        st.success("后端服务正常")
        st.write(f"向量数量：{health_data['vector_count']}")
        st.write(f"向量维度：{health_data['vector_dimension']}")
        st.write(f"大模型：{health_data['llm_model']}")
    except requests.RequestException:
        st.error("无法连接 FastAPI 后端")

    st.divider()
    st.subheader("知识库文档")

    try:
        documents = get_documents()
        for document in documents:
            size_mb = document["size_bytes"] / (1024 * 1024)
            st.caption(
                f"• {document['file_name']} "
                f"({size_mb:.2f} MB)"
            )
    except requests.RequestException:
        st.caption("暂时无法读取文档列表")

    uploaded_file = st.file_uploader(
        "上传 PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="最大 20 MB；同名文件不会被覆盖",
    )

    if st.button(
        "上传并更新索引",
        use_container_width=True,
        disabled=uploaded_file is None,
    ):
        try:
            with st.spinner("正在解析 PDF 并重建索引..."):
                upload_result = upload_pdf(uploaded_file)

            st.success(
                f"已加入 {upload_result['file_name']}；"
                f"当前共有 {upload_result['vector_count']} 个向量"
            )
            st.rerun()
        except Exception as error:
            st.error(f"上传失败：{error}")

    st.divider()

    if st.button("清空聊天记录", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("elapsed_ms") is not None:
            st.caption(f"处理耗时：{message['elapsed_ms']:.0f} ms")

        if message.get("sources"):
            display_sources(message["sources"])

question = st.chat_input("请输入关于论文的问题")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("正在检索资料并生成答案..."):
                result = ask_api(question=question, top_k=top_k)

            st.markdown(result["answer"])
            st.caption(f"处理耗时：{result['elapsed_ms']:.0f} ms")
            display_sources(result["sources"])

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "elapsed_ms": result["elapsed_ms"],
                }
            )
        except requests.ConnectionError:
            st.error("无法连接 FastAPI 后端，请确认后端已经启动。")
        except Exception as error:
            st.error(f"请求失败：{error}")
