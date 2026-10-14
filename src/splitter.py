from langchain_text_splitters import RecursiveCharacterTextSplitter

from loader import DATA_DIR, load_pdfs


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    # 给每个 chunk 添加编号
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


if __name__ == "__main__":
    documents = load_pdfs(DATA_DIR)
    chunks = split_documents(documents)

    print(f"\nPDF 总页数：{len(documents)}")
    print(f"chunk 总数量：{len(chunks)}")

    print("\n第一个 chunk 的 metadata：")
    print(chunks[0].metadata)

    print("\n第一个 chunk 的内容：")
    print(chunks[0].page_content)

    print("\n最后一个 chunk 的 metadata：")
    print(chunks[-1].metadata)