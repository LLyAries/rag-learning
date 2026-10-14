from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_pdf(pdf_path: Path):
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    for document in documents:
        document.metadata["file_name"] = (
            pdf_path.name
        )

        document.metadata["document_id"] = (
            pdf_path.stem
        )

        document.metadata["page_number"] = (
            document.metadata.get("page", 0) + 1
        )

    return documents


def load_pdfs(data_dir: Path):
    all_documents = []

    pdf_paths = sorted(
        data_dir.glob("*.pdf")
    )

    for pdf_path in pdf_paths:
        documents = load_pdf(pdf_path)

        all_documents.extend(documents)

        print(
            f"已读取：{pdf_path.name}，"
            f"页数：{len(documents)}"
        )

    return all_documents


if __name__ == "__main__":
    documents = load_pdfs(DATA_DIR)

    print(f"\nPDF 总页数：{len(documents)}")

    print("\n第一条 Document 的 metadata：")
    print(documents[0].metadata)

    print("\n第一条 Document 的文本预览：")
    print(documents[0].page_content[:300])