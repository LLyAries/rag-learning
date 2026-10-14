from sentence_transformers import CrossEncoder

from vectorstore import INDEX_PATH, load_vectorstore


RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

QUERY = (
    "Why are siamese and triplet "
    "network structures used?"
)

# 第一阶段：FAISS 召回多少个候选
CANDIDATE_K = 20

# 第二阶段：重排序后保留多少个
FINAL_K = 5


def create_reranker():
    return CrossEncoder(
        RERANKER_MODEL_NAME,
        device="cpu",
    )


def retrieve_candidates(
    vectorstore,
    query,
    candidate_k,
):
    """
    第一阶段：FAISS 快速召回候选。

    输出：
    list[tuple[Document, float]]
    """
    candidates = (
        vectorstore.similarity_search_with_score(
            query=query,
            k=candidate_k,
        )
    )

    return candidates


def rerank_candidates(
    reranker,
    query,
    candidates,
    final_k,
):
    """
    第二阶段：Cross-Encoder 重排序。

    输入：
    - query：问题
    - candidates：FAISS 召回结果
    - final_k：最终保留数量

    输出：
    - Document
    - FAISS 距离
    - Reranker 分数
    """
    query_document_pairs = [
        [query, document.page_content]
        for document, faiss_score in candidates
    ]

    reranker_scores = reranker.predict(
        query_document_pairs,
        show_progress_bar=False,
    )

    combined_results = [
        (
            document,
            float(faiss_score),
            float(reranker_score),
        )
        for (
            document,
            faiss_score,
        ), reranker_score in zip(
            candidates,
            reranker_scores,
        )
    ]

    ranked_results = sorted(
        combined_results,
        key=lambda item: item[2],
        reverse=True,
    )

    return ranked_results[:final_k]


def print_faiss_results(candidates, top_k=5):
    print("\n========== FAISS 原始排名 ==========")

    for rank, (
        document,
        faiss_score,
    ) in enumerate(
        candidates[:top_k],
        start=1,
    ):
        file_name = document.metadata.get(
            "file_name",
            "未知文件",
        )

        page_number = document.metadata.get(
            "page_number",
            -1,
        )

        preview = document.page_content.replace(
            "\n",
            " ",
        )[:120]

        print(
            f"\nTop {rank} | "
            f"文件={file_name} | "
            f"页码={page_number} | "
            f"FAISS 距离={faiss_score:.4f}"
        )
        print(f"文本预览：{preview}")


def print_reranked_results(results):
    print("\n========== Reranker 排名 ==========")

    for rank, (
        document,
        faiss_score,
        reranker_score,
    ) in enumerate(
        results,
        start=1,
    ):
        file_name = document.metadata.get(
            "file_name",
            "未知文件",
        )

        page_number = document.metadata.get(
            "page_number",
            -1,
        )

        chunk_id = document.metadata.get(
            "chunk_id",
            -1,
        )

        preview = document.page_content.replace(
            "\n",
            " ",
        )[:200]

        print(
            f"\nTop {rank} | "
            f"文件={file_name} | "
            f"页码={page_number} | "
            f"Chunk={chunk_id}"
        )
        print(
            f"FAISS 距离：{faiss_score:.4f}"
        )
        print(
            f"Reranker 分数："
            f"{reranker_score:.4f}"
        )
        print(f"文本预览：{preview}")


if __name__ == "__main__":
    vectorstore = load_vectorstore(INDEX_PATH)
    reranker = create_reranker()

    candidates = retrieve_candidates(
        vectorstore=vectorstore,
        query=QUERY,
        candidate_k=CANDIDATE_K,
    )

    reranked_results = rerank_candidates(
        reranker=reranker,
        query=QUERY,
        candidates=candidates,
        final_k=FINAL_K,
    )

    print(f"问题：{QUERY}")
    print(f"FAISS 候选数量：{len(candidates)}")

    print_faiss_results(
        candidates=candidates,
        top_k=FINAL_K,
    )

    print_reranked_results(
        results=reranked_results,
    )