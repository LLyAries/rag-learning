from vectorstore import INDEX_PATH, load_vectorstore


QUERIES = [
    "Why do chunks overlap?",
    (
        "Explain document loading, chunking, "
        "embeddings, and vector retrieval."
    ),
]

K = 3
FETCH_K = 10

LAMBDA_VALUES = [
    0.2,
    0.5,
    0.8,
    1.0,
]


def get_page(document):
    return document.metadata.get(
        "page",
        -1,
    ) + 1


def get_preview(document, length=140):
    return document.page_content.replace(
        "\n",
        " ",
    )[:length]


def run_similarity_search(
    vectorstore,
    query,
):
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=K,
    )

    print("\n普通相似度检索：")

    for rank, (document, score) in enumerate(
        results,
        start=1,
    ):
        print(
            f"Top {rank} | "
            f"页码={get_page(document)} | "
            f"距离={score:.4f}"
        )
        print(f"文本：{get_preview(document)}")


def run_mmr_experiments(
    vectorstore,
    query,
):
    for lambda_mult in LAMBDA_VALUES:
        documents = (
            vectorstore.max_marginal_relevance_search(
                query=query,
                k=K,
                fetch_k=FETCH_K,
                lambda_mult=lambda_mult,
            )
        )

        print(
            "\n"
            f"MMR：lambda_mult={lambda_mult}"
        )

        for rank, document in enumerate(
            documents,
            start=1,
        ):
            print(
                f"结果 {rank} | "
                f"页码={get_page(document)}"
            )
            print(f"文本：{get_preview(document)}")


if __name__ == "__main__":
    vectorstore = load_vectorstore(INDEX_PATH)

    for query_index, query in enumerate(
        QUERIES,
        start=1,
    ):
        print(
            "\n\n"
            "########################################"
        )
        print(f"测试问题 {query_index}：{query}")
        print(
            "########################################"
        )

        run_similarity_search(
            vectorstore=vectorstore,
            query=query,
        )

        run_mmr_experiments(
            vectorstore=vectorstore,
            query=query,
        )