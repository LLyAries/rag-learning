from vectorstore import INDEX_PATH, load_vectorstore


TOP_K = 5


TEST_CASES = [
    # FAISS GPU 论文
    {
        "question": (
            "How do GPUs accelerate "
            "large-scale similarity search?"
        ),
        "expected_document_id": "faiss_gpu",
    },
    {
        "question": (
            "What GPU implementation is used "
            "for fast nearest-neighbor search?"
        ),
        "expected_document_id": "faiss_gpu",
    },
    {
        "question": (
            "How can similarity search scale "
            "to billions of vectors?"
        ),
        "expected_document_id": "faiss_gpu",
    },

    # Sentence-BERT 论文
    {
        "question": (
            "How does Sentence-BERT produce "
            "sentence embeddings?"
        ),
        "expected_document_id": "sentence_bert",
    },
    {
        "question": (
            "Why are siamese and triplet "
            "network structures used?"
        ),
        "expected_document_id": "sentence_bert",
    },
    {
        "question": (
            "How does Sentence-BERT reduce "
            "the cost of semantic similarity search?"
        ),
        "expected_document_id": "sentence_bert",
    },

    # 原来的 RAG 教学 PDF
    {
        "question": (
            "Why are documents divided into "
            "overlapping chunks?"
        ),
        "expected_document_id": "test",
    },
    {
        "question": (
            "What role does FAISS play "
            "in a simple RAG system?"
        ),
        "expected_document_id": "test",
    },
]


def evaluate(vectorstore, test_cases, k):
    hit_at_1_count = 0
    hit_at_k_count = 0
    reciprocal_rank_sum = 0.0

    for index, test_case in enumerate(
        test_cases,
        start=1,
    ):
        question = test_case["question"]
        expected_document_id = test_case[
            "expected_document_id"
        ]

        results = vectorstore.similarity_search_with_score(
            query=question,
            k=k,
        )

        retrieved_document_ids = [
            document.metadata.get(
                "document_id",
                "unknown",
            )
            for document, score in results
        ]

        hit_at_1 = (
            len(retrieved_document_ids) > 0
            and retrieved_document_ids[0]
            == expected_document_id
        )

        hit_at_k = (
            expected_document_id
            in retrieved_document_ids
        )

        correct_rank = None

        for rank, document_id in enumerate(
            retrieved_document_ids,
            start=1,
        ):
            if document_id == expected_document_id:
                correct_rank = rank
                break

        if hit_at_1:
            hit_at_1_count += 1

        if hit_at_k:
            hit_at_k_count += 1

        if correct_rank is not None:
            reciprocal_rank_sum += 1 / correct_rank

        print(
            f"\n========== 测试问题 {index} =========="
        )
        print(f"问题：{question}")
        print(
            f"正确文档：{expected_document_id}"
        )
        print(
            f"检索文档：{retrieved_document_ids}"
        )
        print(
            f"Hit@1："
            f"{'命中' if hit_at_1 else '未命中'}"
        )
        print(
            f"Hit@{k}："
            f"{'命中' if hit_at_k else '未命中'}"
        )
        print(
            f"正确结果首次排名："
            f"{correct_rank}"
        )

        print("\n检索详情：")

        for rank, (document, score) in enumerate(
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
            )[:120]

            print(
                f"Top {rank} | "
                f"文件={file_name} | "
                f"页码={page_number} | "
                f"Chunk={chunk_id} | "
                f"距离={score:.4f}"
            )
            print(f"文本预览：{preview}")

    total = len(test_cases)

    hit_at_1_rate = hit_at_1_count / total
    hit_at_k_rate = hit_at_k_count / total
    mrr = reciprocal_rank_sum / total

    print("\n========== 总体评估 ==========")
    print(f"问题数量：{total}")

    print(
        f"Hit@1：{hit_at_1_count}/{total} "
        f"= {hit_at_1_rate:.1%}"
    )

    print(
        f"Hit@{k}：{hit_at_k_count}/{total} "
        f"= {hit_at_k_rate:.1%}"
    )

    print(f"MRR：{mrr:.4f}")


if __name__ == "__main__":
    vectorstore = load_vectorstore(INDEX_PATH)

    print(
        f"已加载向量数量："
        f"{vectorstore.index.ntotal}"
    )

    evaluate(
        vectorstore=vectorstore,
        test_cases=TEST_CASES,
        k=TOP_K,
    )