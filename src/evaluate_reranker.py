from time import perf_counter

from evaluate import TEST_CASES
from reranker import (
    CANDIDATE_K,
    FINAL_K,
    create_reranker,
    rerank_candidates,
    retrieve_candidates,
)
from vectorstore import INDEX_PATH, load_vectorstore


def get_document_ids_from_faiss(results):
    """
    FAISS 结果格式：
    list[tuple[Document, faiss_score]]
    """
    return [
        document.metadata.get(
            "document_id",
            "unknown",
        )
        for document, faiss_score in results
    ]


def get_document_ids_from_reranker(results):
    """
    Reranker 结果格式：
    list[
        tuple[
            Document,
            faiss_score,
            reranker_score,
        ]
    ]
    """
    return [
        document.metadata.get(
            "document_id",
            "unknown",
        )
        for (
            document,
            faiss_score,
            reranker_score,
        ) in results
    ]


def find_correct_rank(
    retrieved_document_ids,
    expected_document_id,
):
    """
    找到正确文档第一次出现的位置。

    返回：
    - 1：正确文档排第一
    - 2：正确文档排第二
    - None：没有进入最终结果
    """
    for rank, document_id in enumerate(
        retrieved_document_ids,
        start=1,
    ):
        if document_id == expected_document_id:
            return rank

    return None


def reciprocal_rank(rank):
    """
    计算单个问题的 Reciprocal Rank。

    Top 1 -> 1.0
    Top 2 -> 0.5
    Top 3 -> 0.333...
    未命中 -> 0.0
    """
    if rank is None:
        return 0.0

    return 1.0 / rank


def evaluate(
    vectorstore,
    reranker,
    test_cases,
    candidate_k,
    final_k,
):
    faiss_hit_at_1_count = 0
    faiss_hit_at_k_count = 0
    faiss_reciprocal_rank_sum = 0.0

    reranker_hit_at_1_count = 0
    reranker_hit_at_k_count = 0
    reranker_reciprocal_rank_sum = 0.0

    faiss_time_sum = 0.0
    reranker_time_sum = 0.0
    pipeline_time_sum = 0.0

    for index, test_case in enumerate(
        test_cases,
        start=1,
    ):
        question = test_case["question"]

        expected_document_id = test_case[
            "expected_document_id"
        ]

        # ---------------------------------
        # 第一阶段：FAISS 召回
        # ---------------------------------
        faiss_start_time = perf_counter()

        candidates = retrieve_candidates(
            vectorstore=vectorstore,
            query=question,
            candidate_k=candidate_k,
        )

        faiss_elapsed_time = (
            perf_counter() - faiss_start_time
        )

        # 普通 FAISS 最终只取前 final_k 条
        faiss_results = candidates[:final_k]

        # ---------------------------------
        # 第二阶段：Reranker 重排序
        # ---------------------------------
        reranker_start_time = perf_counter()

        reranked_results = rerank_candidates(
            reranker=reranker,
            query=question,
            candidates=candidates,
            final_k=final_k,
        )

        reranker_elapsed_time = (
            perf_counter() - reranker_start_time
        )

        pipeline_elapsed_time = (
            faiss_elapsed_time
            + reranker_elapsed_time
        )

        faiss_time_sum += faiss_elapsed_time
        reranker_time_sum += reranker_elapsed_time
        pipeline_time_sum += pipeline_elapsed_time

        # ---------------------------------
        # 提取两种方法的文档排名
        # ---------------------------------
        faiss_document_ids = (
            get_document_ids_from_faiss(
                faiss_results
            )
        )

        reranker_document_ids = (
            get_document_ids_from_reranker(
                reranked_results
            )
        )

        faiss_correct_rank = find_correct_rank(
            retrieved_document_ids=(
                faiss_document_ids
            ),
            expected_document_id=(
                expected_document_id
            ),
        )

        reranker_correct_rank = find_correct_rank(
            retrieved_document_ids=(
                reranker_document_ids
            ),
            expected_document_id=(
                expected_document_id
            ),
        )

        # ---------------------------------
        # 更新 FAISS 指标
        # ---------------------------------
        if faiss_correct_rank == 1:
            faiss_hit_at_1_count += 1

        if faiss_correct_rank is not None:
            faiss_hit_at_k_count += 1

        faiss_reciprocal_rank_sum += (
            reciprocal_rank(
                faiss_correct_rank
            )
        )

        # ---------------------------------
        # 更新 Reranker 指标
        # ---------------------------------
        if reranker_correct_rank == 1:
            reranker_hit_at_1_count += 1

        if reranker_correct_rank is not None:
            reranker_hit_at_k_count += 1

        reranker_reciprocal_rank_sum += (
            reciprocal_rank(
                reranker_correct_rank
            )
        )

        # ---------------------------------
        # 打印当前问题的对比
        # ---------------------------------
        print(
            f"\n========== 测试问题 {index} =========="
        )
        print(f"问题：{question}")
        print(
            f"正确文档：{expected_document_id}"
        )

        print("\nFAISS：")
        print(
            f"检索文档：{faiss_document_ids}"
        )
        print(
            f"正确文档排名：{faiss_correct_rank}"
        )
        print(
            f"耗时："
            f"{faiss_elapsed_time * 1000:.2f} ms"
        )

        print("\nFAISS + Reranker：")
        print(
            f"检索文档：{reranker_document_ids}"
        )
        print(
            f"正确文档排名："
            f"{reranker_correct_rank}"
        )
        print(
            f"Reranker 耗时："
            f"{reranker_elapsed_time * 1000:.2f} ms"
        )
        print(
            f"完整检索耗时："
            f"{pipeline_elapsed_time * 1000:.2f} ms"
        )

        # 标记排名发生了什么变化
        if (
            faiss_correct_rank is not None
            and reranker_correct_rank is not None
        ):
            if (
                reranker_correct_rank
                < faiss_correct_rank
            ):
                print("结果：Reranker 提升了正确排名")

            elif (
                reranker_correct_rank
                > faiss_correct_rank
            ):
                print("结果：Reranker 降低了正确排名")

            else:
                print("结果：正确排名保持不变")

        elif (
            faiss_correct_rank is None
            and reranker_correct_rank is not None
        ):
            print(
                "结果：Reranker 将正确结果排入最终 Top K"
            )

        elif (
            faiss_correct_rank is not None
            and reranker_correct_rank is None
        ):
            print(
                "结果：Reranker 将正确结果移出了最终 Top K"
            )

        else:
            print("结果：两种方法都没有命中")

    # ---------------------------------
    # 计算总体指标
    # ---------------------------------
    total = len(test_cases)

    faiss_hit_at_1 = (
        faiss_hit_at_1_count / total
    )

    faiss_hit_at_k = (
        faiss_hit_at_k_count / total
    )

    faiss_mrr = (
        faiss_reciprocal_rank_sum / total
    )

    reranker_hit_at_1 = (
        reranker_hit_at_1_count / total
    )

    reranker_hit_at_k = (
        reranker_hit_at_k_count / total
    )

    reranker_mrr = (
        reranker_reciprocal_rank_sum / total
    )

    average_faiss_time = (
        faiss_time_sum / total
    )

    average_reranker_time = (
        reranker_time_sum / total
    )

    average_pipeline_time = (
        pipeline_time_sum / total
    )

    print("\n" + "=" * 50)
    print("总体评估")
    print("=" * 50)

    print("\n普通 FAISS：")
    print(
        f"Hit@1："
        f"{faiss_hit_at_1_count}/{total} "
        f"= {faiss_hit_at_1:.1%}"
    )
    print(
        f"Hit@{final_k}："
        f"{faiss_hit_at_k_count}/{total} "
        f"= {faiss_hit_at_k:.1%}"
    )
    print(f"MRR：{faiss_mrr:.4f}")
    print(
        f"平均耗时："
        f"{average_faiss_time * 1000:.2f} ms"
    )

    print("\nFAISS + Reranker：")
    print(
        f"Hit@1："
        f"{reranker_hit_at_1_count}/{total} "
        f"= {reranker_hit_at_1:.1%}"
    )
    print(
        f"Hit@{final_k}："
        f"{reranker_hit_at_k_count}/{total} "
        f"= {reranker_hit_at_k:.1%}"
    )
    print(f"MRR：{reranker_mrr:.4f}")
    print(
        f"平均 Reranker 耗时："
        f"{average_reranker_time * 1000:.2f} ms"
    )
    print(
        f"平均完整检索耗时："
        f"{average_pipeline_time * 1000:.2f} ms"
    )

    print("\n指标变化：")
    print(
        f"Hit@1 变化："
        f"{reranker_hit_at_1 - faiss_hit_at_1:+.1%}"
    )
    print(
        f"Hit@{final_k} 变化："
        f"{reranker_hit_at_k - faiss_hit_at_k:+.1%}"
    )
    print(
        f"MRR 变化："
        f"{reranker_mrr - faiss_mrr:+.4f}"
    )


if __name__ == "__main__":
    print("正在加载 FAISS...")
    vectorstore = load_vectorstore(INDEX_PATH)

    print("正在加载 Reranker...")
    reranker = create_reranker()

    print(
        f"FAISS 向量数量："
        f"{vectorstore.index.ntotal}"
    )
    print(
        f"FAISS 候选数量：{CANDIDATE_K}"
    )
    print(
        f"最终结果数量：{FINAL_K}"
    )

    evaluate(
        vectorstore=vectorstore,
        reranker=reranker,
        test_cases=TEST_CASES,
        candidate_k=CANDIDATE_K,
        final_k=FINAL_K,
    )