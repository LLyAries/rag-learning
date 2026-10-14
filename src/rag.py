from llm import create_llm
from vectorstore import INDEX_PATH, load_vectorstore


TOP_K = 5


def retrieve(vectorstore, query, k):
    """
    从 FAISS 中检索 Top-K chunks。

    输入：
    - vectorstore：FAISS 向量库
    - query：用户问题
    - k：返回结果数量

    输出：
    - list[tuple[Document, float]]
    """
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=k,
    )

    return results


def build_context(results):
    """
    将检索结果转换成大模型能够阅读的上下文。

    每个 chunk 都包含：
    - 来源编号
    - 文件名
    - 页码
    - Chunk ID
    - 原文内容
    """
    context_parts = []

    for source_number, (
        document,
        score,
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

        chunk_text = (
            document.page_content.strip()
        )

        context_part = (
            f"[来源{source_number}]\n"
            f"文件名：{file_name}\n"
            f"页码：{page_number}\n"
            f"Chunk ID：{chunk_id}\n"
            f"内容：\n{chunk_text}"
        )

        context_parts.append(context_part)

    context = "\n\n".join(context_parts)

    return context


def generate_answer(llm, query, context):
    """
    将问题和检索上下文发送给大模型。
    """
    system_prompt = """
你是一个严谨的文档问答助手。

请严格遵守以下规则：

1. 只能根据用户提供的检索上下文回答。
2. 不要使用检索上下文以外的知识。
3. 如果上下文没有提供回答问题所需的信息，
   必须回答：
   “根据当前检索到的资料，无法回答这个问题。”
4. 不要因为用户问题中出现了某些名词，
   就假设上下文包含答案。
5. 回答中的关键结论必须使用
   [来源1]、[来源2] 这样的格式标注来源。
6. 只能引用实际出现在检索上下文中的来源编号。
7. 不得编造文件名、页码、数据或事实。
8. 使用中文回答。
9. 回答应当清晰、直接，避免无关内容。
""".strip()

    human_prompt = f"""
用户问题：
{query}

检索上下文：
{context}

请判断检索上下文是否足以回答问题。

如果足够，请根据上下文回答并标注来源。
如果不足，请按照系统要求明确拒绝回答。
""".strip()

    response = llm.invoke(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )

    return response.content


def print_retrieval_results(results):
    """
    显示发送给大模型的检索结果。
    """
    print("\n========== 检索结果 ==========")

    for source_number, (
        document,
        score,
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
        )[:160]

        print(f"\n[来源{source_number}]")
        print(f"文件名：{file_name}")
        print(f"页码：{page_number}")
        print(f"Chunk ID：{chunk_id}")
        print(f"FAISS 距离：{score:.4f}")
        print(f"文本预览：{preview}")


def print_source_list(results):
    """
    按原始来源编号打印全部来源。

    不删除同一页的不同 chunks，
    保证回答中的 [来源N] 可以找到对应记录。
    """
    print("\n========== 来源列表 ==========")

    for source_number, (
        document,
        score,
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

        print(
            f"[来源{source_number}] "
            f"{file_name}，"
            f"第 {page_number} 页，"
            f"Chunk {chunk_id}，"
            f"距离 {score:.4f}"
        )


def answer_question(
    vectorstore,
    llm,
    query,
    top_k,
):
    """
    执行一次完整的 RAG 问答。

    流程：
    1. FAISS 检索
    2. 构造上下文
    3. 调用大模型
    4. 返回答案和来源
    """
    results = retrieve(
        vectorstore=vectorstore,
        query=query,
        k=top_k,
    )

    print_retrieval_results(results)

    context = build_context(results)

    print("\n正在生成答案...")

    answer = generate_answer(
        llm=llm,
        query=query,
        context=context,
    )

    return answer, results


if __name__ == "__main__":
    print("正在加载 FAISS...")
    vectorstore = load_vectorstore(INDEX_PATH)

    print("正在创建大模型客户端...")
    llm = create_llm()

    print("\nRAG 系统已启动。")
    print("输入问题后按回车。")
    print("输入 exit 或 quit 退出。")

    while True:
        try:
            query = input(
                "\n请输入问题："
            ).strip()

        except (KeyboardInterrupt, EOFError):
            print("\nRAG 系统已退出。")
            break

        if not query:
            print("问题不能为空。")
            continue

        if query.lower() in {
            "exit",
            "quit",
        }:
            print("RAG 系统已退出。")
            break

        try:
            answer, results = answer_question(
                vectorstore=vectorstore,
                llm=llm,
                query=query,
                top_k=TOP_K,
            )

        except Exception as error:
            print("\n处理问题失败：")
            print(error)
            continue

        print("\n========== RAG 回答 ==========")
        print(answer)

        print_source_list(results)