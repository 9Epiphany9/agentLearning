from HelloAgentsLLM import HelloAgentsLLM
from plan_and_resolve import Planner
from react import ReActAgent, ToolExecutor, search
from reflection import ReflectionAgent


def main() -> None:
    llm_client = HelloAgentsLLM()
    question = "如何练习男伪女伪音？已能发出啊的女声，其他的字说的不像，说不了别的字和话"

    # 1. 生成计划
    plan = Planner(llm_client).plan(question)
    if not plan:
        return

    plan_text = "\n".join(
        f"{i}. {step}" for i, step in enumerate(plan, start=1)
    )

    # 2. 交给 ReAct 执行
    tool_executor = ToolExecutor()
    tool_executor.registerTool(
        "Search",
        "Search the web for current information.",
        search,
    )

    react_task = f"""
原始任务：
{question}

请按照以下计划完成任务：
{plan_text}
"""

    draft = ReActAgent(llm_client, tool_executor).run(react_task)
    if not draft:
        print("ReAct 执行失败")
        return

    # 3. 交给 Reflection 优化
    final_answer = ReflectionAgent(
        llm_client,
        max_iterations=2,
    ).run(
        task=question,
        draft=draft,
    )

    print("\n最终结果：")
    print(final_answer)


if __name__ == "__main__":
    main()