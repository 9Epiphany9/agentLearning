from HelloAgentsLLM import HelloAgentsLLM
from .executor import Executor
from .plan import Planner
class PlanAndSolveAgent:
    def __init__(self, llm_client):
        """
        初始化智能体，同时创建规划器和执行器实例。
        """
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self, question: str):
        """
        运行智能体的完整流程:先规划，后执行。
        """
        print(f"\n--- 开始处理问题 ---\n问题: {question}")

        # 1. 调用规划器生成计划
        plan = self.planner.plan(question)

        # 检查计划是否成功生成
        if not plan:
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return

        # 2. 调用执行器执行计划
        final_answer = self.executor.execute(question, plan)

        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

def main() -> None:
    """Create the default agent and run one example question."""
    try:
        llm_client = HelloAgentsLLM()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return

    planner = Planner(llm_client)
    question = '如何练习男伪女伪音'
    plan = planner.plan(question)
    plan_executor = Executor(llm_client)
    print(plan_executor.execute(question, plan))

if __name__ == "__main__":
    main()
