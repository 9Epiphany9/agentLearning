import re
from typing import List, Optional, Tuple

from HelloAgentsLLM import HelloAgentsLLM
from .search import search
from .tool import ToolExecutor


REACT_PROMPT_TEMPLATE = """
You are an assistant that can call external tools.

Available tools:
{tools}

Follow this format strictly:
Thought: explain your reasoning briefly.
Action: use exactly one of these forms:
- ToolName[tool input]
- Finish[final answer]

After receiving an Observation, decide whether another tool call is needed.
When you have enough information, use Finish[final answer].

Question: {question}
History:
{history}
""".strip()


class ReActAgent:
    def __init__(
        self,
        llm_client: HelloAgentsLLM,
        tool_executor: ToolExecutor,
        max_steps: int = 5,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history: List[str] = []

    def run(self, question: str) -> Optional[str]:
        """Run the ReAct loop and return the final answer, if one is produced."""
        if not question.strip():
            raise ValueError("question must not be empty")

        self.history = []

        for step in range(1, self.max_steps + 1):
            print(f"--- Step {step}/{self.max_steps} ---")
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=self.tool_executor.getAvailableTools(),
                question=question,
                history="\n".join(self.history) or "(none)",
            )

            try:
                response_text = self.llm_client.think(
                    messages=[{"role": "user", "content": prompt}]
                )
            except Exception as exc:
                print(f"LLM call failed: {exc}")
                return None

            if not response_text:
                print("LLM returned an empty response.")
                return None

            thought, action = self._parse_output(response_text)
            if thought:
                print(f"Thought: {thought}")

            # Keep the model response so the next iteration can use its reasoning.
            self.history.append(response_text.strip())

            if not action:
                observation = "Could not parse a valid Action from the LLM response."
                print(observation)
                self.history.append(f"Observation: {observation}")
                continue

            finish_match = re.fullmatch(r"Finish\[(.*)\]", action, re.DOTALL)
            if finish_match:
                final_answer = finish_match.group(1).strip()
                print(f"Final answer: {final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                observation = (
                    "Invalid Action. Use ToolName[tool input] or "
                    "Finish[final answer]."
                )
                print(observation)
                self.history.append(f"Observation: {observation}")
                continue

            print(f"Action: {tool_name}[{tool_input}]")
            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"Unknown tool: {tool_name}"
            else:
                try:
                    observation = str(tool_function(tool_input))
                except Exception as exc:
                    observation = f"Tool execution failed: {exc}"

            print(f"Observation: {observation}")
            self.history.append(f"Observation: {observation}")

        print("Maximum number of steps reached without a final answer.")
        return None

    @staticmethod
    def _parse_output(text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract Thought and Action from an LLM response."""
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\n\s*Action:|$)", text, re.IGNORECASE | re.DOTALL
        )
        action_match = re.search(
            r"Action:\s*(.*?)(?=\n\s*(?:Thought|Observation|Action):|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    @staticmethod
    def _parse_action(action_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract a tool name and input from ToolName[tool input]."""
        match = re.fullmatch(r"([A-Za-z_]\w*)\[(.*)\]", action_text.strip(), re.DOTALL)
        if not match:
            return None, None
        return match.group(1), match.group(2).strip()


def main() -> None:
    """Create the default agent and run one example question."""
    try:
        llm_client = HelloAgentsLLM()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return

    tool_executor = ToolExecutor()
    tool_executor.registerTool(
        "Search",
        "Search the web for current information.",
        search,
    )

    agent = ReActAgent(llm_client, tool_executor)
    agent.run("碧蓝之海第三季上限时间是什么?，中文回答我")


if __name__ == "__main__":
    main()
