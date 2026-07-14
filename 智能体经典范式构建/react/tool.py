from typing import Dict, Any
from .search import search
class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数。
        """
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])


# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)

    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")

# >> >
# 工具
# 'Search'
# 已注册。
#
# --- 可用的工具 - --
# - Search: 一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。
#
# --- 执行
# Action: Search['英伟达最新的GPU型号是什么'] - --
# 🔍 正在执行[SerpApi]
# 网页搜索: 英伟达最新的GPU型号是什么
# --- 观察(Observation) - --
# [1]
# GeForce
# RTX
# 50
# 系列显卡
# GeForce
# RTX™ 50
# 系列GPU
# 搭载NVIDIA
# Blackwell
# 架构，为游戏玩家和创作者带来全新玩法。RTX
# 50
# 系列具备强大的AI
# 算力，带来升级体验和更逼真的画面。
#
# [2]
# 比较GeForce
# 系列最新一代显卡和前代显卡
# 比较最新一代RTX
# 30
# 系列显卡和前代的RTX
# 20
# 系列、GTX
# 10
# 和900
# 系列显卡。查看规格、功能、技术支持等内容。
#
# [3]
# GeForce
# 显卡 | NVIDIA
# DRIVE
# AGX.强大的车载计算能力，适用于AI
# 驱动的智能汽车系统 · Clara
# AGX.适用于创新型医疗设备和成像的AI
# 计算.游戏和创作.GeForce.探索显卡、游戏解决方案、AI...
