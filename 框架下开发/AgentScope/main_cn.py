# -*- coding: utf-8 -*-
"""
三国狼人杀 - 基于AgentScope的中文版狼人杀游戏
融合三国演义角色和传统狼人杀玩法
"""
import asyncio
import os
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, List, Dict, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from agentscope.agent import AgentBase, ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.pipeline import MsgHub, sequential_pipeline, fanout_pipeline
from agentscope.formatter import (
    DeepSeekMultiAgentFormatter,
    FormatterBase,
    OpenAIMultiAgentFormatter,
)

from prompt_cn import ChinesePrompts
from game_roles import GameRoles
from structured_output_cn import (
    get_vote_model_cn,
    WitchActionModelCN,
    get_seer_model_cn,
    get_hunter_model_cn,
    WerewolfKillModelCN
)
from utils_cn import (
    check_winning_cn,
    majority_vote_cn,
    get_chinese_name,
    get_agent_name,
    format_player_list,
    GameModerator,
    MAX_GAME_ROUND,
    MAX_DISCUSSION_ROUND,
)


ENV_FILE = Path(__file__).resolve().with_name(".env")
AI_ENV_VARS = ("LLM_MODEL_ID", "LLM_API_KEY", "LLM_BASE_URL")


def configure_console_encoding() -> None:
    """确保 Windows 控制台能够输出游戏中的中文和 emoji。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def load_ai_config() -> Dict[str, str]:
    """从项目根目录的 .env 加载 OpenAI 兼容模型配置。"""
    load_dotenv(ENV_FILE)
    config = {name: os.getenv(name, "").strip() for name in AI_ENV_VARS}
    missing = [name for name, value in config.items() if not value]
    if missing:
        raise RuntimeError(
            f"AI 配置缺失：{', '.join(missing)}。请在 {ENV_FILE} 中配置。"
        )
    return config


def create_multi_agent_formatter(base_url: str) -> FormatterBase:
    """根据兼容接口的服务商选择对应的多人消息格式。"""
    hostname = (urlparse(base_url).hostname or "").lower()
    if hostname == "deepseek.com" or hostname.endswith(".deepseek.com"):
        return DeepSeekMultiAgentFormatter()
    return OpenAIMultiAgentFormatter()


class GameChatModel(OpenAIChatModel):
    """让 ReActAgent 在游戏中优先调用完成工具，而不是只输出解释文字。"""

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        tools = kwargs.get("tools")
        if tools and kwargs.get("tool_choice") is None:
            has_finish_tool = any(
                tool.get("function", {}).get("name") == "generate_response"
                for tool in tools
            )
            if has_finish_tool:
                kwargs["tool_choice"] = "generate_response"
        return await super().__call__(*args, **kwargs)


class GameReActAgent(ReActAgent):
    """清理上一次行动的结构化模型，避免影响后续普通讨论。"""

    async def reply(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("structured_model") is None:
            self.toolkit.set_extended_model(self.finish_function_name, None)
        return await super().reply(*args, **kwargs)


class ThreeKingdomsWerewolfGame:
    """三国狼人杀游戏主类"""
    
    def __init__(self, ai_config: Optional[Dict[str, str]] = None):
        configure_console_encoding()
        self.ai_config = ai_config or load_ai_config()
        self.players: Dict[str, ReActAgent] = {}
        self.roles: Dict[str, str] = {}
        self.moderator = GameModerator()
        self.alive_players: List[AgentBase] = []
        self.werewolves: List[AgentBase] = []
        self.villagers: List[AgentBase] = []
        self.seer: List[AgentBase] = []
        self.witch: List[AgentBase] = []
        self.hunter: List[AgentBase] = []
        
        # 女巫道具状态
        self.witch_has_antidote = True
        self.witch_has_poison = True
        
    async def create_player(self, role: str, character: str) -> ReActAgent:
        """创建具有三国背景的玩家"""
        name = get_chinese_name(character)
        self.roles[name] = role
        
        agent = GameReActAgent(
            name=name,
            sys_prompt=ChinesePrompts.get_role_prompt(role, character),
            model=GameChatModel(
                model_name=self.ai_config["LLM_MODEL_ID"],
                api_key=self.ai_config["LLM_API_KEY"],
                stream=False,
                client_args={
                    "base_url": self.ai_config["LLM_BASE_URL"],
                },
            ),
            formatter=create_multi_agent_formatter(
                self.ai_config["LLM_BASE_URL"],
            ),
        )
        
        # 角色身份确认
        await agent.observe(
            await self.moderator.announce(
                f"【{name}】你在这场三国狼人杀中扮演{GameRoles.get_role_desc(role)}，"
                f"你的角色是{character}。{GameRoles.get_role_ability(role)}"
            )
        )
        
        self.players[name] = agent
        return agent
    
    async def setup_game(self, player_count: int = 6):
        """设置游戏"""
        print("🎮 开始设置三国狼人杀游戏...")
        
        # 获取角色配置
        roles = GameRoles.get_standard_setup(player_count)
        characters = random.sample([
            "刘备", "关羽", "张飞", "诸葛亮", "赵云",
            "曹操", "司马懿", "周瑜", "孙权"
        ], player_count)
        
        # 创建玩家
        for i, (role, character) in enumerate(zip(roles, characters)):
            agent = await self.create_player(role, character)
            self.alive_players.append(agent)
            
            # 分配到对应阵营
            if role == "狼人":
                self.werewolves.append(agent)
            elif role == "预言家":
                self.seer.append(agent)
            elif role == "女巫":
                self.witch.append(agent)
            elif role == "猎人":
                self.hunter.append(agent)
            else:
                self.villagers.append(agent)
        
        # 游戏开始公告
        await self.moderator.announce(
            f"三国狼人杀游戏开始！参与者：{format_player_list(self.alive_players)}"
        )
        
        print(f"✅ 游戏设置完成，共{len(self.alive_players)}名玩家")

    def choose_alive_target(
        self,
        excluded_names: Optional[set[str]] = None,
    ) -> Optional[str]:
        """从存活玩家中随机选择一个合法目标。"""
        excluded_names = excluded_names or set()
        candidates = [
            get_agent_name(player)
            for player in self.alive_players
            if get_agent_name(player) not in excluded_names
        ]
        return random.choice(candidates) if candidates else None
    
    async def werewolf_phase(self, round_num: int) -> Optional[str]:
        """狼人阶段"""
        if not self.werewolves:
            return None
            
        await self.moderator.announce(f"🐺 狼人请睁眼，选择今晚要击杀的目标...")

        # 狼人阵营的身份是公开给队友的，但不能通过主持人公告泄露给好人。
        for wolf in self.werewolves:
            teammates = [
                get_agent_name(other)
                for other in self.werewolves
                if other is not wolf
            ]
            await wolf.observe(
                await self.moderator.private_announce(
                    f"你的狼人队友是：{'、'.join(teammates) if teammates else '无'}。"
                    "你们属于同一狼人阵营，可以在本阶段直接协商。"
                )
            )
        
        # 狼人讨论
        async with MsgHub(
            self.werewolves,
            enable_auto_broadcast=True,
            announcement=await self.moderator.announce(
                f"狼人们，请讨论今晚的击杀目标。存活玩家：{format_player_list(self.alive_players)}"
            ),
        ) as werewolves_hub:
            # 讨论阶段
            for _ in range(MAX_DISCUSSION_ROUND):
                for wolf in self.werewolves:
                    await wolf()
            
            # 投票击杀
            werewolves_hub.set_auto_broadcast(False)
            kill_votes = await fanout_pipeline(
                self.werewolves,
                msg=await self.moderator.announce("请选择击杀目标"),
                structured_model=WerewolfKillModelCN,
                enable_gather=False,
            )
            
            # 统计投票
            votes: Dict[str, Optional[str]] = {}
            werewolf_names = {
                get_agent_name(wolf) for wolf in self.werewolves
            }
            valid_targets = [
                get_agent_name(player)
                for player in self.alive_players
                if get_agent_name(player) not in werewolf_names
            ]
            for i, vote_msg in enumerate(kill_votes):
                wolf_name = get_agent_name(self.werewolves[i])
                # 检查vote_msg是否为None或metadata是否存在
                target: Optional[str] = None
                if vote_msg is not None and hasattr(vote_msg, 'metadata') and vote_msg.metadata is not None:
                    raw_target = vote_msg.metadata.get("target")
                    target = raw_target if isinstance(raw_target, str) else None

                if target not in valid_targets:
                    # 模型没有提交合法目标时，使用规则层兜底，保证夜晚能继续。
                    print(f"⚠️ {wolf_name} 的击杀投票无效,随机选择目标")
                    target = random.choice(valid_targets) if valid_targets else None
                votes[wolf_name] = target
            
            killed_player, _ = majority_vote_cn(votes)
            return killed_player
    
    async def seer_phase(self):
        """预言家阶段"""
        if not self.seer:
            return
            
        seer_agent = self.seer[0]
        seer_notice = await self.moderator.announce(
            "🔮 预言家请睁眼，选择要查验的玩家..."
        )
        await seer_agent.observe(seer_notice)
        
        check_result = await seer_agent(
            structured_model=get_seer_model_cn(self.alive_players)
        )

        # 检查返回结果是否有效
        if check_result is None or not hasattr(check_result, 'metadata') or check_result.metadata is None:
            print(f"⚠️ 预言家查验失败,跳过此阶段")
            return

        raw_target = check_result.metadata.get("target")
        target_name = raw_target if isinstance(raw_target, str) else None
        valid_targets = {
            get_agent_name(player)
            for player in self.alive_players
            if player is not seer_agent
        }
        if target_name not in valid_targets:
            print("⚠️ 预言家未选择合法查验目标,随机选择目标")
            target_name = self.choose_alive_target(
                {get_agent_name(seer_agent)}
            )
            if target_name is None:
                return

        target_role = self.roles.get(target_name, "村民")
        
        # 告知预言家结果
        result_msg = f"查验结果：{target_name}是{'狼人' if target_role == '狼人' else '好人'}"
        await seer_agent.observe(await self.moderator.announce(result_msg))
    
    async def witch_phase(
        self,
        killed_player: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """女巫阶段"""
        if not self.witch:
            return killed_player, None
            
        witch_agent = self.witch[0]
        witch_notice = await self.moderator.announce("🧙‍♀️ 女巫请睁眼...")
        await witch_agent.observe(witch_notice)
        
        # 告知女巫死亡信息
        death_info = f"今晚{killed_player}被狼人击杀" if killed_player else "今晚平安无事"
        await witch_agent.observe(await self.moderator.announce(death_info))
        
        # 女巫行动
        witch_action = await witch_agent(structured_model=WitchActionModelCN)

        saved_player: Optional[str] = None
        poisoned_player: Optional[str] = None

        # 检查返回结果是否有效
        if witch_action is None or not hasattr(witch_action, 'metadata') or witch_action.metadata is None:
            print(f"⚠️ 女巫行动失败,视为不使用技能")
        else:
            if witch_action.metadata.get("use_antidote") and self.witch_has_antidote:
                if killed_player:
                    saved_player = killed_player
                    self.witch_has_antidote = False
                    await witch_agent.observe(await self.moderator.announce(f"你使用解药救了{killed_player}"))

            if witch_action.metadata.get("use_poison") and self.witch_has_poison:
                target_name = witch_action.metadata.get("target_name")
                if isinstance(target_name, str) and target_name:
                    poisoned_player = target_name
                    self.witch_has_poison = False
                    await witch_agent.observe(await self.moderator.announce(f"你使用毒药毒杀了{poisoned_player}"))
        
        # 确定最终死亡玩家
        final_killed = killed_player if not saved_player else None
        
        return final_killed, poisoned_player
    
    async def hunter_phase(
        self,
        shot_by_hunter: Optional[str],
    ) -> Optional[str]:
        """猎人阶段"""
        if not self.hunter:
            return None
            
        hunter_agent = self.hunter[0]
        hunter_name = get_agent_name(hunter_agent)
        if hunter_name == shot_by_hunter:
            hunter_notice = await self.moderator.announce(
                "🏹 猎人发动技能，可以带走一名玩家..."
            )
            await hunter_agent.observe(hunter_notice)
            
            hunter_action = await hunter_agent(
                structured_model=get_hunter_model_cn(self.alive_players)
            )

            # 检查返回结果是否有效
            if hunter_action is None or not hasattr(hunter_action, 'metadata') or hunter_action.metadata is None:
                print(f"⚠️ 猎人技能使用失败,视为放弃开枪")
                return None

            if hunter_action.metadata.get("shoot"):
                target = hunter_action.metadata.get("target")
                if isinstance(target, str) and target:
                    await self.moderator.announce(f"猎人{hunter_name}开枪带走了{target}")
                    return target
                else:
                    print(f"⚠️ 猎人选择开枪但未指定目标,视为放弃")
                    return None
        
        return None
    
    def update_alive_players(
        self,
        dead_players: Sequence[Optional[str]],
    ) -> None:
        """更新存活玩家列表"""
        for dead_name in dead_players:
            if dead_name:
                # 从存活列表移除
                self.alive_players = [
                    p for p in self.alive_players
                    if get_agent_name(p) != dead_name
                ]
                # 从各阵营移除
                self.werewolves = [p for p in self.werewolves if get_agent_name(p) != dead_name]
                self.villagers = [p for p in self.villagers if get_agent_name(p) != dead_name]
                self.seer = [p for p in self.seer if get_agent_name(p) != dead_name]
                self.witch = [p for p in self.witch if get_agent_name(p) != dead_name]
                self.hunter = [p for p in self.hunter if get_agent_name(p) != dead_name]
    
    async def day_phase(self, round_num: int):
        """白天阶段"""
        await self.moderator.day_announcement(round_num)
        
        # 讨论阶段
        async with MsgHub(
            self.alive_players,
            enable_auto_broadcast=True,
            announcement=await self.moderator.announce(
                f"现在开始自由讨论。存活玩家：{format_player_list(self.alive_players)}"
            ),
        ) as all_hub:
            # 每人发言一轮
            await sequential_pipeline(self.alive_players)
            
            # 投票阶段
            all_hub.set_auto_broadcast(False)
            vote_msgs = await fanout_pipeline(
                self.alive_players,
                await self.moderator.announce("请投票选择要淘汰的玩家"),
                structured_model=get_vote_model_cn(self.alive_players),
                enable_gather=False,
            )
            
            # 统计投票
            votes: Dict[str, Optional[str]] = {}
            for i, vote_msg in enumerate(vote_msgs):
                player_name = get_agent_name(self.alive_players[i])
                # 检查vote_msg是否为None或metadata是否存在
                vote: Optional[str] = None
                if vote_msg is not None and hasattr(vote_msg, 'metadata') and vote_msg.metadata is not None:
                    raw_vote = vote_msg.metadata.get("vote")
                    vote = raw_vote if isinstance(raw_vote, str) else None

                valid_targets = {
                    get_agent_name(player)
                    for player in self.alive_players
                    if get_agent_name(player) != player_name
                }
                if vote not in valid_targets:
                    print(f"⚠️ {player_name} 的投票无效,随机选择合法目标")
                    vote = self.choose_alive_target({player_name})
                votes[player_name] = vote
            
            voted_out, vote_count = majority_vote_cn(votes)
            await self.moderator.vote_result_announcement(voted_out, vote_count)
            
            return voted_out
    
    async def run_game(self):
        """运行游戏主循环"""
        try:
            await self.setup_game()
            
            for round_num in range(1, MAX_GAME_ROUND + 1):
                print(f"\n🌙 === 第{round_num}轮游戏开始 ===")
                
                # 夜晚阶段
                await self.moderator.night_announcement(round_num)
                
                # 狼人击杀
                killed_player = await self.werewolf_phase(round_num)
                
                # 预言家查验
                await self.seer_phase()
                
                # 女巫行动
                final_killed, poisoned_player = await self.witch_phase(killed_player)
                
                # 更新死亡玩家
                night_deaths = [p for p in [final_killed, poisoned_player] if p]
                self.update_alive_players(night_deaths)
                
                # 死亡公告
                await self.moderator.death_announcement(night_deaths)
                
                # 检查胜利条件
                winner = check_winning_cn(self.alive_players, self.roles)
                if winner:
                    await self.moderator.game_over_announcement(winner)
                    return
                
                # 白天阶段
                voted_out = await self.day_phase(round_num)
                
                # 猎人技能
                hunter_shot = await self.hunter_phase(voted_out)
                
                # 更新死亡玩家
                day_deaths = [p for p in [voted_out, hunter_shot] if p]
                self.update_alive_players(day_deaths)
                
                # 检查胜利条件
                winner = check_winning_cn(self.alive_players, self.roles)
                if winner:
                    await self.moderator.game_over_announcement(winner)
                    return
                
                print(f"第{round_num}轮结束，存活玩家：{format_player_list(self.alive_players)}")
        
        except Exception as e:
            print(f"❌ 游戏运行出错：{e}")
            import traceback
            traceback.print_exc()


async def main():
    """主函数"""
    configure_console_encoding()
    try:
        ai_config = load_ai_config()
    except RuntimeError as error:
        print(f"❌ {error}")
        return
    
    print("🎮 欢迎来到三国狼人杀！")
    
    # 创建并运行游戏
    game = ThreeKingdomsWerewolfGame(ai_config=ai_config)
    await game.run_game()


if __name__ == "__main__":
    asyncio.run(main())
