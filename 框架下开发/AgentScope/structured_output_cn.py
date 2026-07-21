# -*- coding: utf-8 -*-
"""三国狼人杀游戏的结构化输出模型"""
from collections.abc import Sequence
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
from agentscope.agent import AgentBase


class DiscussionModelCN(BaseModel):
    """中文版讨论输出格式"""
    
    reach_agreement: bool = Field(
        description="是否已达成一致意见",
        default=False,
    )
    confidence_level: int = Field(
        description="对当前推理的信心程度(1-10)",
        ge=1, le=10,
        default=5,
    )
    key_evidence: Optional[str] = Field(
        description="支持你观点的关键证据",
        default=None
    )


def _agent_names(agents: Sequence[AgentBase]) -> tuple[str, ...]:
    """提取可用于结构化输出约束的智能体名称。"""
    return tuple(str(getattr(agent, "name", "")) for agent in agents)


def get_vote_model_cn(agents: Sequence[AgentBase]) -> type[BaseModel]:
    """获取中文版投票模型"""
    agent_names = _agent_names(agents)
    
    class VoteModelCN(BaseModel):
        """中文版投票输出格式"""
        
        vote: Optional[str] = Field(
            description="你要投票淘汰的玩家姓名",
            json_schema_extra={"enum": list(agent_names)},
            default=None,
        )
        reason: Optional[str] = Field(
            description="投票理由，简要说明为什么选择此人",
            default=None,
        )
        suspicion_level: Optional[int] = Field(
            description="对被投票者的怀疑程度(1-10)",
            ge=1, le=10,
            default=None,
        )

        @field_validator("vote")
        @classmethod
        def validate_vote(cls, value: Optional[str]) -> Optional[str]:
            if value is not None and value not in agent_names:
                raise ValueError(f"投票目标必须是：{', '.join(agent_names)}")
            return value
    
    return VoteModelCN


class WitchActionModelCN(BaseModel):
    """中文版女巫行动模型"""
    
    use_antidote: bool = Field(
        description="是否使用解药救人",
        default=False
    )
    use_poison: bool = Field(
        description="是否使用毒药杀人", 
        default=False
    )
    target_name: Optional[str] = Field(
        description="目标玩家姓名（救人或毒杀的对象）",
        default=None
    )
    action_reason: Optional[str] = Field(
        description="行动理由",
        default=None
    )


def get_seer_model_cn(agents: Sequence[AgentBase]) -> type[BaseModel]:
    """获取中文版预言家模型"""
    agent_names = _agent_names(agents)
    
    class SeerModelCN(BaseModel):
        """中文版预言家查验格式"""
        
        target: Optional[str] = Field(
            description="要查验的玩家姓名",
            json_schema_extra={"enum": list(agent_names)},
            default=None,
        )
        check_reason: Optional[str] = Field(
            description="查验此人的原因",
            default=None,
        )
        priority_level: Optional[int] = Field(
            description="查验优先级(1-10)",
            ge=1, le=10,
            default=None,
        )

        @field_validator("target")
        @classmethod
        def validate_target(cls, value: Optional[str]) -> Optional[str]:
            if value is not None and value not in agent_names:
                raise ValueError(f"查验目标必须是：{', '.join(agent_names)}")
            return value
    
    return SeerModelCN


def get_hunter_model_cn(agents: Sequence[AgentBase]) -> type[BaseModel]:
    """获取中文版猎人模型"""
    agent_names = _agent_names(agents)
    
    class HunterModelCN(BaseModel):
        """中文版猎人开枪格式"""
        
        shoot: bool = Field(
            description="是否使用开枪技能",
            default=False,
        )
        target: Optional[str] = Field(
            description="开枪目标玩家姓名",
            default=None,
            json_schema_extra={"enum": [*agent_names, None]},
        )
        shoot_reason: Optional[str] = Field(
            description="开枪理由",
            default=None
        )

        @field_validator("target")
        @classmethod
        def validate_target(cls, value: Optional[str]) -> Optional[str]:
            if value is not None and value not in agent_names:
                raise ValueError(f"开枪目标必须是：{', '.join(agent_names)}")
            return value
    
    return HunterModelCN


class WerewolfKillModelCN(BaseModel):
    """中文版狼人击杀模型"""
    
    target: Optional[str] = Field(
        description="要击杀的玩家姓名",
        default=None,
    )
    kill_strategy: Optional[str] = Field(
        description="击杀策略说明",
        default=None,
    )
    team_coordination: Optional[str] = Field(
        description="与狼队友的配合计划",
        default=None
    )


class GameAnalysisModelCN(BaseModel):
    """中文版游戏分析模型"""
    
    suspected_werewolves: List[str] = Field(
        description="怀疑的狼人名单",
        default_factory=list
    )
    trusted_players: List[str] = Field(
        description="信任的玩家名单", 
        default_factory=list
    )
    key_clues: List[str] = Field(
        description="关键线索列表",
        default_factory=list
    )
    next_strategy: str = Field(
        description="下一步策略",
        default="",
    )
