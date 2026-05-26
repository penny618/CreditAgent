"""Agent 状态定义:在各节点间流转的数据结构。"""
from typing import TypedDict, List, Dict, Any, Optional


class CreditState(TypedDict, total=False):
    # 输入
    application_text: str          # 原始申请材料/征信描述(自由文本)

    # 信息抽取节点产出
    facts: Dict[str, Any]          # 结构化要素(收入、负债、逾期、查询次数等)

    # 检索节点产出
    retrieval_query: str           # 由要素拼出的检索查询
    contexts: List[Dict[str, Any]] # 检索片段 [{'text','source','score'}]

    # 推理节点产出
    analysis: str                  # 基于知识的风险分析(带引用)
    risk_score: int                # 0-100 风险评分(越高越安全)

    # 决策节点产出
    decision: str                  # 通过 / 转人工复核 / 拒绝
    decision_reason: str           # 决策理由
