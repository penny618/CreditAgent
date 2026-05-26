"""
四个 Agent 节点:
  extract  : 从自由文本抽取结构化信贷要素(LLM)
  retrieve : 基于要素检索风控/征信知识(RAG)
  reason   : 结合知识做可溯源风险分析 + 打分(LLM)
  decide   : 按阈值映射为最终决策(规则)
"""
import json
import re

from rag import Retriever
from .llm import LLMClient
from .state import CreditState

# 单例,避免每次调用重复加载模型
_llm = None
_retriever = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _extract_json(text: str):
    """从模型输出里稳健地抠出 JSON。"""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


# ---------------- 节点 1:信息抽取 ----------------
def extract_node(state: CreditState) -> CreditState:
    llm = _get_llm()
    system = "你是信贷风控信息抽取助手,只输出 JSON,不要任何多余文字。"
    user = (
        "从下面的贷款申请材料中抽取结构化要素。"
        "字段:monthly_income(月收入,元), existing_monthly_debt(现有月供合计,元), "
        "requested_amount(申请金额,元), new_monthly_payment(本次新增月供,元), "
        "overdue_history(逾期情况,字符串), hard_inquiries_1m(近1月硬查询次数,整数), "
        "credit_card_utilization(信用卡额度利用率,0-1小数), job_years(当前工作年限), "
        "other(其他要点,字符串)。无法确定的字段填 null。\n\n"
        f"申请材料:\n{state['application_text']}\n\n只输出 JSON。"
    )
    raw = llm.ask(system, user)
    facts = _extract_json(raw)
    return {"facts": facts}


# ---------------- 节点 2:检索 ----------------
def retrieve_node(state: CreditState) -> CreditState:
    retriever = _get_retriever()
    facts = state.get("facts", {})

    # 用要素拼出检索意图,引导命中相关知识条目
    query_parts = ["信贷风控 审批 风险点"]
    if facts.get("credit_card_utilization") is not None:
        query_parts.append("信用卡额度利用率")
    if facts.get("overdue_history"):
        query_parts.append("逾期记录 风险等级")
    if facts.get("hard_inquiries_1m"):
        query_parts.append("征信查询 多头借贷")
    if facts.get("monthly_income") or facts.get("new_monthly_payment"):
        query_parts.append("债务收入比 DTI")
    query = " ".join(query_parts)

    results = retriever.search(query)
    return {"retrieval_query": query, "contexts": results}


# ---------------- 节点 3:推理分析 + 打分 ----------------
def reason_node(state: CreditState) -> CreditState:
    llm = _get_llm()
    retriever = _get_retriever()

    context = retriever.format_context(state.get("contexts", []))
    facts = state.get("facts", {})

    system = (
        "你是资深信贷风控分析师。请严格依据【知识库】给出风险分析,"
        "引用知识时用 [编号] 标注来源,确保结论可溯源。"
        "最后必须给出 risk_score(0-100 的整数,分数越高越安全)。"
    )
    user = (
        f"【知识库】\n{context}\n\n"
        f"【结构化要素】\n{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n"
        f"【原始材料】\n{state['application_text']}\n\n"
        "请输出:\n"
        "1) 风险分析(分条,引用 [编号]);\n"
        "2) 最后一行严格按格式输出:RISK_SCORE=<0-100整数>"
    )
    analysis = llm.ask(system, user)

    # 解析风险分
    m = re.search(r"RISK_SCORE\s*=\s*(\d{1,3})", analysis)
    score = int(m.group(1)) if m else 50
    score = max(0, min(100, score))
    return {"analysis": analysis, "risk_score": score}


# ---------------- 节点 4:决策 ----------------
def decide_node(state: CreditState) -> CreditState:
    from config import load_config
    acfg = load_config()["agent"]
    score = state.get("risk_score", 50)

    if score >= acfg["approve_threshold"]:
        decision = "通过"
    elif score >= acfg["review_threshold"]:
        decision = "转人工复核"
    else:
        decision = "拒绝"

    reason = (
        f"综合风险评分 {score}(阈值:通过≥{acfg['approve_threshold']}, "
        f"复核≥{acfg['review_threshold']})。详见风险分析。"
    )
    return {"decision": decision, "decision_reason": reason}
