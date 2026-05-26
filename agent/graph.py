"""
用 LangGraph 串起多步 Agent:
    extract -> retrieve -> reason -> decide -> END
对外暴露 build_graph() 与便捷函数 run_pipeline(text)。
"""
from langgraph.graph import StateGraph, END

from .state import CreditState
from .nodes import extract_node, retrieve_node, reason_node, decide_node

_compiled = None


def build_graph():
    g = StateGraph(CreditState)
    g.add_node("extract", extract_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("reason", reason_node)
    g.add_node("decide", decide_node)

    g.set_entry_point("extract")
    g.add_edge("extract", "retrieve")
    g.add_edge("retrieve", "reason")
    g.add_edge("reason", "decide")
    g.add_edge("decide", END)

    return g.compile()


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def run_pipeline(application_text: str) -> dict:
    """一次性跑完整条链路,返回最终 state。"""
    graph = get_graph()
    result = graph.invoke({"application_text": application_text})
    return result


def stream_pipeline(application_text: str):
    """流式逐节点返回,供 Demo 展示中间过程。yield (node_name, partial_state)。"""
    graph = get_graph()
    for chunk in graph.stream({"application_text": application_text}):
        for node_name, partial in chunk.items():
            yield node_name, partial
