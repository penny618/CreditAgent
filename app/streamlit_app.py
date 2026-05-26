"""
CreditAgent 交互 Demo —— 单笔贷端到端推理验证。
启动: streamlit run app/streamlit_app.py
"""
import os
import sys

# 让 streamlit 子进程能 import 到项目包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from agent import stream_pipeline

st.set_page_config(page_title="CreditAgent 信贷风控智能体", page_icon="🏦", layout="wide")

st.title("🏦 CreditAgent · 信贷风控智能体")
st.caption("微调 → 检索 → 编排 → 决策 全链路 Demo（单笔贷端到端推理）")

DEFAULT_CASE = (
    "客户申请 20 万元消费贷。月收入 1.5 万元,现有月供合计 3000 元,"
    "本次贷款月供约 6000 元。征信近 6 个月无逾期,但近 1 个月硬查询 7 次"
    "(其中网络小贷审批 5 次)。名下 4 张信用卡,总授信 8 万元,已用 7.2 万元。"
    "当前单位工作 5 年,公积金连续缴纳。"
)

with st.sidebar:
    st.header("说明")
    st.markdown(
        "- 输入一笔贷款申请的自由文本\n"
        "- Agent 按 **抽取→检索→推理→决策** 四步运行\n"
        "- 每步过程实时展示,决策结果可溯源"
    )
    st.divider()
    st.markdown("**链路节点**")
    st.markdown("1. 信息抽取 (LLM)\n2. 知识检索 (BGE-M3 + Rerank)\n3. 风险推理 (微调 Qwen3)\n4. 决策建议 (规则阈值)")

application_text = st.text_area("贷款申请材料", value=DEFAULT_CASE, height=160)

run = st.button("🚀 开始分析", type="primary")

if run:
    if not application_text.strip():
        st.warning("请输入申请材料。")
        st.stop()

    facts, contexts, analysis, decision = None, None, None, None

    progress = st.progress(0, text="启动 Agent...")
    step_titles = {"extract": "① 信息抽取", "retrieve": "② 知识检索",
                   "reason": "③ 风险推理", "decide": "④ 决策建议"}
    step_order = ["extract", "retrieve", "reason", "decide"]

    col_left, col_right = st.columns([1, 1])

    try:
        for i, (node, partial) in enumerate(stream_pipeline(application_text)):
            progress.progress((i + 1) / len(step_order),
                              text=f"已完成:{step_titles.get(node, node)}")

            if node == "extract":
                facts = partial.get("facts", {})
                with col_left:
                    st.subheader("① 结构化要素")
                    st.json(facts)

            elif node == "retrieve":
                contexts = partial.get("contexts", [])
                with col_left:
                    st.subheader("② 检索到的知识(可溯源)")
                    for j, c in enumerate(contexts, 1):
                        with st.expander(f"[{j}] {c['source']}  ·  相关度 {c['score']:.3f}"):
                            st.write(c["text"])

            elif node == "reason":
                analysis = partial.get("analysis", "")
                score = partial.get("risk_score", None)
                with col_right:
                    st.subheader("③ 风险分析")
                    if score is not None:
                        st.metric("风险评分(越高越安全)", score)
                    st.markdown(analysis)

            elif node == "decide":
                decision = partial.get("decision", "")
                reason = partial.get("decision_reason", "")
                with col_right:
                    st.subheader("④ 决策建议")
                    color = {"通过": "green", "转人工复核": "orange", "拒绝": "red"}.get(decision, "gray")
                    st.markdown(f"### :{color}[{decision}]")
                    st.caption(reason)

        progress.progress(1.0, text="完成 ✅")
        st.success("端到端推理完成。")
    except Exception as e:
        st.error(f"运行出错:{e}")
        st.info("请确认:① 已运行 `python -m rag.build_index` 建好索引;"
                "② LLM 服务已启动且 config.yaml 配置正确。")
