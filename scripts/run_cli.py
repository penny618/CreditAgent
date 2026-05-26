"""
命令行快速验证整条链路(无需 Streamlit):
    python -m scripts.run_cli "客户申请10万元..."
不带参数则使用内置样例。
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import run_pipeline

DEFAULT = (
    "客户申请 10 万元消费贷。月收入 2 万元,现有月供 2000 元,本次月供约 3000 元。"
    "征信无逾期,近 1 月硬查询 1 次,信用卡利用率 25%,工作 6 年,公积金连续缴纳。"
)


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    print("=" * 60)
    print("输入申请材料:\n", text)
    print("=" * 60)

    result = run_pipeline(text)

    print("\n【① 结构化要素】")
    print(json.dumps(result.get("facts", {}), ensure_ascii=False, indent=2))

    print("\n【② 检索片段】")
    for i, c in enumerate(result.get("contexts", []), 1):
        print(f"  [{i}] ({c['source']}, score={c['score']:.3f}) {c['text'][:50]}...")

    print("\n【③ 风险分析】")
    print(result.get("analysis", ""))

    print("\n【④ 决策建议】")
    print(f"  决策: {result.get('decision')}")
    print(f"  评分: {result.get('risk_score')}")
    print(f"  理由: {result.get('decision_reason')}")


if __name__ == "__main__":
    main()
