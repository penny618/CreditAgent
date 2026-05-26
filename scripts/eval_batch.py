"""
批量评测:在测试样本上跑完整 Agent 链路,与参考决策对比,输出准确率与明细。

用法:
    # 1) 先生成样本
    python -m scripts.gen_test_samples -n 30
    # 2) 跑真实链路评测(需 LLM 服务 + 已建索引)
    python -m scripts.eval_batch
    # 3) 无服务时快速自检:用透明规则代替 LLM,验证数据与评测流程
    python -m scripts.eval_batch --mock

输出:
    控制台明细表 + 决策一致率;结果落盘 data/test/eval_results.json
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_samples(path: str):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        raise SystemExit(f"未找到测试样本 {path},请先运行: python -m scripts.gen_test_samples")
    with open(full, "r", encoding="utf-8") as f:
        return json.load(f)


def run_real(text: str) -> dict:
    """调用真实 Agent 链路。"""
    from agent import run_pipeline
    res = run_pipeline(text)
    return {"decision": res.get("decision"), "score": res.get("risk_score")}


def run_mock(sample: dict) -> dict:
    """无 LLM 时,用生成器的透明规则直接复算,验证评测流程本身跑得通。"""
    from scripts.gen_test_samples import reference_decision
    d, s, _ = reference_decision(sample["params"])
    return {"decision": d, "score": s}


def main():
    ap = argparse.ArgumentParser(description="批量评测 Agent 决策")
    ap.add_argument("-i", "--input", default="data/test/test_samples.json")
    ap.add_argument("-o", "--output", default="data/test/eval_results.json")
    ap.add_argument("--mock", action="store_true", help="用规则代替 LLM,快速自检流程")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条(0=全部)")
    args = ap.parse_args()

    samples = load_samples(args.input)
    if args.limit:
        samples = samples[: args.limit]

    mode = "MOCK(规则)" if args.mock else "REAL(Agent 链路)"
    print(f"评测模式: {mode} | 样本数: {len(samples)}\n")

    print(f"{'ID':<10}{'画像':<8}{'参考决策':<10}{'Agent决策':<10}{'参考分':<7}{'Agent分':<8}{'一致'}")
    print("-" * 64)

    results, correct = [], 0
    t0 = time.time()
    for s in samples:
        try:
            out = run_mock(s) if args.mock else run_real(s["application_text"])
        except Exception as e:
            out = {"decision": f"ERROR", "score": -1, "error": str(e)}

        hit = (out["decision"] == s["expected_decision"])
        correct += int(hit)
        mark = "✓" if hit else "✗"
        print(f"{s['id']:<10}{s['risk_tier']:<8}{s['expected_decision']:<10}"
              f"{str(out['decision']):<10}{s['expected_score']:<7}{str(out['score']):<8}{mark}")

        results.append({
            "id": s["id"], "risk_tier": s["risk_tier"],
            "expected_decision": s["expected_decision"], "agent_decision": out["decision"],
            "expected_score": s["expected_score"], "agent_score": out["score"],
            "match": hit,
        })

    elapsed = time.time() - t0
    acc = correct / len(samples) if samples else 0
    print("-" * 64)
    print(f"决策一致率: {correct}/{len(samples)} = {acc:.1%} | 耗时 {elapsed:.1f}s")

    # 按画像分桶统计
    from collections import defaultdict
    bucket = defaultdict(lambda: [0, 0])
    for r in results:
        bucket[r["risk_tier"]][1] += 1
        bucket[r["risk_tier"]][0] += int(r["match"])
    print("\n分画像一致率:")
    for tier, (c, t) in bucket.items():
        print(f"  {tier:<8}{c}/{t} = {c/t:.1%}")

    out_path = os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "accuracy": acc, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n明细已保存: {args.output}")


if __name__ == "__main__":
    main()
