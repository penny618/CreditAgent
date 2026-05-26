"""
自动生成信贷风控测试样本。

按「优质 / 中等 / 高风险」三类画像随机组合要素,生成自然语言申请材料,
并用一套透明的规则计算「参考决策」(标准答案),供批量评测对比。

用法:
    python -m scripts.gen_test_samples                 # 默认生成 30 条
    python -m scripts.gen_test_samples -n 60 --seed 7  # 自定义数量与随机种子
    python -m scripts.gen_test_samples -o data/test/my_samples.json

输出:
    data/test/test_samples.json   每条含:
        id, application_text(自由文本), params(结构化要素),
        expected_decision(参考决策), expected_score(参考评分), risk_tier(画像)
"""
import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------- 三类风险画像的参数取值区间 ----------
# 字段含义见下方 build_params
PROFILES = {
    "low": {       # 优质客户:大概率通过
        "income": (15000, 35000),
        "existing_debt_ratio": (0.0, 0.12),   # 现有月供 / 收入
        "amount": (60000, 150000),
        "term": [36, 48, 60],
        "overdue": ["无"],
        "inquiries_1m": (0, 2),
        "cc_util": (0.05, 0.30),
        "job_years": (3, 12),
        "fund": [True],
    },
    "medium": {    # 中等客户:多数转人工复核
        "income": (10000, 18000),
        "existing_debt_ratio": (0.12, 0.28),
        "amount": (100000, 200000),
        "term": [36, 48],
        "overdue": ["无", "近6个月有1次逾期1-30天"],
        "inquiries_1m": (2, 5),
        "cc_util": (0.45, 0.72),
        "job_years": (1, 5),
        "fund": [True, False],
    },
    "high": {      # 高风险客户:多数拒绝
        "income": (5000, 11000),
        "existing_debt_ratio": (0.3, 0.55),
        "amount": (180000, 400000),
        "term": [12, 24],
        "overdue": [
            "近6个月有3次逾期(连三)",
            "近1年累计逾期6次",
            "近6个月有2次逾期31-60天",
        ],
        "inquiries_1m": (6, 12),
        "cc_util": (0.8, 0.98),
        "job_years": (0, 2),
        "fund": [False],
    },
}

# 自由文本模板(多套,增加表达多样性)
TEMPLATES = [
    ("客户申请 {amount_w} 万元{loan_type}。月收入 {income} 元,现有月供合计 {ed} 元,"
     "本次贷款期限 {term} 个月、预计月供约 {nmp} 元。征信方面:{overdue};"
     "近 1 个月硬查询 {inq} 次;名下信用卡总授信 {cc_total_w} 万元,已用约 {cc_util_pct}%。"
     "当前单位工作 {job} 年,公积金{fund}。"),
    ("申请信息:{loan_type},金额 {amount} 元,期限 {term} 个月,月供约 {nmp} 元。"
     "申请人月收入 {income} 元,目前每月需偿还其他贷款 {ed} 元。"
     "信用记录:{overdue},近 30 天征信查询 {inq} 次,信用卡使用率 {cc_util_pct}%。"
     "工作年限 {job} 年,{fund_text}。"),
    ("一笔{loan_type}申请,借款 {amount} 元(分 {term} 期,月还 {nmp} 元)。"
     "借款人税前月入 {income} 元,既有月度负债 {ed} 元;{overdue};"
     "一个月内被查询征信 {inq} 次;信用卡额度利用率 {cc_util_pct}%;"
     "在职 {job} 年,公积金{fund}。"),
]

LOAN_TYPES = ["消费贷", "经营贷", "装修贷", "信用贷"]


def emi(principal: float, annual_rate: float, months: int) -> float:
    """等额本息月供。"""
    r = annual_rate / 12
    if r == 0:
        return principal / months
    factor = (1 + r) ** months
    return principal * r * factor / (factor - 1)


def build_params(tier: str, rng: random.Random) -> dict:
    p = PROFILES[tier]
    income = rng.randint(*p["income"]) // 100 * 100
    existing_debt = int(income * rng.uniform(*p["existing_debt_ratio"])) // 50 * 50
    amount = rng.randint(*p["amount"]) // 1000 * 1000
    term = rng.choice(p["term"])
    annual_rate = rng.choice([0.06, 0.072, 0.084, 0.096])
    new_mp = round(emi(amount, annual_rate, term) / 10) * 10
    cc_util = round(rng.uniform(*p["cc_util"]), 2)
    cc_total = rng.choice([30000, 50000, 80000, 100000, 120000])

    # 控制月供在可信范围:必要时延长期限或降额,避免 DTI 失真(高风险也设上限)
    disposable = max(income - existing_debt, income * 0.1)
    max_mp = disposable * 1.1
    if new_mp > max_mp:
        for t in (36, 48, 60):
            if t > term:
                cand = round(emi(amount, annual_rate, t) / 10) * 10
                if cand <= max_mp:
                    term, new_mp = t, cand
                    break
        if new_mp > max_mp:  # 仍超则按比例降额
            scale = max_mp / new_mp
            amount = max(int(amount * scale) // 1000 * 1000, 10000)
            new_mp = round(emi(amount, annual_rate, term) / 10) * 10

    return {
        "monthly_income": income,
        "existing_monthly_debt": existing_debt,
        "requested_amount": amount,
        "loan_term_months": term,
        "annual_rate": annual_rate,
        "new_monthly_payment": int(new_mp),
        "overdue_history": rng.choice(p["overdue"]),
        "hard_inquiries_1m": rng.randint(*p["inquiries_1m"]),
        "credit_card_total_limit": cc_total,
        "credit_card_utilization": cc_util,
        "job_years": rng.randint(*p["job_years"]),
        "has_housing_fund": rng.choice(p["fund"]),
        "loan_type": rng.choice(LOAN_TYPES),
    }


def reference_decision(params: dict) -> tuple:
    """
    透明规则计算参考决策与参考评分(0-100,越高越安全),作为评测标准答案。
    规则与知识库口径一致:连三累六 / DTI / 多头借贷 / 利用率 等。
    """
    score = 100
    notes = []

    dti = (params["existing_monthly_debt"] + params["new_monthly_payment"]) / max(params["monthly_income"], 1)
    if dti >= 0.7:
        score -= 45; notes.append(f"DTI={dti:.0%}严重超标")
    elif dti >= 0.5:
        score -= 25; notes.append(f"DTI={dti:.0%}偏高")
    elif dti >= 0.35:
        score -= 10; notes.append(f"DTI={dti:.0%}中等")

    ov = params["overdue_history"]
    if ("连三" in ov) or ("累计逾期6次" in ov) or ("31-60" in ov):
        score -= 40; notes.append("严重逾期(红线)")
    elif "1-30" in ov:
        score -= 10; notes.append("轻微逾期")

    inq = params["hard_inquiries_1m"]
    if inq >= 6:
        score -= 25; notes.append(f"硬查询{inq}次,多头借贷")
    elif inq >= 4:
        score -= 10; notes.append(f"硬查询{inq}次偏多")

    util = params["credit_card_utilization"]
    if util >= 0.8:
        score -= 15; notes.append(f"信用卡利用率{util:.0%}过高")
    elif util >= 0.6:
        score -= 8; notes.append(f"信用卡利用率{util:.0%}偏高")

    if params["job_years"] >= 3:
        score += 3
    if params["has_housing_fund"]:
        score += 3

    score = max(0, min(100, score))

    # 硬红线:严重逾期或 DTI>=0.7 直接拒绝
    hard_reject = ("连三" in ov) or ("累计逾期6次" in ov) or ("31-60" in ov) or dti >= 0.7
    if hard_reject or score < 50:
        decision = "拒绝"
    elif score >= 70:
        decision = "通过"
    else:
        decision = "转人工复核"
    return decision, score, notes


def render_text(params: dict, rng: random.Random) -> str:
    fund_word = "连续缴纳" if params["has_housing_fund"] else "未缴纳"
    fund_text = "有公积金" if params["has_housing_fund"] else "无公积金"
    fields = {
        "amount": params["requested_amount"],
        "amount_w": round(params["requested_amount"] / 10000, 1),
        "income": params["monthly_income"],
        "ed": params["existing_monthly_debt"],
        "term": params["loan_term_months"],
        "nmp": params["new_monthly_payment"],
        "overdue": params["overdue_history"],
        "inq": params["hard_inquiries_1m"],
        "cc_total_w": round(params["credit_card_total_limit"] / 10000, 1),
        "cc_util_pct": int(params["credit_card_utilization"] * 100),
        "job": params["job_years"],
        "fund": fund_word,
        "fund_text": fund_text,
        "loan_type": params["loan_type"],
    }
    return rng.choice(TEMPLATES).format(**fields)


def generate(n: int, seed: int) -> list:
    rng = random.Random(seed)
    # 三类画像大致均衡分配
    tiers = (["low"] * (n // 3) + ["medium"] * (n // 3) + ["high"] * (n - 2 * (n // 3)))
    rng.shuffle(tiers)

    samples = []
    for i, tier in enumerate(tiers, 1):
        params = build_params(tier, rng)
        decision, score, notes = reference_decision(params)
        samples.append({
            "id": f"case_{i:03d}",
            "risk_tier": tier,
            "application_text": render_text(params, rng),
            "params": params,
            "expected_decision": decision,
            "expected_score": score,
            "expected_notes": notes,
        })
    return samples


def main():
    ap = argparse.ArgumentParser(description="生成信贷风控测试样本")
    ap.add_argument("-n", "--num", type=int, default=30, help="样本数量(默认 30)")
    ap.add_argument("--seed", type=int, default=42, help="随机种子(默认 42,保证可复现)")
    ap.add_argument("-o", "--output", default="data/test/test_samples.json", help="输出路径")
    args = ap.parse_args()

    samples = generate(args.num, args.seed)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # 统计分布
    from collections import Counter
    tier_dist = Counter(s["risk_tier"] for s in samples)
    dec_dist = Counter(s["expected_decision"] for s in samples)
    print(f"已生成 {len(samples)} 条测试样本 -> {args.output}")
    print(f"  画像分布: {dict(tier_dist)}")
    print(f"  参考决策分布: {dict(dec_dist)}")
    print("\n示例(第 1 条):")
    print("  文本:", samples[0]["application_text"])
    print("  参考决策:", samples[0]["expected_decision"], "| 评分:", samples[0]["expected_score"])


if __name__ == "__main__":
    main()
