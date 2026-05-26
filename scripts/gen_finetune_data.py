"""
自动扩充信贷微调数据至 1k+(alpaca 格式)。

复用 gen_test_samples 的案例生成逻辑(随机要素 → 等额本息月供 → 透明风控规则),
不仅生成「问题」,还按规则自动构造「正确且多样」的答案,覆盖四类任务:

  1. 征信报告解读   —— 给定案例,指出风险点(对应「征信报告解读」)
  2. 审批建议       —— 给出 通过/复核/拒绝 + 量化理由(对应「信贷问答」)
  3. 要素抽取(JSON)—— 自由文本 → 结构化要素(对应 Agent 的抽取节点)
  4. 风控知识问答   —— 概念题(DTI / 多头借贷 / 逾期分级 等)

每条样本的数字均随机、互不相同,答案由规则推导,内部一致且可解释。

用法:
    python -m scripts.gen_finetune_data                  # 默认生成 1200 条
    python -m scripts.gen_finetune_data -n 2000 --seed 7
    python -m scripts.gen_finetune_data --use-llm        # 额外用 LLM 润色一部分(可选)

输出:
    覆盖写入 data/finetune/credit_qa.json(LLaMA-Factory 已注册的 credit_qa 数据集)
"""
import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.gen_test_samples import build_params, reference_decision, render_text, PROFILES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- 保留的高质量人工种子(始终置于数据集开头,保证运行幂等)----------
SEED_SAMPLES = [
    {
        "instruction": "解读以下征信报告片段,指出潜在风险点。",
        "input": "客户当前有 3 笔信用卡,总授信额度 8 万元,已使用额度 7.6 万元;近 6 个月有 2 次「逾期 1-30 天」记录;名下有 1 笔住房贷款,月供 4500 元,从未逾期。",
        "output": "风险点如下:\n1. 信用卡使用率约 95%(7.6/8 万),额度利用率过高,反映短期流动性紧张,是重要的高风险信号。\n2. 近 6 个月出现 2 次轻微逾期(1-30 天),说明还款行为存在波动,需关注是否为习惯性拖延。\n3. 正面因素:住房贷款月供 4500 元从未逾期,说明客户对大额刚性负债有较强的还款意愿与能力。\n综合判断:短期信用风险偏高,建议核实其月收入与负债比(DTI),若 DTI 超过 50% 应审慎授信。"
    },
    {
        "instruction": "客户申请 20 万元消费贷,请基于以下信息给出审批建议。",
        "input": "月收入 1.5 万元,现有月供合计 3000 元,本次贷款月供约 6000 元,征信无逾期,工作年限 5 年,公积金连续缴纳。",
        "output": "审批分析:\n1. 授信后 DTI = (3000 + 6000) / 15000 = 60%,超过 50% 的警戒线,还款负担偏重。\n2. 正面因素:征信无逾期、工作稳定 5 年、公积金连续缴纳,还款意愿与稳定性较好。\n3. 结论:虽然信用记录良好,但 DTI 过高是硬约束。建议:① 降低授信额度至 12-14 万元,使 DTI 回落到 50% 以内;② 或要求延长还款期限以摊薄月供;③ 不建议按原额度直接通过。\n最终建议:有条件通过(降额),转人工复核确认收入流水。"
    },
    {
        "instruction": "判断该客户是否存在多头借贷风险。",
        "input": "近 1 个月征信硬查询 8 次(其中网络小贷审批 6 次),名下已有 5 笔小额网贷在贷,总余额 4 万元。",
        "output": "明确存在多头借贷高风险,判断依据:\n1. 近 1 个月硬查询 8 次,远超 6 次警戒阈值,且 6 次为网络小贷审批,说明客户在短期内密集申请借款。\n2. 名下已有 5 笔网贷在贷,「以贷养贷」或资金极度紧张的概率高。\n3. 多头借贷客户的违约率通常显著高于单一渠道借款人。\n建议:拒绝本次申请,或转人工严格审查资金用途与真实负债。"
    },
    {
        "instruction": "什么是 DTI(债务收入比),在信贷审批中如何使用?",
        "input": "",
        "output": "DTI(Debt-to-Income Ratio,债务收入比)= 借款人每月债务还款总额 / 每月税前收入 × 100%,衡量还款负担。常见用法:DTI<35% 负债健康;35%≤DTI<50% 中等负担,需结合其他因素;DTI≥50% 还款压力大、违约概率上升,通常需降额或拒绝。计算时应把本次新增月供一并计入,评估授信后的真实负担。"
    },
    {
        "instruction": "解释征信报告中「查询记录」过多代表什么。",
        "input": "",
        "output": "查询分硬查询(贷款审批、信用卡审批、担保资格审查)和软查询(本人查询、贷后管理)。短期内(1-2 个月)硬查询次数过多(如 ≥6 次)通常意味着多头借贷、四处申请额度,资金链可能紧张,是显著高风险信号;软查询一般不影响信用评估。风控中常统计近 1/3/6 个月硬查询次数作为多头借贷的代理指标。"
    },
]

# ---------- 风控知识问答种子(概念题,问句会做有限改写)----------
KNOWLEDGE_QA = [
    ("信用卡额度利用率多少算高?为什么?",
     "信用卡额度利用率 = 已用额度 / 总授信额度。长期高于 80% 是高风险信号,反映借款人短期流动性紧张;低于 30% 通常视为健康水平。多张卡同时高利用率,叠加表明资金链紧张。"),
    ("征信里「连三累六」是什么意思?",
     "「连三累六」指连续逾期 3 次或累计逾期 6 次,是多数金融机构的拒贷红线,代表借款人还款意愿或能力存在严重问题。"),
    ("逾期记录按严重程度怎么分级?",
     "通常按逾期天数分级:M1(逾期 1-30 天)为轻微逾期,可能为疏忽,但频繁出现需警惕;M2(31-60 天)及以上为严重逾期,违约风险显著;触及「连三累六」即为拒贷红线。"),
    ("什么是多头借贷,风险点在哪?",
     "多头借贷指借款人在短期内向多家机构密集申请借款,常表现为近 1 个月硬查询 ≥6 次、名下多笔小额网贷在贷。这类客户「以贷养贷」或资金极度紧张的概率高,违约率显著高于单一渠道借款人。"),
    ("信贷综合授信决策遵循什么原则?",
     "遵循「还款能力 + 还款意愿」双维度:还款能力看 DTI、收入水平、资产状况;还款意愿看历史逾期、查询记录、负债结构。任一维度出现硬红线(连三累六、DTI 远超 50%、严重多头借贷)即应否决或转严格人工复核。"),
    ("工作年限和公积金对信贷审批有什么意义?",
     "当前单位工作年限越长、社保公积金连续缴纳,代表收入越稳定、真实性越强,一般在职 2 年以上为佳;频繁换工作、收入来源单一且不稳定的借款人风险较高。"),
]

# ---------- 各类任务的指令措辞池(增加多样性)----------
INSTR_INTERPRET = [
    "解读以下征信/申请信息,指出潜在风险点。",
    "请作为风控分析师,分析下面这笔申请的主要风险。",
    "审阅以下客户资料,列出需要关注的风险信号。",
    "下面是一笔贷款申请,请指出其中的征信与负债风险点。",
]
INSTR_APPROVAL = [
    "基于以下信息给出审批建议(通过/转人工复核/拒绝)并说明理由。",
    "请判断这笔贷款是否可以批准,并给出量化依据。",
    "作为信贷审批人,请对下面的申请给出处理意见。",
    "结合还款能力与还款意愿,给出本笔申请的审批结论。",
]
INSTR_EXTRACT = [
    "从下面的贷款申请材料中抽取结构化要素,只输出 JSON。",
    "请把以下申请文本解析为结构化字段,以 JSON 返回,不要多余文字。",
    "提取下列材料中的关键信贷要素,输出 JSON。",
]
INSTR_KNOWLEDGE_PARAPHRASE = ["{q}", "请解释:{q}", "{q}请简要说明。"]


def dti_of(p):
    return (p["existing_monthly_debt"] + p["new_monthly_payment"]) / max(p["monthly_income"], 1)


def dti_judgement(dti):
    if dti >= 0.7:
        return f"DTI={dti:.0%},远超 50% 警戒线,还款负担极重"
    if dti >= 0.5:
        return f"DTI={dti:.0%},超过 50% 警戒线,还款负担偏重"
    if dti >= 0.35:
        return f"DTI={dti:.0%},处于中等负担区间"
    return f"DTI={dti:.0%},负债健康"


def risk_sentences(p, rng):
    """把命中的风险点写成自然语句(顺序随机,措辞多样)。"""
    s = []
    dti = dti_of(p)
    if dti >= 0.5:
        s.append(dti_judgement(dti) + "。")
    ov = p["overdue_history"]
    if ("连三" in ov) or ("累计逾期6次" in ov):
        s.append("征信触及「连三累六」红线,还款意愿存在严重问题。")
    elif "31-60" in ov:
        s.append("存在 31-60 天的严重逾期(M2),违约风险显著。")
    elif "1-30" in ov:
        s.append("近期有 1-30 天轻微逾期,还款行为存在波动,需关注是否习惯性拖延。")
    inq = p["hard_inquiries_1m"]
    if inq >= 6:
        s.append(f"近 1 个月硬查询 {inq} 次,超过 6 次警戒阈值,存在多头借贷迹象。")
    elif inq >= 4:
        s.append(f"近 1 个月硬查询 {inq} 次偏多,需留意多头借贷倾向。")
    util = p["credit_card_utilization"]
    if util >= 0.8:
        s.append(f"信用卡额度利用率约 {util:.0%},长期高位反映短期流动性紧张。")
    elif util >= 0.6:
        s.append(f"信用卡额度利用率约 {util:.0%},偏高,需关注资金占用。")
    rng.shuffle(s)
    return s


def positive_sentences(p):
    pos = []
    if "无" in p["overdue_history"]:
        pos.append("征信无逾期记录,还款意愿良好")
    if p["job_years"] >= 3:
        pos.append(f"当前单位在职 {p['job_years']} 年,收入稳定性较好")
    if p["has_housing_fund"]:
        pos.append("公积金连续缴纳,收入真实性有保障")
    if p["credit_card_utilization"] < 0.3:
        pos.append(f"信用卡利用率仅 {p['credit_card_utilization']:.0%},负债健康")
    return pos


def suggestion_for(decision, p):
    dti = dti_of(p)
    if decision == "通过":
        return "可按申请额度通过,建议放款后纳入常规贷后监控。"
    if decision == "转人工复核":
        tips = []
        if dti >= 0.5:
            target = int((0.5 * p["monthly_income"] - p["existing_monthly_debt"]))
            tips.append(f"建议降额或延长期限,使新增月供降至约 {max(target,0)} 元以内以将 DTI 压到 50% 以下")
        tips.append("转人工复核,核实收入流水与贷款用途")
        return ";".join(tips) + "。"
    # 拒绝
    if ("连三" in p["overdue_history"]) or ("累计逾期6次" in p["overdue_history"]) or ("31-60" in p["overdue_history"]):
        return "因征信存在严重逾期红线,建议拒绝本次申请。"
    if dti >= 0.7:
        return "因还款负担过重(DTI≥70%),建议拒绝;若客户坚持,可大幅降额后重新评估。"
    return "综合风险偏高,建议拒绝或转严格人工审查资金用途与真实负债。"


def compose_interpretation(p, rng):
    risks = risk_sentences(p, rng)
    pos = positive_sentences(p)
    lines = ["风险点如下:"]
    if risks:
        for i, r in enumerate(risks, 1):
            lines.append(f"{i}. {r}")
    else:
        lines.append("1. 未发现明显高风险信号,各项指标处于健康区间。")
    if pos:
        lines.append("正面因素:" + ";".join(pos) + "。")
    decision, score, _ = reference_decision(p)
    lines.append(f"综合判断:风险评分约 {score} 分,初步结论为「{decision}」。")
    return "\n".join(lines)


def compose_approval(p, rng):
    decision, score, _ = reference_decision(p)
    dti = dti_of(p)
    lines = ["审批分析:"]
    lines.append(f"1. 授信后 {dti_judgement(dti)}(现有月供 {p['existing_monthly_debt']} + "
                 f"新增月供 {p['new_monthly_payment']} 占月收入 {p['monthly_income']})。")
    risks = risk_sentences(p, rng)
    idx = 2
    for r in risks:
        if r.startswith("DTI"):
            continue
        lines.append(f"{idx}. {r}")
        idx += 1
    pos = positive_sentences(p)
    if pos:
        lines.append(f"{idx}. 正面因素:" + ";".join(pos) + "。")
    lines.append(f"结论:综合风险评分约 {score} 分,建议「{decision}」。{suggestion_for(decision, p)}")
    return "\n".join(lines)


def compose_extraction(p):
    text = render_text(p, random.Random(p["requested_amount"] + p["monthly_income"]))
    facts = {
        "monthly_income": p["monthly_income"],
        "existing_monthly_debt": p["existing_monthly_debt"],
        "requested_amount": p["requested_amount"],
        "new_monthly_payment": p["new_monthly_payment"],
        "overdue_history": p["overdue_history"],
        "hard_inquiries_1m": p["hard_inquiries_1m"],
        "credit_card_utilization": p["credit_card_utilization"],
        "job_years": p["job_years"],
        "other": ("公积金连续缴纳" if p["has_housing_fund"] else "无公积金"),
    }
    return text, json.dumps(facts, ensure_ascii=False)


def maybe_llm_polish(samples, frac, rng):
    """可选:用配置好的 LLM 对一部分样本的答案做口语化润色,增加自然度。"""
    try:
        from agent.llm import LLMClient
        llm = LLMClient()
    except Exception as e:
        print(f"[--use-llm] 无法初始化 LLM({e}),跳过润色。")
        return samples
    n = int(len(samples) * frac)
    picks = rng.sample(range(len(samples)), n)
    print(f"[--use-llm] 对 {n} 条样本润色中...")
    for k, i in enumerate(picks, 1):
        s = samples[i]
        if not s["output"].startswith("{"):  # 不润色 JSON 抽取样本
            try:
                polished = llm.ask(
                    "你是信贷风控写作助手。在不改变事实、数字、结论的前提下,"
                    "把下面的分析改写得更自然连贯,保持专业、简洁。只输出改写后的正文。",
                    s["output"],
                )
                if polished and len(polished) > 20:
                    samples[i]["output"] = polished.strip()
            except Exception:
                pass
        if k % 20 == 0:
            print(f"  已润色 {k}/{n}")
    return samples


def generate(n, seed, use_llm=False):
    rng = random.Random(seed)
    samples = list(SEED_SAMPLES)

    # 1) 知识问答(概念题,有限改写)
    for q, a in KNOWLEDGE_QA:
        for tmpl in rng.sample(INSTR_KNOWLEDGE_PARAPHRASE, 2):
            samples.append({"instruction": tmpl.format(q=q), "input": "", "output": a})

    # 2) 案例类样本:解读 / 审批 / 抽取,按比例随机生成,数字互不相同
    remaining = max(0, n - len(samples))
    cats = (["interpret"] * int(remaining * 0.35)
            + ["approval"] * int(remaining * 0.35)
            + ["extract"] * (remaining - int(remaining * 0.35) - int(remaining * 0.35)))
    rng.shuffle(cats)
    tiers = ["low", "medium", "high"]

    for cat in cats:
        p = build_params(rng.choice(tiers), rng)
        if cat == "interpret":
            samples.append({
                "instruction": rng.choice(INSTR_INTERPRET),
                "input": render_text(p, rng),
                "output": compose_interpretation(p, rng),
            })
        elif cat == "approval":
            samples.append({
                "instruction": rng.choice(INSTR_APPROVAL),
                "input": render_text(p, rng),
                "output": compose_approval(p, rng),
            })
        else:  # extract
            text, facts_json = compose_extraction(p)
            samples.append({
                "instruction": rng.choice(INSTR_EXTRACT),
                "input": text,
                "output": facts_json,
            })

    rng.shuffle(samples)
    if use_llm:
        samples = maybe_llm_polish(samples, frac=0.15, rng=rng)
    return samples


def main():
    ap = argparse.ArgumentParser(description="自动扩充信贷微调数据至 1k+")
    ap.add_argument("-n", "--num", type=int, default=1200, help="目标样本数(默认 1200)")
    ap.add_argument("--seed", type=int, default=42, help="随机种子(默认 42,可复现)")
    ap.add_argument("-o", "--output", default="data/finetune/credit_qa.json")
    ap.add_argument("--use-llm", action="store_true", help="额外用配置的 LLM 润色 ~15% 样本(可选)")
    args = ap.parse_args()

    samples = generate(args.num, args.seed, use_llm=args.use_llm)

    out = os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # 统计
    from collections import Counter
    def cat_of(s):
        if s["output"].startswith("{"):
            return "要素抽取(JSON)"
        if s["instruction"] in INSTR_APPROVAL:
            return "审批建议"
        if s["instruction"] in INSTR_INTERPRET:
            return "征信解读"
        return "知识问答"
    dist = Counter(cat_of(s) for s in samples)
    print(f"已生成 {len(samples)} 条微调样本 -> {args.output}")
    print("  类别分布:")
    for k, v in dist.items():
        print(f"    {k:<16}{v}")
    print("\n示例(随机 1 条):")
    eg = samples[len(samples) // 2]
    print("  instruction:", eg["instruction"])
    print("  input:", (eg["input"][:60] + "...") if eg["input"] else "(空)")
    print("  output:", eg["output"][:120].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()
