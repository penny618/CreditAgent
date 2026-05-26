# 微调模块:LLaMA-Factory 对 Qwen3-4B 做 LoRA

本模块用 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 对 Qwen3-4B 做 LoRA 微调,
让模型适配「信贷问答 + 征信报告解读」场景。

## 1. 安装 LLaMA-Factory

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..
```

## 2. 准备数据

仓库自带一个**微调数据生成器**,可一键把样本自动扩充到 1k+(alpaca 格式):

```bash
# 在项目根目录 CreditAgent/ 下执行
python -u -m scripts.gen_finetune_data -n 2000 --seed 42 --use-llm 2>&1 \
  | tee "./logs/gen_finetune_$(date +%Y%m%d_%H%M%S).log"
#   -> 覆盖写入 data/finetune/credit_qa.json,并已在 dataset_info.json 注册为 credit_qa
```

生成的样本覆盖四类任务,每条数字随机、答案由透明风控规则推导(内部一致、可解释):
- **征信报告解读**:给定案例,列出风险点;
- **审批建议**:给出 通过/转人工复核/拒绝 + 量化理由(含 DTI 计算);
- **要素抽取(JSON)**:自由文本 → 结构化字段,直接对应 Agent 的抽取节点;
- **风控知识问答**:DTI、多头借贷、逾期分级等概念题。

> 可选:加 `--use-llm` 会调用 `config.yaml` 里配置的 LLM 对约 15% 样本做口语化润色,
> 进一步提升自然度(需先起好 LLM 服务;不加则纯离线生成)。
> 真实项目中也可把这里替换为人工标注的真实信贷问答与征信解读样本。


## 3. 启动训练

```bash
# 在项目根目录 CreditAgent/ 下执行
nohup .venv/bin/llamafactory-cli train finetune/train_lora.yaml \
    > "logs/train_lora_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```
```bash
   cd /home/penny/yxr_llm/CreditAgent
   LOG="logs/train_lora_$(date +%Y%m%d_%H%M%S).log"
   echo "训练日志: $LOG"
   nohup .venv/bin/llamafactory-cli train finetune/train_lora.yaml > "$LOG" 2>&1 &
   echo "训练已启动，PID=$!"
   echo "$LOG" > /tmp/creditagent_last_train_log.txt
   sleep 2
   # 起 tensorboard(后台)，监听 6006
   nohup .venv/bin/tensorboard --logdir finetune/output/credit_lora --port 6006 --host 0.0.0.0 > logs/tensorboard.log 2>&1 &
   echo "TensorBoard 已启动，PID=$!  -> http://localhost:6006"
```
```bash
# 进入项目目录后
tensorboard --logdir runs/May25_17-24-51_DESKTOP-0OLBSET

# 或者指向 runs 上级目录, 可以同时看到所有历史实验
tensorboard --logdir runs
```

训练产物(LoRA 适配器)会保存到 `finetune/output/credit_lora/`。

> 显存参考:Qwen3-4B + LoRA + bf16 + gradient_checkpointing,单卡约需 16~20GB。
> 显存紧张时,在 `train_lora.yaml` 加入 `quantization_bit: 4` 走 QLoRA,可压到 ~8GB。

## 4.(可选)合并 LoRA 权重

如果想得到一个独立的全量权重模型:

```bash
llamafactory-cli export \
  --model_name_or_path /home/penny/yxr_llm/credit_risk_pboc/models/Qwen3-4B \
  --adapter_name_or_path ./finetune/output/credit_lora \
  --template qwen \
  --finetuning_type lora \
  --export_dir ./finetune/output/credit_qwen3_merged \
  --export_size 2 \
  --export_legacy_format false \
  2>&1 | tee logs/export_merge.log
```

## 5. 部署成 OpenAI 兼容服务(供 Agent 调用)

推荐用 vLLM 拉起推理服务,Agent 与 RAG 都通过 OpenAI 接口调用它:

```bash
# 方式一:加载 base + LoRA(无需合并)
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --enable-lora \
  --lora-modules credit-lora=./finetune/output/credit_lora \
  --served-model-name credit-qwen3-4b \
  --port 8000

# 方式二:加载已合并的权重
nohup python -m vllm.entrypoints.openai.api_server \
  --model ./finetune/output/credit_qwen3_merged \
  --served-model-name credit-qwen3-4b \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  > logs/vllm_server.log 2>&1 &

echo "服务已后台启动, PID=$!"
echo "实时看日志: tail -f logs/vllm_server.log"


# 方式三：执行start_vllm.sh
cd /home/penny/yxr_llm/CreditAgent
chmod +x scripts/start_vllm.sh
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 1
nohup bash scripts/start_vllm.sh > logs/vllm_server.log 2>&1 &
echo "launched pid=$!"
```

服务起来后,`config/config.yaml` 里的 `llm.base_url` 指向 `http://localhost:8000/v1` 即可。

> 没有 GPU 时,可临时把 `config.yaml` 的 `llm.model_name` 换成任意可用的
> OpenAI 兼容服务(如 Ollama 跑 qwen,或线上 API),先把整条链路跑通再回来替换微调模型。
