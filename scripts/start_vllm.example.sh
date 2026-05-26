#!/usr/bin/env bash
# 启动 vLLM 服务,对外暴露微调后的 credit-qwen3-4b。
# 用法: cp scripts/start_vllm.example.sh scripts/start_vllm.sh,改好 CUDA_HOME 后:
#       nohup bash scripts/start_vllm.sh > logs/vllm_server.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

# 在较新 GPU(如 Blackwell sm_120)上,FlashInfer 需要在启动时 JIT 编译 CUDA 算子,
# 因此要提供一套完整的 CUDA 工具链(含 nvcc)。改成你本机的 CUDA 路径:
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
# 若该工具链把库放在 lib/ 而非 lib64/(如 conda 安装),用 LIBRARY_PATH 兜底链接,
# LD_LIBRARY_PATH 供运行期加载(WSL 下驱动库通常在 /usr/lib/wsl/lib)。
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

exec .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model ./finetune/output/credit_qwen3_merged \
  --served-model-name credit-qwen3-4b \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code
