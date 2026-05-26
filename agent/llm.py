"""
LLM 封装:统一对外暴露 chat(messages) -> str。
支持两种后端:
  - openai_api : 走 vLLM/Ollama 等 OpenAI 兼容接口(推荐,微调模型用 vLLM 拉起)
  - transformers: 本地直接加载 base + LoRA(无服务时备用)
"""
from config import load_config, resolve_path


class LLMClient:
    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        self.lcfg = self.cfg["llm"]
        self.backend = self.lcfg.get("backend", "openai_api")
        self._client = None
        self._hf = None  # (model, tokenizer)

    # ---------- 后端一:OpenAI 兼容接口 ----------
    def _chat_openai(self, messages):
        import httpx
        from openai import OpenAI
        if self._client is None:
            # base_url 指向本地/局域网 vLLM,绝不应走系统代理。
            # 注意:no_proxy 里的 glob(如 192.168.*)httpx/curl 都不识别,
            # 故直接用 trust_env=False 让 httpx 忽略所有代理环境变量,避免 502。
            self._client = OpenAI(
                base_url=self.lcfg["base_url"],
                api_key=self.lcfg.get("api_key", "EMPTY"),
                http_client=httpx.Client(trust_env=False),
            )
        resp = self._client.chat.completions.create(
            model=self.lcfg["model_name"],
            messages=messages,
            temperature=self.lcfg.get("temperature", 0.3),
            max_tokens=self.lcfg.get("max_tokens", 1024),
        )
        return resp.choices[0].message.content

    # ---------- 后端二:本地 transformers + LoRA ----------
    def _chat_transformers(self, messages):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        if self._hf is None:
            base = self.lcfg["base_model_path"]
            tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            adapter = resolve_path(self.lcfg["lora_adapter_path"])
            try:
                model = PeftModel.from_pretrained(model, adapter)
            except Exception as e:
                print(f"[LLM] 未加载 LoRA 适配器({e}),使用 base 模型。")
            model.eval()
            self._hf = (model, tok)

        model, tok = self._hf
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=self.lcfg.get("max_tokens", 1024),
            temperature=self.lcfg.get("temperature", 0.3),
            do_sample=self.lcfg.get("temperature", 0.3) > 0,
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(gen, skip_special_tokens=True)

    def chat(self, messages):
        if self.backend == "transformers":
            return self._chat_transformers(messages)
        return self._chat_openai(messages)

    def ask(self, system: str, user: str):
        return self.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
