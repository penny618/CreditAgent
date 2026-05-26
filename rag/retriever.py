"""
检索器:dense 召回 (BGE-M3 + FAISS) -> 精排 (BGE-reranker)。
返回结果带 source,实现可溯源问答。
"""
import os
import pickle

import numpy as np
import faiss

from config import load_config, resolve_path


class Retriever:
    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        rcfg = self.cfg["rag"]
        index_dir = resolve_path(rcfg["index_dir"])

        idx_file = os.path.join(index_dir, "faiss.index")
        rec_file = os.path.join(index_dir, "records.pkl")
        if not (os.path.exists(idx_file) and os.path.exists(rec_file)):
            raise RuntimeError(
                f"未找到索引,请先运行 `python -m rag.build_index`(目录: {index_dir})"
            )

        self.index = faiss.read_index(idx_file)
        with open(rec_file, "rb") as f:
            self.records = pickle.load(f)

        self.top_k = rcfg["top_k"]
        self.rerank_top_n = rcfg["rerank_top_n"]
        self.use_fp16 = rcfg.get("use_fp16", True)

        # 延迟加载模型(首次检索时再载入,加快导入速度)
        self._embed_model_name = rcfg["embed_model"]
        self._rerank_model_name = rcfg["rerank_model"]
        self._embed_model = None
        self._reranker = None          # (tokenizer, model) 元组;原生 cross-encoder 精排
        self._rerank_disabled = False  # 加载/调用失败后置位,后续仅按 dense 召回排序

    def _ensure_models(self):
        from FlagEmbedding import BGEM3FlagModel
        if self._embed_model is None:
            self._embed_model = BGEM3FlagModel(self._embed_model_name, use_fp16=self.use_fp16)
        # 用 transformers 原生 API 加载 reranker,避开 FlagReranker 对已被
        # transformers>=5 移除的 tokenizer.prepare_for_model 的依赖。
        if self._reranker is None and not self._rerank_disabled:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                tok = AutoTokenizer.from_pretrained(self._rerank_model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self._rerank_model_name)
                model.eval()
                if torch.cuda.is_available():
                    model = model.to("cuda")
                    if self.use_fp16:
                        model = model.half()
                self._reranker = (tok, model)
            except Exception as e:
                print(f"[retriever] reranker 加载失败,改用 dense 召回排序({type(e).__name__}: {e})")
                self._rerank_disabled = True

    def _rerank_scores(self, query, candidates):
        """用 cross-encoder 对 (query, passage) 打分,sigmoid 归一化到 0~1。"""
        import torch
        tok, model = self._reranker
        queries = [query] * len(candidates)
        passages = [c["text"] for c in candidates]
        device = next(model.parameters()).device
        with torch.no_grad():
            inputs = tok(queries, passages, padding=True, truncation=True,
                         max_length=512, return_tensors="pt").to(device)
            logits = model(**inputs, return_dict=True).logits.view(-1).float()
            scores = torch.sigmoid(logits)
        return scores.cpu().tolist()

    def search(self, query: str):
        """返回精排后的 top_n 片段: [{'text','source','score'}]。"""
        self._ensure_models()

        # 1) dense 召回
        q_emb = self._embed_model.encode([query])["dense_vecs"]
        q_emb = np.asarray(q_emb, dtype="float32")
        faiss.normalize_L2(q_emb)
        sims, idxs = self.index.search(q_emb, min(self.top_k, len(self.records)))
        candidates, dense_scores = [], []
        for i, s in zip(idxs[0], sims[0]):
            if i >= 0:
                candidates.append(self.records[i])
                dense_scores.append(float(s))

        if not candidates:
            return []

        # 2) rerank 精排;reranker 不可用时回退到 dense 余弦相似度排序
        scores = dense_scores
        if self._reranker is not None:
            try:
                scores = self._rerank_scores(query, candidates)
            except Exception as e:
                print(f"[retriever] rerank 调用失败,回退 dense 排序({type(e).__name__}: {e})")
                scores = dense_scores

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        results = []
        for c, s in ranked[: self.rerank_top_n]:
            results.append({"text": c["text"], "source": c["source"], "score": float(s)})
        return results

    @staticmethod
    def format_context(results):
        """把检索结果拼成带编号引用的上下文,供 LLM 引用溯源。"""
        if not results:
            return "(未检索到相关知识)"
        blocks = []
        for i, r in enumerate(results, 1):
            blocks.append(f"[{i}] (来源: {r['source']})\n{r['text']}")
        return "\n\n".join(blocks)
