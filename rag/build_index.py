"""
构建向量索引:读取知识库 -> 切分 -> BGE-M3 编码 -> 写入 FAISS。

用法:
    python -m rag.build_index
"""
import os
import json
import glob
import pickle

import numpy as np
import faiss

from config import load_config, resolve_path


def split_text(text: str, chunk_size: int, overlap: int):
    """按字符长度切分,保留重叠。优先按段落/句子边界切,避免切碎语义。"""
    # 先按段落切,再对超长段落滑窗切分
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end])
                start += chunk_size - overlap
    return chunks


def load_documents(knowledge_dir: str):
    """读取知识库目录下的 .md / .txt,返回 [{'text':..., 'source':...}]。"""
    docs = []
    patterns = ["*.md", "*.txt"]
    for pat in patterns:
        for fp in glob.glob(os.path.join(knowledge_dir, "**", pat), recursive=True):
            with open(fp, "r", encoding="utf-8") as f:
                docs.append({"text": f.read(), "source": os.path.basename(fp)})
    return docs


def build_index(cfg: dict = None):
    from FlagEmbedding import BGEM3FlagModel

    cfg = cfg or load_config()
    rcfg = cfg["rag"]

    knowledge_dir = resolve_path(rcfg["knowledge_dir"])
    index_dir = resolve_path(rcfg["index_dir"])
    os.makedirs(index_dir, exist_ok=True)

    print(f"[build_index] 加载嵌入模型 {rcfg['embed_model']} ...")
    model = BGEM3FlagModel(rcfg["embed_model"], use_fp16=rcfg.get("use_fp16", True))

    print(f"[build_index] 读取知识库 {knowledge_dir} ...")
    docs = load_documents(knowledge_dir)
    if not docs:
        raise RuntimeError(f"知识库为空: {knowledge_dir}")

    # 切分
    records = []  # 每条:{'text','source'}
    for d in docs:
        for chunk in split_text(d["text"], rcfg["chunk_size"], rcfg["chunk_overlap"]):
            records.append({"text": chunk, "source": d["source"]})
    print(f"[build_index] 共 {len(docs)} 篇文档,切分为 {len(records)} 个片段")

    # 编码(取 dense 向量)
    texts = [r["text"] for r in records]
    embeddings = model.encode(texts, batch_size=16, max_length=rcfg["chunk_size"] + 64)["dense_vecs"]
    embeddings = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)  # 归一化后用内积 = 余弦相似度

    # 建 FAISS 索引
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # 落盘
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "records.pkl"), "wb") as f:
        pickle.dump(records, f)
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"dim": int(dim), "count": len(records),
                   "embed_model": rcfg["embed_model"]}, f, ensure_ascii=False, indent=2)

    print(f"[build_index] 完成。索引已保存到 {index_dir}(维度 {dim}, 片段 {len(records)})")


if __name__ == "__main__":
    build_index()
