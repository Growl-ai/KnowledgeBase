import os
from typing import List

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from dotenv import load_dotenv

load_dotenv()

_bge_m3 = None

def get_bge_m3():
    """
    获取全局单例的BGEM3EmbeddingFunction对象
    :return:  BGEM3EmbeddingFunction对象
    """
    global _bge_m3
    if _bge_m3 is not None:
        return _bge_m3

    _bge_m3 = BGEM3FlagModel(
        model_name_or_path=os.getenv("BGE_M3_PATH", ""),
        device=os.getenv("BGE_DEVICE"),
        use_fp16=os.getenv("BGE_FP16", "False")
    )
    return _bge_m3


def generate_embeddings(texts: List[str], batch_size: int = 8):
    model = get_bge_m3()
    embeddings = model.encode(texts, batch_size=batch_size, return_sparse=True, return_dense=True)

    dense_vecs = [vec.tolist() for vec in embeddings["dense_vecs"]]

    sparse_vecs = []
    for lexical_weights in embeddings["lexical_weights"]:
        sparse_vecs.append({int(k): float(v) for k, v in lexical_weights.items()})

    return dense_vecs, sparse_vecs


if __name__ == '__main__':
    texts = ["hello world", "hello china"]
    embeddings = generate_embeddings(texts)
    print(embeddings)