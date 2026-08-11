import os

from dotenv import load_dotenv
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from utils.logger import logger
load_dotenv()
_milvus_client = None

def get_milvus_client():

    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client
    milvus_url = os.getenv("MILVUS_URL", "")
    _milvus_client = MilvusClient(milvus_url)
    return _milvus_client

def escape_milvus_string(value: str) -> str:
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    return value

def create_hybrid_search_requests(
        dense_vector,
        sparse_vector,
        dense_params=None,
        sparse_params=None,
        expr=None,
        limit=5
):
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )
    return [dense_req, sparse_req]


def hybrid_search(
        client,
        collection_name,
        reqs,
        ranker_weights=(0.5, 0.5),
        norm_score=True,
        limit=5,
        output_fields=None,
        search_params=None
):
    if output_fields is None:
        output_fields = ["item_name"]
    try:
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        logger.info(f"Milvus混合搜索完成，集合[{collection_name}]共检索到{len(res[0])}条结果")
        return res

    except Exception as e:
        logger.error(f"Milvus混合搜索执行失败，集合[{collection_name}]：{str(e)}", exc_info=True)
        return None