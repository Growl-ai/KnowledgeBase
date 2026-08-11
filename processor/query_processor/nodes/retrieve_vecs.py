import os

from dotenv import load_dotenv

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from utils.embedding import generate_embeddings
from utils.logger import logger
from utils.milvus import create_hybrid_search_requests, get_milvus_client, hybrid_search
from utils.mongo import serialize_json

load_dotenv()
class RetrieveVecs(NodeBase):

    name: str = "retrieve_vecs"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        try:
            query = state.get("rewritten_query")
            item_names = state.get("item_names")

            dense_vecs, sparse_vecs = generate_embeddings([query])
            dense_vec = dense_vecs[0]
            sparse_vec = sparse_vecs[0]

            expr = f'item_name in {item_names}'
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                expr=expr,
                limit=10
            )
            client = get_milvus_client()
            collection_name = os.getenv("CHUNKS_COLLECTION")
            search_res = hybrid_search(
                client=client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                output_fields=["chunk_id", "content", "item_name"]
            )
            return {"embedding_chunks": search_res[0] if search_res else []}
        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")
            return {}

if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "item_names": ["兄弟HAK180烫金机", "BrotherHAK-180烫金机"]  # 上一节点从向量数据库中匹配到的结果
    }
    retrieve_vecs = RetrieveVecs()
    result = retrieve_vecs(init_state)
    logger.info(serialize_json(result, indent=4))