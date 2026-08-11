import os

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from utils.embedding import generate_embeddings
from utils.llm import get_llm
from utils.logger import logger
from utils.milvus import create_hybrid_search_requests, get_milvus_client, hybrid_search
from utils.mongo import serialize_json
from utils.prompt import HYDE_PROMPT


class RetrieveHyde(NodeBase):

    name: str = "retrieve_hyde"

    def process(self, state: QueryGraphState):
        try:
            rewritten_query = state.get("rewritten_query")
            item_names = state.get("item_names")

            hyde_doc = self._1_generate_hyde_doc(rewritten_query)
            res = self._2_retrieve_hyde(
                rewritten_query=rewritten_query,
                hyde_doc=hyde_doc,
                item_names=item_names
            )
            return {
                "hyde_embedding_chunks": res,
                "hyde_doc": hyde_doc
            }

        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")
            return {}


    def _1_generate_hyde_doc(self, rewritten_query):
        try:
            llm = get_llm()
            gyde_prompt = HYDE_PROMPT.format(rewritten_query=rewritten_query)
            hyde_doc = llm.invoke(gyde_prompt).content

            return hyde_doc
        except Exception as e:
            logger.exception(f"LLM调用失败: {e}")
            raise e

    def _2_retrieve_hyde(self, rewritten_query, hyde_doc, item_names):
        try:
            combined_text = rewritten_query + " " + hyde_doc
            dense_vecs, sparse_vecs = generate_embeddings([combined_text])
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
            return search_res[0] if search_res else []

        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")
            raise e

if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "item_names": ["兄弟HAK180烫金机", "BrotherHAK-180烫金机"]  # 上一节点从向量数据库中匹配到的结果
    }
    retrieve_hyde = RetrieveHyde()
    result = retrieve_hyde(init_state)
    logger.info(serialize_json(result, indent=4))