import json
from typing import Any, Dict, List

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from utils.rerank import rerank_docs

RERANK_MAX_TOPK: int = 5
RERANK_MIN_TOPK: int = 2
RERANK_GAP_ABS: float = 0.3
RERANK_GAP_RATIO: float = 0.25

class Rerank(NodeBase):

    name: str = "rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        merged_docs: List[Dict[str, Any]] = self._1_merge_multi_source_docs(state)

        reranked_docs: List[Dict[str, Any]] = self._2_rerank_merged_docs(state, merged_docs)

        cutoff_docs = self._3_cliff_cutoff(reranked_docs)

        state['reranked_docs'] = cutoff_docs
        return state

    def _1_merge_multi_source_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """合并本地 RRF 结果和网络搜索结果为统一格式"""

        rrf_chunks = state.get("rrf_chunks")
        web_search_docs = state.get("web_search_docs")
        final_docs = []

        for rrf_doc in rrf_chunks:
            final_rrf_doc = {
                "chunk_id": rrf_doc.get("chunk_id"),
                "title": rrf_doc.get("item_name"),
                "content": rrf_doc.get("content"),
                "url": None,
                "source": "local"
            }
            final_docs.append(final_rrf_doc)

        for web_doc in web_search_docs:
            final_web_doc = {
                "chunk_id": None,
                "title": web_doc.get("title"),
                "content": web_doc.get("snippet"),
                "url": web_doc.get("url"),
                "source": "web"
            }
            final_docs.append(final_web_doc)

        return final_docs

    def _2_rerank_merged_docs(self, state: QueryGraphState, merged_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        user_query = state.get("rewritten_query")
        contents = [doc.get("content") for doc in merged_docs]
        rerank_scores = rerank_docs(user_query, contents)
        reranked_docs = [{**doc, "score": score} for doc, score in zip(merged_docs, rerank_scores)]
        sorted_docs = sorted(reranked_docs, key=lambda x: x["score"], reverse=True)
        return sorted_docs

    def _3_cliff_cutoff(self, ranked_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not ranked_docs:
            return []
        upper_bound = min(RERANK_MAX_TOPK, len(ranked_docs))
        lower_bound = min(RERANK_MIN_TOPK, upper_bound)
        cutoff_pos = upper_bound

        for idx in range(lower_bound - 1, upper_bound - 1):
            current_score = ranked_docs[idx].get("score")
            next_score = ranked_docs[idx + 1].get("score")

            if current_score is None or next_score is None:
                continue

            # 计算相邻文档的分数差
            abs_gap = current_score - next_score
            rel_gap = abs_gap / (abs(current_score) + 1e-6)

            if abs_gap >= RERANK_GAP_ABS or rel_gap >= RERANK_GAP_RATIO:
                cutoff_pos = idx + 1
                break

        return ranked_docs[:cutoff_pos]

if __name__ == "__main__":

    multi_search_res_path = r"/Users/lyinlu/PycharmProjects/KnowledgeBase/out/multi_search_res.json"
    with open(multi_search_res_path, "r") as f:
        res_dict = json.load(f)

    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "rrf_chunks": res_dict.get("rrf_chunks"),
        "web_search_docs": res_dict.get("web_search_docs")
    }
    rerank = Rerank()
    result = rerank(init_state)
    print(json.dumps(result, ensure_ascii=False, indent=4))