import json

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState


class RRF(NodeBase):

    name: str = "rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        embedding_chunks = state.get("embedding_chunks") or []
        hyde_embedding_chunks = state.get("hyde_embedding_chunks") or []

        embedding_res = [doc.get("entity") for doc in embedding_chunks if isinstance(doc, dict)]
        hyde_embedding_res = [doc.get("entity") for doc in hyde_embedding_chunks if isinstance(doc, dict)]
        rrf_inputs = [
            (embedding_res, 1.0),
            (hyde_embedding_res, 1.0)
        ]
        rrf_merge_results = self._rrf_merge(rrf_inputs, max_results=5)
        rrf_chunks = [doc for doc, _ in rrf_merge_results]
        state["rrf_chunks"] = rrf_chunks

        return state

    def _rrf_merge(self, rrf_inputs, max_results, k: int = 60):
        merged_chunks = {}
        chunk_scores = {}
        for rrf_input, weight in rrf_inputs:
            for rank, chunk in enumerate(rrf_input, start=1):
                chunk_id = chunk.get("chunk_id")
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (k + rank)
                merged_chunks[chunk_id] = chunk
        unsorted_results = [(merged_chunks[chunk_id], score) for chunk_id, score in chunk_scores.items()]
        sorted_results = sorted(unsorted_results, key=lambda x: x[1], reverse=True)
        return sorted_results[:max_results]

if __name__ == "__main__":

    retrieve_res_path = r"/Users/lyinlu/PycharmProjects/KnowledgeBase/out/retrieve_res.json"
    with open(retrieve_res_path, "r") as f:
        res_dict = json.load(f)

    init_state = {
        "embedding_chunks": res_dict.get("embedding_chunks"),
        "hyde_embedding_chunks": res_dict.get("hyde_embedding_chunks")
    }
    rrf = RRF()
    result = rrf(init_state)
    print(json.dumps(result, ensure_ascii=False, indent=4))