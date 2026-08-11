import os

import dashscope
from http import HTTPStatus
from dotenv import load_dotenv
load_dotenv()

def rerank_docs(query, documents):
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    resp = dashscope.TextReRank.call(
        model=os.getenv("QWEN_RERANK_MODEL", "qwen3-rerank"),
        query=query,
        documents=documents,
        top_n=len(documents),
        return_documents=False,
        instruct="基于给定的查询，检索能够回答该查询的相关文档"
    )
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"DashScope qwen3-rerank API 调用失败，状态码：{resp.status_code}")

    results = resp.output.get("results", [])

    scores = [0.0] * len(results)
    for result in results:
        score = result.get("relevance_score")
        index = result.get("index")
        scores[index] = score

    return  scores

if __name__ == '__main__':
    rerank_docs()