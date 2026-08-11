import asyncio
import json
import os

from agents.mcp import MCPServerStreamableHttp
from dotenv import load_dotenv

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from utils.logger import logger
from utils.mongo import serialize_json


load_dotenv()

class WebSearchMcp(NodeBase):

    name: str = "web_search_mcp"

    def process(self, state: QueryGraphState):
        query = state.get("rewritten_query")
        result = asyncio.run(self._call_web_mcp(query))
        web_docs = []
        if result:
            json_str = result.content[0].text
            pages = json.loads(json_str).get("page")
            for page in pages:
                snippet = page.get("snippet")
                url = page.get("url")
                title = page.get("title")
                web_docs.append({
                    "snippet": snippet,
                    "url": url,
                    "title": title
                })
        if web_docs:
            return {"web_search_docs": web_docs}
        return {}

    async def _call_web_mcp(self, query):
        mcp_base_url = os.getenv("MCP_DASHSCOPE_BASE_URL")
        api_key = os.getenv("DASHSCOPE_API_KEY")
        mcp_client = MCPServerStreamableHttp(
            name="Streamable HTTP Python Server",
            params={
                "url": mcp_base_url,
                "headers": {"Authorization": f"Bearer {api_key}"},
                "timeout": 10,
            },
            cache_tools_list=True,
            max_retry_attempts=3,
        )
        try:
            await mcp_client.connect()
            result = await mcp_client.call_tool(
                tool_name="bailian_web_search",
                arguments={
                    "query": query,
                    "count": 5
                }
            )
            return result
        finally:
            await mcp_client.cleanup()


if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？"
    }

    # 执行节点的业务调用
    node_web_search_mcp = WebSearchMcp()
    result = node_web_search_mcp(init_state)
    logger.info(serialize_json(result, indent=4))