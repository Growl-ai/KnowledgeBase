from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.query_processor.nodes.answer_output import AnswerOutput
from processor.query_processor.nodes.item_name_confirm import ItemNameConfirm
from processor.query_processor.nodes.rerank import Rerank
from processor.query_processor.nodes.retrieve_hyde import RetrieveHyde
from processor.query_processor.nodes.retrieve_vecs import RetrieveVecs
from processor.query_processor.nodes.rrf import RRF
from processor.query_processor.nodes.web_search_mcp import WebSearchMcp
from processor.query_processor.state import QueryGraphState
from utils.logger import logger


class KBQueryWorkflow:

    def __init__(self):
        """初始化工作流：创建状态图、注册节点、定义路由规则"""
        self.workflow = StateGraph(QueryGraphState)
        self._init_nodes()
        self._register_nodes()
        self._build_graph()
        self._compiled_app = None

    def _init_nodes(self):
        self.item_name_confirm = ItemNameConfirm()
        self.retrieve_vecs = RetrieveVecs()
        self.retrieve_hyde = RetrieveHyde()
        self.web_search_mcp = WebSearchMcp()
        self.rrf = RRF()
        self.rerank = Rerank()
        self.answer_output = AnswerOutput()
        

    def _register_nodes(self):
        self.workflow.add_node("item_name_confirm", self.item_name_confirm)  # 确认主体
        self.workflow.add_node("multi_search", lambda x: x)
        self.workflow.add_node("retrieve_vecs", self.retrieve_vecs)  # 向量搜索
        self.workflow.add_node("retrieve_hyde", self.retrieve_hyde)  # 假设性答案向量搜索
        self.workflow.add_node("web_search_mcp", self.web_search_mcp)  # 联网搜索
        self.workflow.add_node("join", lambda x: {})  # 虚拟节点：多路搜索合并点
        self.workflow.add_node("rrf", self.rrf)  # 排序
        self.workflow.add_node("rerank", self.rerank)  # 重排
        self.workflow.add_node("answer_output", self.answer_output)

    def _build_graph(self):
        self.workflow.set_entry_point("item_name_confirm")

        # 2、注册条件路由边
        self.workflow.add_conditional_edges(
            "item_name_confirm",
            self._route_after_item_name_confirm,
            {
                "answer_output": "answer_output",
                "multi_search": "multi_search"
            }
        )

        # 3. 并发执行搜索
        self.workflow.add_edge("multi_search", "retrieve_vecs")
        self.workflow.add_edge("multi_search", "retrieve_hyde")
        self.workflow.add_edge("multi_search", "web_search_mcp")

        # 4. 多路搜索结果合并
        self.workflow.add_edge("retrieve_vecs", "join")
        self.workflow.add_edge("retrieve_hyde", "join")
        self.workflow.add_edge("web_search_mcp", "join")

        # 5. 合并 -> 排序 -> 重排 -> 生成 -> 结束
        self.workflow.add_edge("join", "rrf")
        self.workflow.add_edge("rrf", "rerank")
        self.workflow.add_edge("rerank", "answer_output")
        self.workflow.add_edge("answer_output", END)

    def _route_after_item_name_confirm(self, state: QueryGraphState) -> str:
        """主体名称确认后的条件路由函数"""
        if state.get("answer"):
            return "answer_output"
        return "multi_search"

    def compile(self):
        """编译工作流（公开方法，支持手动触发编译）"""
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app


    def run(self, init_state: QueryGraphState, stream: bool = False) -> QueryGraphState:
        """
        统一执行入口，支持切换invoke/stream
        :param init_state:  初始状态对象
        :param stream: 是否是流式输出
        :return: 执行完成后的状态对象
        """
        """"""
        if not self._compiled_app:
            self.compile()
        if stream:
            return self._compiled_app.stream(init_state)
        else:
            return self._compiled_app.invoke(init_state)

if __name__ == "__main__":
    init_state = {
        "session_id": "session_001",
        "original_query": "怎么调节180烫金机的温度"
    }

    # 创建流程对象
    workflow = KBQueryWorkflow()
    # for chunk in workflow.run(init_state, stream=True):
    #     logger.debug(chunk)

    for chunk in workflow.run(init_state, stream=False):
        logger.debug(chunk)

    # 打印工作流的ascii码图
    logger.info(workflow.compile().get_graph().draw_ascii())