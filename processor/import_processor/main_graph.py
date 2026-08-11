import json

from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.import_processor.nodes.bge_embed import BGEEmbed
from processor.import_processor.nodes.doc_split import DocSplit
from processor.import_processor.nodes.entry import Entry
from processor.import_processor.nodes.item_name_extract import ItemNameExtract
from processor.import_processor.nodes.md_img import MDImg
from processor.import_processor.nodes.milvus_store import MilvusStore
from processor.import_processor.nodes.pdf_to_md import PDFToMD
from processor.import_processor.state import ImportGraphState


class KBImportWorkflow:
    def __init__(self):
        """初始化工作流"""
        self._compiled_graph = None

    @staticmethod
    def entry_route(state: ImportGraphState):
        """入口路由"""
        if state.get("is_pdf_read_enabled"):
            return "pdf_to_md"
        elif state.get("is_md_read_enabled"):
            return "md_img"
        else:
            return END

    def build_graph(self):
        """构建工作流图"""
        graph = StateGraph(ImportGraphState)
        graph.add_node("entry", Entry())
        graph.add_node("pdf_to_md", PDFToMD())
        graph.add_node("md_img", MDImg())
        graph.add_node("doc_split", DocSplit())
        graph.add_node("item-name_extract", ItemNameExtract())
        graph.add_node("bge_embed", BGEEmbed())
        graph.add_node("milvus_store", MilvusStore())

        graph.set_entry_point("entry")
        graph.add_conditional_edges(
            "entry",
            self.entry_route,
            {
                "pdf_to_md": "pdf_to_md",
                "md_img": "md_img",
                END: END
            }
        )
        graph.add_edge("pdf_to_md", "md_img")
        graph.add_edge("md_img", "doc_split")
        graph.add_edge("doc_split", "item-name_extract")
        graph.add_edge("item-name_extract", "bge_embed")
        graph.add_edge("bge_embed", "milvus_store")
        graph.add_edge("milvus_store", END)

        return graph.compile()


    def graph(self):
        """懒加载：只在第一次使用时编译图"""
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph()
        return self._compiled_graph

    def run(self, state: ImportGraphState, stream: bool=False):
        """运行工作流"""
        app = self.graph()
        if stream:
            return app.stream(state)
        else:
            return app.invoke(state)


if __name__ == "__main__":
    init_state = {
        "import_file_path": r"/Users/lyinlu/PycharmProjects/KnowledgeBase/assets/hak180产品安全手册.pdf",
        "file_dir": r"/Users/lyinlu/PycharmProjects/KnowledgeBase/out"
    }
    workflow = KBImportWorkflow()
    result = workflow.run(init_state)
    json_res = json.dumps(result, ensure_ascii=False, indent=4)
    print(json_res)
    workflow.graph().get_graph().print_ascii()