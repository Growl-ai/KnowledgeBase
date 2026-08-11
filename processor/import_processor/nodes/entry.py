import json
from pathlib import Path

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState


class Entry(BaseNode):
    name = "entry"
    def process(self, state: ImportGraphState) -> dict:
        """
        根据文档类型路由分发。
        1. 获取文件路径
        2. 判断文件类型
        3. 设置标记，更新state
        :param state: 文档路径
        :return: 是否为PDF、是否为MD、PDF/MD文档路径、文档标题
        """
        import_file_path = state.get("import_file_path")
        if not import_file_path:  # 判断字段是否为空
            raise StateFieldError(node_name=self.name, field_name="import_file_path", expected_type=str)
        file_path_obj = Path(import_file_path)
        if not file_path_obj.exists():  # 判断文件路径是否存在
            raise FileProcessingError(f"{import_file_path}文件不存在", node_name=self.name)
        if file_path_obj.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif file_path_obj.suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        else:
            raise FileProcessingError(f"{import_file_path}文件类型不支持", node_name=self.name)
        state["file_title"] = file_path_obj.stem

        return state

if __name__ == "__main__":
    entry = Entry()
    init_state = {"import_file_path": r"/Users/lyinlu/PycharmProjects/KnowledgeBase/assets/华为擎云 M272Q 用户指南-(XSN-27QBZ,02,zh-cn).pdf"}
    result = entry.process(init_state)
    json_res = json.dumps(result, ensure_ascii=False, indent=4)
    print(json_res)