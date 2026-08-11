import json
import re
from pathlib import Path
from typing import Tuple, List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class DocSplit(BaseNode):
    name = "doc_split"

    def process(self, state) -> dict:
        """流程：加载输入→按MD标题初切→长切短合→统计输出→结果备份"""
        md_content, file_title = self._1_get_inputs(state)

        sections, title_count, line_num = self._2_split_by_titles(md_content, file_title)
        if title_count == 0:
            sections = [{
                "title": "",
                "content": md_content,
                "file_title": file_title
            }]

        short_sections = []
        for section in sections:
            sub_sections = self._split_long_section(section)
            short_sections.extend(sub_sections)

        final_sections = self._merge_short_sections(short_sections)
        for sec in final_sections:
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title") or ""

        chunk_num = len(sections)
        self.logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)
        self.logger.info(f"MD原始文本总行数：{line_num}")
        self.logger.info(f"最终生成Chunk数量：{chunk_num}")

        backup_path = PROJECT_ROOT / "out" / state.get("file_title") / "chunks.json"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(final_sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Chunk结果备份成功，备份文件路径：{backup_path}")

        state["chunks"] = final_sections
        return state

    def _1_get_inputs(self, state: ImportGraphState) -> Tuple[str, str]:
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        md_content = state.get("md_content")
        if not md_content:
            raise StateFieldError(field_name="md_content", message="文件内容不能为空", expected_type=str)

        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        return md_content, file_title

    def _2_split_by_titles(self, content: str, file_title: str) -> Tuple[List[Dict[str, str]], int, int]:
        """
        return: (
            [{
                "title": current_title,
                "content": "\n".join(current_lines), 
                "file_title": file_title,
            }],
            title_count, 
            len(lines)
        )
        """
        title_pattern = r'\s*#{1,6}\s+.+'
        lines = content.split("\n")  # ?
        sections = []  # 章节列表
        title_count = 0  # 标题数量
        current_title = ""  # 当前章节的标题
        current_lines = []  # 当前标题和下一个标题之间的文本内容
        in_code_block = False

        def _flush_section():
            if not current_lines:
                return
            sections.append({
                "title": current_title,
                "content": "\n".join(current_lines),  # ?
                "file_title": file_title,
            })

        for line in lines:
            striped_line = line.strip()
            # 处理代码块
            code_block = re.match(r'^(`{3,}|~{3,})$', striped_line)
            if code_block:
                marker = code_block.group(1)
                if not in_code_block:
                    in_code_block = True
                    code_block_start = marker
                elif in_code_block and striped_line == code_block_start:
                    in_code_block = False
                    code_block_start = None

                current_lines.append(line)
                continue
            # 处理标题行、正文行
            is_valid_title = (not in_code_block) and re.match(title_pattern, line)
            if is_valid_title:
                #遇到标题行则先将上一个片段写入section
                _flush_section()
                current_title = striped_line
                current_lines = [current_title]
                title_count += 1
            else:
                current_lines.append(striped_line)

        _flush_section()

        return sections, title_count, len(lines)

    def _split_long_section(self, section: Dict[str, str]) -> List[Dict[str, str]]:
        content = section.get("content")
        if len(content) <= self.config.max_content_length:
            return [section]

        title = section.get("title")
        prefix = f"{title}\n" if title else ""
        available_len = self.config.max_content_length - len(prefix)
        if available_len <= 0:
            self.logger.warning("章节标题过长，无法切分")
            return [section]

        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title) + len(title): ].lstrip()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len, # 切片长度（扣除标题）
            chunk_overlap=0, 
            separators=["\n\n", "\n", "。", "？", "！", "；", ".", "?", "!", ";", " ",]
        )
        sub_sections = []
        chunks = splitter.split_text(body)
        for idx, chunk in enumerate(chunks, start=1):
            text = chunk.strip()
            if not text:
                continue

            sub_text = (prefix + text).strip()
            sub_title = f"{title}-{idx}"

            sub_sections.append(
                {
                    "parent_title": title,
                    "title": sub_title,
                    "content": sub_text,
                    "part": idx,
                    "file_title": section.get("file_title")
                }
            )
        return sub_sections

    def _merge_short_sections(self, sections: List[Dict[str,str]]) -> List[Dict[str,str]]:
        if not sections:
            return []

        merged_sections = []
        current_chunk = None
        for section in sections:
            if current_chunk is None:
                current_chunk = section  # 第一个section
                continue
            is_current_short = len(current_chunk["content"]) < self.config.min_content_length
            is_same_parent = current_chunk.get("parent_title") == section.get("parent_title")

            if is_current_short and is_same_parent:
                parent_title = section.get("parent_title")
                section_content = section.get("content")

                if parent_title and section_content.startswith(parent_title):
                    section_content = section_content[len(parent_title):].lstrip()
                current_chunk["content"] += "\n\n" + section_content

                if "part" in section:
                    current_chunk["part"] = section["part"]
            else:
                # 保存当前段落
                merged_sections.append(current_chunk)
                current_chunk = section

        if current_chunk is not None:
            merged_sections.append(current_chunk)

        return merged_sections

if __name__ == "__main__":
    setup_logging()
    md_path = r"/Users/lyinlu/PycharmProjects/KnowledgeBase/out/华为擎云 M272Q 用户指南-(XSN-27QBZ,02,zh-cn)/华为擎云 M272Q 用户指南-(XSN-27QBZ,02,zh-cn)_new.md"
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    init_state= {
        "md_path": md_path,
        "md_content": md_content,
        "file_title": "华为擎云 M272Q 用户指南-(XSN-27QBZ,02,zh-cn)"
        }
    doc_split = DocSplit()
    result = doc_split.process(init_state)
    json_res = json.dumps(result, ensure_ascii=False, indent=4)
    print(json_res)