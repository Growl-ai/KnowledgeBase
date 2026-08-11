import re
from typing import List, Dict, Tuple

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from utils.llm import get_llm
from utils.mongo import save_chat_message
from utils.prompt import ANSWER_PROMPT
from utils.sse import push_to_session, SSEEvent
from utils.task_trace import add_done_task, set_task_result
from utils.logger import logger

MAX_CONTEXT_CHARS = 12000
class AnswerOutput(NodeBase):

    name: str = "answer_output"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        answer_exists = self._1_check_answer(state)
        if not answer_exists:
            prompt = self._2_construct_prompt(state)
            state["prompt"] = prompt
            self._3_generate_response(state, prompt)

        image_urls = self._extract_images_from_docs(state.get("reranked_docs") or [])
        if state.get("answer"):
            logger.info("---写入MongoDB历史记录---")
            self._4_write_history(state, image_urls=image_urls)

        add_done_task(state['session_id'], self.name, state.get("is_stream"))
        logger.info(f"---发送 final 事件---图片为：{image_urls}")
        if state.get("is_stream"):
            push_to_session(
                state['session_id'],
                SSEEvent.FINAL,
                {
                    "answer": state["answer"],
                    "status": "completed",
                    "image_urls": image_urls  # 发送图片URL给前端
                }
            )

        logger.info("---node_answer_output 节点处理结束---")
        return state

    def _1_check_answer(self, state):
        answer = state.get("answer")
        is_stream = state.get("is_stream")
        if answer:
            if is_stream:
                logger.info("---Step 1: 发现已有答案，执行流式推送---")
                push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
            else:
                set_task_result(state["session_id"], "answer", answer)
            return True
        else:
            return False


    def _2_construct_prompt(self, state):
        char_budget = MAX_CONTEXT_CHARS
        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state["item_names"]
        context_str, char_budget = self._format_reranked_docs(
            state.get("reranked_docs") or [], char_budget
        )
        history_str, char_budget = self._format_chat_history(
            state.get("history") or [], char_budget
        )
        item_names_str = ", ".join(item_names) if item_names else "无指定商品"
        prompt = ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str if history_str else "暂无历史对话",
            item_names=item_names_str,
            question=question,
        )
        logger.info(f"组装后的提示词为：{prompt}")
        return prompt

    def _3_generate_response(self, state, prompt):
        logger.info("---Step 3: 开始生成回答 (LLM Generation)---")
        llm = get_llm()
        session_id = state.get("session_id")
        is_stream = state.get("is_stream")

        if is_stream:
            logger.info(f"模式: 流式输出 (Streaming), Session: {session_id}")
            final_text = ""
            try:
                # 使用 stream 方法进行流式生成
                for chunk in llm.stream(prompt):
                    delta = getattr(chunk, "content", "") or ""
                    if delta:
                        final_text += delta
                        # 将增量内容放入队列
                        push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})

                logger.info(f"流式输出完成，总长度: {len(final_text)}")

            except Exception as e:
                logger.error(f"流式生成出错: {e}", exc_info=True)
                # 发生错误时，尝试推送到前端
                push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})

            state["answer"] = final_text
        else:
            # 非流式直接调用
            logger.info(f"模式: 非流式输出 (Blocking), Session: {session_id}")
            try:
                response = llm.invoke(prompt)
                content = response.content
                state["answer"] = content
                set_task_result(session_id, "answer", content)
                logger.info(f"生成回答完成，长度: {len(content)}")
            except Exception as e:
                logger.error(f"生成回答出错: {e}", exc_info=True)
                state["answer"] = "抱歉，生成回答时出现错误。"

        return state

    def _extract_images_from_docs(self, docs):
        images = []
        seen = set()  # 用于去重，避免同一张图片重复出现
        if not docs:
            return []
        md_img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')

        logger.info(f"开始提取图片，待处理文档数: {len(docs)}")

        for i, doc in enumerate(docs):
            # 1. 优先检查 url 字段 (主要针对 Web Search 结果)
            url = (doc.get("url") or "").strip()
            if url:
                # 简单后缀判断：确保是静态图片资源
                if url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg')):
                    if url not in seen:
                        logger.debug(f"文档[{i}] 发现图片 URL (字段): {url}")
                        seen.add(url)
                        images.append(url)

            # 2. 检查 text 字段中的 Markdown 图片 (主要针对 Local Chunk)
            text = (doc.get("content") or "").strip()
            if text:
                matches = md_img_pattern.findall(text)
                for img_url in matches:
                    img_url = img_url.strip()
                    if img_url and img_url not in seen:
                        logger.debug(f"文档[{i}] 正文发现 Markdown 图片: {img_url}")
                        seen.add(img_url)
                        images.append(img_url)

            logger.info(f"图片提取完成，共找到 {len(images)} 张唯一图片: {images}")
            return images

    def _4_write_history(self, state, image_urls):
        session_id = state.get("session_id", "default")
        answer = (state.get("answer") or "").strip()
        item_names = state.get("item_names") or []

        try:
            if answer:
                save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    text=answer,
                    rewritten_query="",
                    item_names=item_names,
                    image_urls=image_urls,
                    message_id=None
                )
        except Exception as e:
            # 写历史失败不应影响主链路
            logger.error(f"写入Mongo历史记录失败: {e}")

        return state

    def _format_reranked_docs(self, reranked_docs: List[Dict], char_budget) -> Tuple[str, int]:
        formatted_lines = []
        used_chars = 0
        for idx, doc in enumerate(reranked_docs, start=1):
            content = doc.get("content")
            meta_tags = [f"[{idx}]"]
            for field, template in [
                ("source", "[source={}]"),
                ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"),
                ("title", "[title={}]"),
            ]:
                field_value = str(doc.get(field)).strip()
                if field_value:
                    meta_tags.append(template.format(field_value))

            relevance_score = doc.get("score")
            if relevance_score is not None:
                meta_tags.append(f"[score={float(relevance_score):.4f}]")

            doc_entry = " ".join(meta_tags) + "\n" + content

            if used_chars + len(doc_entry) > char_budget:
                break

            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars

    def _format_chat_history(self, chat_history: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化历史对话"""
        formatted_lines = []
        used_chars = 0

        role_label_map = {"user": "用户", "assistant": "助手"}

        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue

            formatted_line = f"{role_label_map[role]}: {text}"
            used_chars += len(formatted_line) + 1

            if used_chars > char_budget:
                return "\n".join(formatted_lines), char_budget - used_chars

            formatted_lines.append(formatted_line)

        return "\n".join(formatted_lines), char_budget - used_chars