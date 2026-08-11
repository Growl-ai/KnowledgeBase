import json
import os
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from utils.embedding import generate_embeddings
from utils.llm import get_llm
from utils.logger import logger
from utils.milvus import get_milvus_client, create_hybrid_search_requests, hybrid_search
from utils.mongo import get_recent_messages, save_chat_message, update_message_item_names
from utils.prompt import ITEM_NAME_CONFIRM_TEMPLATE, ITEM_NAME_CONFIRM_SYSTEM_PROMPT


class ItemNameConfirm(NodeBase):

    name: str = "item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        session_id, original_query = self._1_validate_param(state)
        logger.info(f"步骤1：参数校验通过")

        history = get_recent_messages(session_id)
        state["history"] = history
        logger.info("步骤2：state 的 history 信息更新成功,")

        message_id = save_chat_message(session_id, "user", original_query)
        logger.info(f"步骤3：用户初始消息已保存, ID: {message_id}")

        extract_res = self._4_extract_info(original_query, history)
        item_names = extract_res.get("item_names")
        rewritten_query = extract_res.get("rewritten_query", original_query)
        state["rewritten_query"] = rewritten_query

        align_result = {}
        if len(item_names) > 0:
            retrieve_results = self._5_embedding_and_retrieve(item_names)
            align_result = self._6_align_by_score(retrieve_results)
        else:
            logger.info("Node: 未提取到商品名，跳过向量检索")

        state = self._7_update_msgs_state(state, align_result, history)

        if state.get("answer"):
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=state.get("answer"),
                item_names=state.get("item_names")
            )
        save_chat_message(
            session_id=session_id,
            role="user",
            text=state.get("original_query"),
            item_names=state.get("item_names"),
            rewritten_query=rewritten_query,
            message_id=message_id
        )

        return state

    def _1_validate_param(self, state):

        original_query = state.get("original_query")
        if not original_query:
            raise ValueError("参数 original_query 不能为空")

        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("参数 session_id 不能为空")

        return session_id, original_query

    def _4_extract_info(self, original_query, history):
        try:
            llm = get_llm(json_mode=True)
            history_text = ""
            for msg in history:
                role = msg.get("role")
                content = msg.get("text")
                history_text += f"{role}: {content}\n"

            user_prompt = ITEM_NAME_CONFIRM_TEMPLATE.format(history_text=history_text, query=original_query)
            messages = [
                SystemMessage(content=ITEM_NAME_CONFIRM_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            response = llm.invoke(messages)
            content = response.content
            if content.startswith("```json"):
                content.replace("```json", "").replace("```", "")

            result = json.loads(content)  # JSON字符串转换为字典
            if "item_names" not in result:
                logger.warning("大模型返回结果缺少item_names字段")
                result = {"item_names": []}

            if "rewritten_query" not in result:
                logger.warning("大模型返回结果缺少rewritten_query字段")
                result = {"rewritten_query": original_query}
            # result["item_names"] = [name.replace(" ", "") for name in result.get("item_names")]
            return result
        except Exception as e:
            logger.error(f"大模型调用异常：{e}")
            return {"item_names": [], "rewritten_query": None}

    def _5_embedding_and_retrieve(self, item_names):
        dense_vecs, sparse_vecs = generate_embeddings(item_names)
        try:
            milvus_client = get_milvus_client()
            results = []
            if not milvus_client:
                logger.warning("Milvus 获取失败，跳过向量化处理")
                return results
            collection_name = os.getenv("ITEM_NAME_COLLECTION")
            # 检索
            for i in range(len(item_names)):
                dense_vector = dense_vecs[i]
                sparse_vector = sparse_vecs[i]
                reqs = create_hybrid_search_requests(dense_vector=dense_vector, sparse_vector=sparse_vector)
                search_res = hybrid_search(
                    client=milvus_client,
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker_weights=(0.8, 0.2),
                    output_fields=["item_name"]
                )
                matches = []
                if search_res and len(search_res) > 0:
                    for hit in search_res[0]:
                        matches.append({
                            "item_name": hit.get("entity").get("item_name"),
                            "score": hit.get("distance")
                        })
                results.append({
                    "extracted_name": item_names[i],
                    "matches": matches
                })
            return  results
        except Exception as e:
            logger.warning(f"混合检索异常：{e}")


    def _6_align_by_score(self, retrieve_results):
        confirmed: List[str] = []
        options: List[str] = []
        # 遍历每一个item_name及其检索结果
        for res in retrieve_results:
            extracted_name = res.get("extracted_name")
            matches = res.get("matches")
            if not matches:
                continue
            high = [m for m in matches if m.get("score") > 0.85]
            mid = [m for m in matches if m.get("score") >= 0.65]
            if len(high) > 0:
                for m in high:
                    confirmed.append(m.get("item_name"))
                continue
            if len(mid) > 0:
                for m in mid[:3]:
                    options.append(m.get("item_name"))
        return {
            "confirmed": list(set(confirmed)),
            "options": list(set(options))
        }

    def _7_update_msgs_state(self, state, align_result, history):
        confirmed = align_result.get("confirmed", [])
        options = align_result.get("options", [])
        if confirmed:
            ids_to_update = [str(msg.get("_id")) for msg in history if not msg.get("item_names")]
            if ids_to_update:
                update_message_item_names(ids_to_update, confirmed)
            state["item_names"] = confirmed
            state["answer"] = ""
            return state

        if options:
            options_str = "，".join(options)
            state["answer"] = f"您是想问以下哪个产品：{options_str}？请明确一下具体的产品型号和名称。"
            state["item_names"] = []
            return state

        state["answer"] = "未找到相关产品，请提供准确的商品品牌、型号和名称。"
        state["item_names"] = []
        return state


if __name__ == "__main__":
    init_state = {
        "session_id": "session_001",
        "original_query": "怎么调节转印温度？"
    }
    item_name_confirm = ItemNameConfirm()
    result = item_name_confirm(init_state)
    logger.info(result)
