import json
import logging
from typing import Tuple, List, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding import generate_embeddings
from utils.llm import get_llm
from utils.milvus import get_milvus_client, escape_milvus_string
from utils.prompt import ITEM_NAME_EXTRACT_TEMPLATE, ITEM_NAME_EXTRACT_SYSTEM_PROMPT


class ItemNameExtract(BaseNode):
    name = "item_name_extract"

    def process(self, state) -> dict:
        """
        1. 获取输入，非空字段校验
        2. 构建大模型上下文
        3. 调用大模型识别商品名称
        4. 回填商品名称到状态和切片
        5. 生成商品名称的稠密/稀疏向量
        6. 将数据存入Milvus向量数据库
        :param state:
        :return:
        """

        file_title, chunks = self._1_get_inputs(state)
        context = self._2_build_context(chunks)
        item_name = self._3_call_llm(file_title, context)

        state["item_name"] = item_name
        for chunk in chunks:
            chunk["item_name"] = item_name
        state["chunks"] = chunks

        vectors = {}
        if not item_name:
            vectors["dense"] = []
            vectors["sparse"] = []
        vectors["dense"], vectors["sparse"] = generate_embeddings([item_name])
        self._4_save_to_milvus(file_title, item_name, vectors["dense"][0], vectors["sparse"][0])
        self.logger.info(f"--- 识别完成: {item_name} ---")

        return state

    def _1_get_inputs(self, state: ImportGraphState) -> Tuple[str, List[Dict]]:
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        chunks = state.get("chunks")
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        return file_title, chunks

    def _2_build_context(self, chunks: List[Dict]) -> str:
        parts: List[str] = []
        total_chars = 0
        for idx, chunk in enumerate(chunks[:self.config.item_name_chunk_k], start=1):

            # 1. 提取前k的切片，避免上下文过长
            chunk_title = chunk.get("title").strip()
            chunk_content = chunk.get("content").strip()

            # 2. 格式化切片
            piece = f"【切片{idx}】\n标题{chunk_title}\n内容：{chunk_content}"
            parts.append(piece)

            # 3. 计算累计的字符数
            total_chars += len(piece)

            # 4. 判断是否需要继续切分
            if total_chars > self.config.item_name_chunk_size:
                self.logger.warning(f"累计字符数{total_chars}已超过限制{self.config.item_name_chunk_size}，停止切分")
                break

        # 5. 使用换行符对切分后的片段进行连接
        context = "\n\n".join(parts).strip()

        # 6. 对返回结果进行二次截断
        final_context = context[: self.config.item_name_chunk_size]
        return final_context

    def _3_call_llm(self, file_title: str, context: str) -> str:
        if not context:
            return file_title
        try:
            prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
                file_title=file_title,
                context=context
            )
            llm = get_llm()
            messages = [
                SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            response = llm.invoke(messages)
            item_name = response.content
            item_name = item_name.replace(" ", "")
            if not item_name:
                return file_title

            return item_name
        except Exception as e:
            self.logger.error(f"大模型调用异常：{e}")
            return file_title

    def _4_save_to_milvus(self, file_title: str, item_name: str, dense_vector, sparse_vector):
        milvus_client = get_milvus_client()
        if not milvus_client:
            self.logger.warning("Milvus 连接失败，跳过主体名称保存")
            return
        try:
            collection_name = self.config.item_name_collection
            if not milvus_client.has_collection(collection_name):
                self._create_item_name_collection(collection_name, milvus_client)
            file_title = escape_milvus_string(file_title)
            milvus_client.delete(collection_name=collection_name, filter=f"file_title=='{file_title}'")

            data = {
                "file_title": file_title,
                "item_name": item_name
            }

            if dense_vector is not None:
                data["dense_vector"] = dense_vector
            if sparse_vector is not None:
                data["sparse_vector"] = sparse_vector

            milvus_client.insert(collection_name=collection_name, data=data)
        except Exception as e:
            self.logger.warning(f"数据存入Milvus失败，跳过主体名称保存，错误原因:{str(e)}")

    def _create_item_name_collection(self, collection_name, milvus_client):
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="ID", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128}
        )
        index_params.add_index(
            field_name="sparse_vector",  # 字段名
            index_name="sparse_vector_index",  # 索引名
            index_type="SPARSE_INVERTED_INDEX",  # 索引类型
            metric_type="IP",  # 相似度计算方式（内积）
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "normalize": True,
                "quantization": "none"
            })
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )


if __name__ == "__main__":

    setup_logging()

    json_path = r"/Users/lyinlu/PycharmProjects/KnowledgeBase/out/华为擎云 M272Q 用户指南-(XSN-27QBZ,02,zh-cn)/chunks.json"
    with open(json_path, "r", encoding="utf-8") as f:
        chuncks_json = f.read()

    chuncks = json.loads(chuncks_json)

    init_state = {
        "chunks": chuncks,
        "file_title": "华为擎云 M272Q 用户指南-(XSN-27QBZ,02,zh-cn)"
    }

    # 执行核心处理流程
    item_name_extract = ItemNameExtract()
    result = item_name_extract(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))