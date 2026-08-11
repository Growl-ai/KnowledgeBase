import json
import logging
from typing import Dict, Any, List

from pymilvus import DataType

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, MilvusError
from utils.milvus import get_milvus_client, escape_milvus_string


class MilvusStore(BaseNode):
    name = "milvus_store"
    def process(self, state) -> dict:
        """
        将向量存储到 Milvus 数据库中。
        :param state:
        :return:
        """
        chunks, vector_dim = self._1_check_input(state)

        milvus_client = get_milvus_client()
        if not milvus_client:
            self.logger.error("Milvus 连接失败")
            raise MilvusError("Milvus 连接失败")

        collections_name = self.config.chunks_collection
        if not milvus_client.has_collection(collections_name):
            self._2_create_chunks_collection(collections_name, milvus_client, vector_dim)

        file_title = chunks[0].get("file_title")
        self._3_clear_chunks_by_file_title(milvus_client, file_title)

        chunks_with_ids = self._4_insert_data(milvus_client, chunks)
        state["chunks"] = chunks_with_ids

        return state


    def _1_check_input(self, state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
        chunks = state.get("chunks")

        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        # 校验2：切片包含dense_vector字段
        first_chunk = chunks[0]
        if 'dense_vector' not in first_chunk:
            raise StateFieldError(field_name="chunks", message="错误: 数据中缺失dense_vector字段")

        # 校验3：切片包含 sparse_vector 字段
        if 'sparse_vector' not in first_chunk:
            raise StateFieldError(field_name="chunks", message="错误: 数据中缺失sparse_vector字段")

        # 提取向量维度
        vector_dim = len(first_chunk['dense_vector'])
        return chunks, vector_dim



    def _2_create_chunks_collection(self, collections_name, milvus_client, vector_dim):
        # 1. 创建schem
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
        # 2. 创建列
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)  # 切片内容
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=100)  # 切片标题
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=100)  # 父标题
        schema.add_field(field_name="part", datatype=DataType.INT8)  # 分片编号
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=100)  # 源文件标题
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=100)  # 商品名称（幂等性依据）
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)  # 稠密向量

        # 3. 创建索引
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
        )
        milvus_client.create_collection(
            collection_name=collections_name,
            schema=schema,
            index_params=index_params
        )

    def _3_clear_chunks_by_file_title(self, milvus_client, file_title):
        try:
            file_title = escape_milvus_string(file_title)
            milvus_client.delete(
                collection_name=self.config.chunks_collection,
                filter=f"file_title=='{file_title}'")
        except Exception as e:
            self.logger.error(f"Milvus 数据删除失败: {str(e)}")
            raise MilvusError(f"Milvus 数据删除失败: {str(e)}")

    def _4_insert_data(self, client, chunks):
        for chunk in chunks:
            if "part" not in chunk:
                chunk["part"] = 0

        result = client.insert(
            collection_name=self.config.chunks_collection,
            data=chunks
        )
        
        inserted_ids = result.get("ids")
        for idx, chunk in enumerate(chunks):
            chunk["chunk_id"] = inserted_ids[idx]

        return chunks

if __name__ == "__main__":

    setup_logging()

    json_path = r"/Users/lyinlu/PycharmProjects/KnowledgeBase/out/华为擎云 M272Q 用户指南-(XSN-27QBZ,02,zh-cn)/chunks_with_vecs.json"
    with open(json_path, "r", encoding="utf-8") as f:
        chunks_json = f.read()

    chunks = json.loads(chunks_json)
    # 从 JSON 加载后，转换稀疏向量的键为 int
    for chunk in chunks:
        if "sparse_vector" in chunk:
            chunk["sparse_vector"] = {
                int(k): v for k, v in chunk["sparse_vector"].items()
            }

    init_state = {
        "chunks": chunks
    }

    milvus_store = MilvusStore()
    result = milvus_store(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))