import json
import logging

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from utils.embedding import generate_embeddings


class BGEEmbed(BaseNode):
    name = "bge_embed"
    def process(self, state) -> dict:
        """

        :param state: 文档路径
        :return: 是否为PDF、是否为MD、PDF/MD文档路径、文档标题
        """

        chunks = state.get("chunks")

        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        texts = [f"{chunk['item_name']} \n{chunk['content']}" for chunk in chunks]
        dense_vecs, sparse_vecs = generate_embeddings(texts)

        output_data = []
        for idx, chunk in enumerate(chunks):
            chunk["dense_vector"] = dense_vecs[idx]
            chunk["sparse_vector"] = sparse_vecs[idx]
            output_data.append(chunk)

        state['chunks'] = output_data
        return state


if __name__ == "__main__":

    setup_logging()

    json_path = r"/Users/lyinlu/PycharmProjects/KnowledgeBase/out/华为擎云 M272Q 用户指南-(XSN-27QBZ,02,zh-cn)/chunks_with_item_name.json"
    with open(json_path, "r", encoding="utf-8") as f:
        chunks_json = f.read()

    chunks = json.loads(chunks_json)

    init_state = {
        "chunks": chunks
    }

    bge_embedding = BGEEmbed()
    result = bge_embedding(init_state)
    json_res = json.dumps(result, ensure_ascii=False, indent=4)
    print(f"result: {result}")
    # print(f"json_res: {json_res}")  # 得到的是json字符串，会把sparse_vector的键（token_id）int类型转换成字符串