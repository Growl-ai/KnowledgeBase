import base64
import json
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Tuple, List, Dict

from langchain.chat_models import init_chat_model
from minio.deleteobjects import DeleteObject
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState
from utils.minio import get_minio_client


class MDImg(BaseNode):

    name = "md_img"
    def process(self, state) -> dict:
        """
        1.数据准备与校验
        2.扫描筛选有效引用的图片
        3.生成图片摘要，API调用速率限制
        4.上传与替换
        5.备份与保存
        :param state: md_path, md_content
        :return:
        """
        md_path_obj, md_content, img_dir = self._1_prepare_input(state)
        if not img_dir.exists():
            self.logger.info("无图片文件夹，跳过图片处理")
            return state

        tgt_imgs = self._2_filter_imgs(md_content, img_dir) 
        if not tgt_imgs:
            self.logger.info("未检测到MD中引用了图片，跳过图片处理")
            return state

        summaries = self._3_generate_summaries(md_path_obj.stem, tgt_imgs)

        new_md_content = self._4_upload_and_replace(md_path_obj.stem, tgt_imgs, summaries, md_content)

        new_md_file = os.path.splitext(state["md_path"])[0] + "_new.md"
        with open(new_md_file, "w", encoding="utf-8") as f:
            f.write(new_md_content)
        self.logger.info(f"处理后MD文件已保存，新文件路径：{new_md_file}")

        state["md_content"] = new_md_content
        state["md_path"] = new_md_file

        return state

    def _1_prepare_input(self, state: ImportGraphState) -> Tuple[Path, str, Path]:
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(field_name="md_path", expected_type=str)
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError(message=f"文件不存在: {md_path_obj.name}")

        md_content = state.get("md_content")
        img_dir = md_path_obj.parent / "images"
        return md_path_obj, md_content, img_dir

    def _2_filter_imgs(self, md_content, img_dir) -> List[Tuple[str, str, Tuple[str, str]]]:
        """
        扫描图片文件夹，过滤【支持格式+md实际引用】，提取上下文，组装元数据
        :param md_content:
        :param img_dir:
        :return: [(img_path, img_name, (前文, 后文)]
        """
        tgt_imgs = []
        for img in os.listdir(img_dir):
            # 检查格式支持
            suffix = Path(img).suffix
            if suffix not in self.config.image_extensions:
                self.logger.warning(f"图片格式不支持: {img}, 跳过")
                continue
            img_path = img_dir / img
            # 查找图片在md中的引用
            pattern = re.compile(r"!\[.*?\]\(.*?"+re.escape(img)+r".*?\)")
            match = pattern.search(md_content)
            if not match:
                self.logger.warning(f"图片未在md中引用: {img}, 跳过")
                continue
            # 提取图片引用上下文
            start, end = match.span()
            ctx_len = self.config.img_content_length
            pre_ctx = md_content[max(0, start-ctx_len): start]
            post_ctx = md_content[end: min(end+ctx_len, len(md_content))]

            tgt_imgs.append((img_path, img, (pre_ctx, post_ctx)))
        return tgt_imgs


    def _3_generate_summaries(self, md_stem: str, tgt_imgs: List[Tuple[str, str, Tuple[str, str]]]) -> Dict[str, str]:
        """
        调用VL模型API，生成摘要，API调用速率限制
        :param tgt_imgs: [(img_path, img_name, (前文, 后文))]
        :return: {img_name: img_summary}
        """
        summaries = {}
        req_times = deque()  # 请求时间戳双端队列
        for img_path, img_name, img_ctx in tgt_imgs:
            self._apply_api_rate_limit(req_times, max_reqs=60)
            summaries[img_name] = self._summarize_img(img_path, img_ctx, md_stem)
        return summaries
        

    def _apply_api_rate_limit(self, req_times: deque, max_reqs: int, window_sec: int=60) -> None:
        """窗口内请求数超上限则自动等待，防止触发第三方API限流"""
        current_time = time.time()
        # 移除超时/过期的请求时间戳
        while req_times and current_time - req_times[0] >= window_sec:
            req_times.popleft()
        # 检查当前窗口内请求数是否超过上限
        if len(req_times) >= max_reqs:
            # 等待直到下一个请求时间戳过期
            wait_time = window_sec - (current_time - req_times[0])
            if wait_time > 0:
                self.logger.warning(f"API调用速率超限，等待 {wait_time} 秒")
                time.sleep(wait_time)
                # 等待完成后，清理过期请求
                current_time = time.time()
                while req_times and current_time - req_times[0] >= window_sec:
                    req_times.popleft()
        req_times.append(current_time)
        self.logger.info(f"当前窗口请求数为{len(req_times)}, 窗口时间：{window_sec}秒")


    def _summarize_img(self, img_path: str, img_ctx: Tuple[str, str], md_stem: str) -> str:
        with open(img_path, "rb") as img_file:
            img_data = img_file.read()
        if not img_data:
            self.logger.warning(f"图片文件为空，跳过：{img_path}")
            return "图片描述"
        base64_image = base64.b64encode(img_data).decode("utf-8")
        try:
            vlm = init_chat_model(
                model=self.config.vl_model,
                model_provider="openai",
                api_key=self.config.dashscope_api_key,
                base_url=self.config.dashscope_base_url
            )
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""这是"{md_stem}"文件中的一张图片，图片上文部分为"{img_ctx[0]}"，下文部分为"{img_ctx[1]}"，请用中文简要总结这张图片的内容，用于 Markdown 图片标题。"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            response = vlm.invoke(messages)
            return response.content.strip().replace("\n", "")
        except Exception as e:
            self.logger.error(f"图像总结失败：{img_path}, 错误{e}")
            return "图片描述"
    def _4_upload_and_replace(self, md_stem: str, target_images: List[Tuple[str, str, Tuple[str, str]]],
                                   summaries: Dict[str, str], md_content: str) -> str:
        minio_client = get_minio_client()
        minio_img_dir = self.config.minio_img_dir
        upload_dir = f"{minio_img_dir}/{md_stem}".replace(" ", "")

        try:
            objects_to_delete = minio_client.list_objects(self.config.minio_bucket, upload_dir, recursive=True)
            # 构造删除列表
            delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]
            if delete_list:
                errors = minio_client.remove_objects(self.config.minio_bucket, delete_list)
                for error in errors:
                    self.logger.error(f"删除失败：{error}")
        except Exception as e:
            self.logger.error(f"清理minio目录失败：{e}")

        urls = {}
        base_url = f"http://{self.config.minio_endpoint}/{self.config.minio_bucket}"
        for img_path, img_file, _ in target_images:
            object_name = f"{upload_dir}/{img_file}"
            try:
                minio_client.fput_object(
                    bucket_name=self.config.minio_bucket,  # MinIO存储桶名（从配置读取）
                    object_name=object_name,  # MinIO对象名称
                    file_path=img_path,  # 本地文件路径
                    content_type=f"image/{os.path.splitext(img_path)[1][1:]}"
                )
            except Exception as e:
                self.logger.error(f"图片上传MinIO失败：{img_path}，错误信息：{str(e)}")
            urls[img_file] = f"{base_url}/{object_name}"

        for image_file, summary in summaries.items():
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
            if url := urls.get(image_file):
                md_content = pattern.sub(lambda m: f"![{summary}]({url})", md_content)
        self.logger.info(f"MD文件图片引用替换完成，共替换{len(urls)}处图片引用")

        return md_content



if __name__ == "__main__":
    setup_logging()
    md_path = r"/Users/lyinlu/PycharmProjects/KnowledgeBase/out/华为擎云 M272Q 用户指南-(XSN-27QBZ,02,zh-cn)/华为擎云 M272Q 用户指南-(XSN-27QBZ,02,zh-cn).md"
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    init_state= {"md_path": md_path, "md_content": md_content}
    md_img = MDImg()
    result = md_img.process(init_state)
    json_res = json.dumps(result, ensure_ascii=False, indent=4)
    print(json_res)