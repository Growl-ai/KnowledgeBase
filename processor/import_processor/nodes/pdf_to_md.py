import shutil
import time
import zipfile
from pathlib import Path

import requests

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, PdfConversionError
from processor.import_processor.state import ImportGraphState

# PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class PDFToMD(BaseNode):
    name = "pdf_to_md"
    def process(self, state: ImportGraphState) -> dict:
        """
        1.准备文件、输出目录
        2.1.申请上传：调用minerU获取上传链接
        2.2.上传文件, 轮询结果：得到zip_url
        3.获取结果：下载zip文件，解压并读取 md内容
        :param state: PDF路径、解析结果输出目录
        :return: md文档内容、md文档路径
        """
        pdf_path_obj, output_dir_obj = self._prepare_paths(state)
        zip_url = self._upload_and_poll(pdf_path_obj)
        md_path = self._get_result(zip_url, output_dir_obj, pdf_path_obj.stem)

        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        state["md_content"] = md_content
        state["md_path"] = md_path

        return state

    def _prepare_paths(self, state: ImportGraphState):
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(field_name="pdf_path", node_name=self.name, expected_type=str)

        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(field_name="file_dir", node_name=self.name, expected_type=str)

        pdf_path_obj = Path(pdf_path)
        file_dir_obj = Path(file_dir)

        if not pdf_path_obj.exists():
            raise FileProcessingError(message=f"PDF文件不存在：{pdf_path}", node_name=self.name)
        if not file_dir_obj.exists():
            self.logger.warning(f"输出目录不存在, 将自动创建{file_dir}")
            file_dir_obj.mkdir(parents=True)

        return pdf_path_obj, file_dir_obj

    def _upload_and_poll(self, pdf_path_obj: Path):
        """申请上传连接，上传文件，轮询结果获取zip_url"""
        token = self.config.mineru_token
        url = f"{self.config.mineru_base_url}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }
        # 申请上传链接
        response = requests.post(url, headers=header, json=data)
        if response.status_code != 200:
            raise PdfConversionError(message=f"申请上传失败,状态码：{response.status_code}, {response.text}")
        result = response.json()
        if result["code"] != 0:
            raise PdfConversionError(message=f"申请上传失败, {result['msg']}")
        batch_id = result["data"]["batch_id"]
        file_urls = result["data"]["file_urls"]

        # 上传文件
        file_path = [pdf_path_obj]
        for i in range(0, len(file_urls)):
            with open(file_path[i], 'rb') as f:
                res_upload = requests.put(file_urls[i], data=f)
                if res_upload.status_code != 200:
                    raise PdfConversionError(message=f"上传文件失败,状态码：{res_upload.status_code}, {res_upload.text}")
                self.logger.info(f"文件 {file_path[i]} 上传成功")
        # 轮询结果
        poll_url = f"{self.config.mineru_base_url}/extract-results/batch/{batch_id}"
        start_time = time.time()
        timeout_sec = 600
        poll_interval = 3
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_sec:
                raise TimeoutError(f"任务轮询超时，已消耗{elapsed_time}秒")
            try:
                response = requests.get(poll_url, headers=header, timeout=10)
            except Exception as e:
                self.logger.warning(f"轮询网络请求异常：{e}，{poll_interval}秒后重试")
                time.sleep(poll_interval)
                continue
            if response.status_code != 200:
                raise PdfConversionError(message=f"轮询请求失败,状态码：{response.status_code}, {response.text}")
            # 解析结果
            result = response.json()
            if result["code"] != 0:
                raise PdfConversionError(message=f"业务错误, {result['msg']}")
            extract_results = result["data"]["extract_result"]
            for i in range(0, len(extract_results)):
                state = extract_results[i]["state"]
                if state == "done":
                    self.logger.info(f"文件{extract_results[i]['file_name']}解析完成, 总耗时{time.time()-start_time}秒")
                    full_zip_url = extract_results[i]["full_zip_url"]
                    return full_zip_url
                elif state == "failed":
                    raise PdfConversionError(f"文件{extract_results[i]['file_name']}解析失败, {extract_results[i]['err_msg']}")
                else:
                    self.logger.info(f"文件{extract_results[i]['file_name']}解析中, 已耗时{time.time()-start_time}秒")
                    time.sleep(poll_interval)


    def _get_result(self, zip_url: str, output_dir_obj: Path, pdf_stem: str):
        """
        下载zip文件，解压并读取 md内容
        """
        self.logger.info(f"开始下载zip文件")
        response = requests.get(zip_url)
        if response.status_code != 200:
            raise RuntimeError(f"下载zip文件失败,状态码：{response.status_code}, {response.text}")

        zip_save_path = output_dir_obj / f"{pdf_stem}.zip"
        with open(zip_save_path, "wb") as f:
            f.write(response.content)
        self.logger.info(f"zip文件下载完成")

        unzip_dir = output_dir_obj / pdf_stem
        if unzip_dir.exists():
            shutil.rmtree(unzip_dir)
        self.logger.info(f"已清空旧的解压目录: {unzip_dir}")
        unzip_dir.mkdir(parents=True)

        self.logger.info(f"开始解压zip文件....")
        with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
        self.logger.info(f"解压完成, 解压目录: {unzip_dir}")

        md_file = unzip_dir / "full.md"
        new_md_file = md_file.with_name(f"{pdf_stem}.md")
        md_file.rename(new_md_file)
        self.logger.info(f"重命名md文件: full.md -> {pdf_stem}.md")
        return str(new_md_file.absolute())


if __name__ == "__main__":
    setup_logging()
    init_state = {
        "pdf_path": r"/Users/lyinlu/PycharmProjects/KnowledgeBase/assets/华为擎云 M272Q 用户指南-(XSN-27QBZ,02,zh-cn).pdf",
        "file_dir": r"/Users/lyinlu/PycharmProjects/KnowledgeBase/out"
        }
    pdf_to_md = PDFToMD()
    result = pdf_to_md.process(init_state)
    print(result)