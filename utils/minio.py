import json
import logging
import os

from dotenv import load_dotenv
from minio import Minio
from processor.import_processor.base import setup_logging

load_dotenv()
setup_logging()

try:
    # 1.创建客户端链接对象
    minio_client = Minio(
        endpoint=os.getenv("MINIO_ENDPOINT", ""),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        #是否强制启用HTTPS的加密链接方式，False：使用http；True：使用https
        secure=False
    )

    # 2.判断bucket是否存在，不存在则创建
    bucket_name = os.getenv("MINIO_BUCKET_NAME", "knowledge-base")
    found = minio_client.bucket_exists(bucket_name)
    if not found:
        minio_client.make_bucket(bucket_name)

    # 3. 定义当前bucket的访问权限
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{bucket_name}/*",
            },
        ],
    }
    # 4. 设置当前bucket的访问权限
    minio_client.set_bucket_policy(bucket_name, json.dumps(policy))

except Exception as e:
    logging.error(f"MinIO连接失败，错误原因：{e}")

def get_minio_client():
    return minio_client

if __name__ == '__main__':
    get_minio_client()