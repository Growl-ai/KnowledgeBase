import json
import logging
import os
from datetime import datetime, date
from decimal import Decimal
from typing import List, Dict, Any
from uuid import UUID

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

load_dotenv()

class MongoTool:
    def __init__(self):
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")
            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]  # 数据库
            self.chat_message = self.db["chat_message"]  # 集合
            # 创建复合索引，提升查询性能, session_id升序 + ts降序: 按会话查最新记录
            self.chat_message.create_index([("session_id", 1), ("ts", -1)])
            logging.info(f"Successfully connected to MongoDB: {self.db_name}")

        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            raise

_mongo_tool = MongoTool()

def get_mongo_tool() -> MongoTool:
    global _mongo_tool
    if _mongo_tool is None:
        _mongo_tool = MongoTool()
    return _mongo_tool

def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        image_urls: List[str] = None,
        message_id: str = None
) -> str:
    ts = datetime.now().timestamp()
    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query or "",
        "item_names": item_names,
        "image_urls": image_urls,
        "ts": ts,
    }
    mongo_tool = get_mongo_tool()
    if message_id:  # 更新
        result = mongo_tool.chat_message.update_one(
            {"_id": ObjectId(message_id)},  # 更新条件：主键匹配（需将字符串转为ObjectId类型）
            {"$set": document}  # 更新操作：$set表示只更新指定字段，保留其他字段
        )
        return message_id
    else:  # 新增
        result = mongo_tool.chat_message.insert_one(document)
        return str(result.inserted_id)

def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    mongo_tool = get_mongo_tool()
    try:
        object_ids = [ObjectId(i) for i in ids]
        result = mongo_tool.chat_message.update_many(
            {"_id": {"$in": object_ids}},
            {"$set": {"item_names": item_names}}
        )
        logging.info(f"Updated {result.modified_count} records to item_names: {item_names}")
        return result.modified_count
    except Exception as e:
        logging.error(f"Error updating history item_names: {e}")
        return 0

def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    mongo_tool = get_mongo_tool()
    try:
        query = {"session_id": session_id}
        cursor = mongo_tool.chat_message.find(query).sort("ts", ASCENDING).limit(limit)
        return list(cursor)
    except Exception as e:
        logging.error(f"Error getting recent messages: {e}")
        return []

def clear_history(session_id: str) -> int:
    mongo_tool = get_mongo_tool()
    try:
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        logging.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        return result.deleted_count
    except Exception as e:
        logging.error(f"Error clearing history for session {session_id}: {e}")
        return 0


class MongoEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)

def serialize_json(data, ensure_ascii=False, indent=None, **kwargs):
    return json.dumps(data, cls=MongoEncoder, ensure_ascii=ensure_ascii, indent=indent, **kwargs)

def to_json_file(data, filepath, ensure_ascii=False, indent=4, **kwargs):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=MongoEncoder, ensure_ascii=ensure_ascii, indent=indent, **kwargs)


if __name__ == "__main__":
    session_id = "session_001"
    save_chat_message(session_id, "user", "你好，有烫金机吗？")
    save_chat_message(session_id, "assistant", "你好！请问你想询问哪个型号？")
    save_chat_message(session_id, "user", "brother的HAK180烫金机")
    save_chat_message(session_id, "assistant", "有的")