<div align="center">

# 产品资料智能知识库问答系统

基于 RAG（检索增强生成）技术的企业级智能知识库系统，面向客服、运营、业务人员提供精准的产品手册、说明书、维修资料检索与问答服务。

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.140+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-FF6F00)](https://langchain-ai.github.io/langgraph)
[![Milvus](https://img.shields.io/badge/Milvus-3.0+-40AE40)](https://milvus.io)
[![LangChain](https://img.shields.io/badge/LangChain-1.3+-1786FC)](https://python.langchain.com)

</div>

---

## 1. 项目简介

本项目聚焦 **产品资料垂直领域智能问答**，解决传统场景下「资料分散、人工翻找成本高、产品手册与维修文档查询效率低」等痛点。系统将 **PDF / Markdown / 图片 / 表格** 等多格式非结构化文档统一导入知识库后，用户只需通过自然语言提问，即可从知识库中召回相关文档片段，并生成有据可依的精准回答。

### 1.1 核心目标

- 📚 非结构化文档统一入库，支持 PDF 复杂排版（公式、表格、图片）的结构化转换
- 🔍 多路召回策略：向量检索 + HyDE + MCP 联网搜索 + RRF 融合 + Rerank 精排
- 💬 流畅的 SSE 流式问答交互，逐字输出打字机效果
- 🧠 多轮对话记忆，上下文连续对话能力

---

## 2. 核心功能

| 功能模块 | 说明 |
| --- | --- |
| **文档智能导入** | PDF/Markdown 文件上传 → MinerU 解析 → 智能切片 → 向量化入库 |
| **商品名识别** | 基于 LLM 提取 + 向量对齐，动态识别用户问题中的商品实体 |
| **混合向量检索** | BGE-M3 稠密向量 + BM25 稀疏向量，Milvus 混合检索 |
| **多路召回融合** | 向量检索 / HyDE 假设性文档检索 / MCP 联网搜索，三路并行召回 |
| **RRF + Rerank** | 倒数秩融合 (RRF) 多路结果 + BGE-Reranker 交叉编码器精排 + 断崖截断 |
| **流式问答** | SSE 实时推送，逐字输出，阶段进度实时展示 |
| **多轮记忆** | MongoDB 会话记忆，自动补全历史上下文，按 session_id 多用户隔离 |

---

## 3. 系统架构

### 3.1 整体架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                        前端 (HTML5 + JS)                              │
│   ┌──────────────┐          ┌───────────────────┐                    │
│   │  聊天界面     │◄───────►│  导入界面          │                    │
│   └──────┬───────┘          └────────┬──────────┘                    │
└──────────┼───────────────────────────┼───────────────────────────────┘
           │ SSE /stream    HTTP /upload /query  /status
┌──────────▼───────────────────────────▼───────────────────────────────┐
│                     后端 (FastAPI + LangGraph)                        │
│  ┌───────────────────────┐  ┌───────────────────────┐                │
│  │  查询服务 (port:8001) │  │  导入服务 (port:8000) │                │
│  │  KBQueryWorkflow      │  │  KBImportWorkflow     │                │
│  └───────────┬───────────┘  └──────────┬────────────┘                │
│              │  LangGraph StateGraph   │                             │
│    ┌─────────▼─────────────────────────▼───────────┐                 │
│    │            工作流节点 (Nodes)                   │                 │
│    │  导入: PDF→MD→图片→切片→商品名→向量→入库      │                 │
│    │  查询: 商品确认→三路召回→RRF→Rerank→生成答案   │                 │
│    └───────────────────────┬────────────────────────┘                 │
└────────────────────────────┼──────────────────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────────────┐
│                       数据层 / 外部服务                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐      │
│  │  Milvus  │  │ MongoDB  │  │  MinIO   │  │ DashScope (LLM)  │      │
│  │ 向量数据库│  │ 会话历史  │  │ 对象存储  │  │ VLM / WebSearch │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘      │
└───────────────────────────────────────────────────────────────────────┘
```

### 3.2 RAG 核心链路

**存储链路（导入）**

```
原始文档 (PDF/MD)
    │
    ▼
MinerU 解析 → Markdown + 图片/表格提取
    │
    ▼
图片处理 → VLM 摘要生成 + MinIO 上传 + 60RPM 滑动窗口限流
    │
    ▼
语义切片 → 商品名实体识别
    │
    ▼
BGE-M3 混合向量化 (稠密 + 稀疏)
    │
    ▼
Milvus 向量数据库 + MinIO 对象存储
```

**查询链路（问答）**

```
用户自然语言问题
    │
    ▼
商品名确认 + 问题改写 (结合多轮历史上下文)
    │
    ▼
三路并行召回 ──► 向量检索 (Milvus)
              ├──► HyDE 假设性文档检索
              └──► MCP 联网搜索 (DashScope WebSearch)
    │
    ▼
RRF 倒数秩融合 (Reciprocal Rank Fusion)
    │
    ▼
BGE-Reranker 交叉编码器精排 + 断崖截断
    │
    ▼
LLM (Qwen) 生成答案 + SSE 流式推送 (progress → delta → final)
    │
    ▼
MongoDB 持久化会话历史 (session_id 维度隔离)
```

---

## 4. 技术栈

| 类别 | 技术选型 | 说明 |
| --- | --- | --- |
| 后端框架 | **FastAPI + Uvicorn** | 异步高性能 HTTP 服务 |
| 工作流编排 | **LangGraph** | 有状态图编排 (StateGraph)，DAG 流程 |
| 大语言模型 | 阿里云 DashScope (Qwen 系列) | qwen-flash / qwen3-vl-flash |
| 向量嵌入 | BGE-M3 + FlagEmbedding | 稠密 + 稀疏混合嵌入模型 |
| 重排序 | BGE-Reranker-Large | 交叉编码器，本地部署 |
| 向量数据库 | Milvus 3.0 | 稠密 + 稀疏混合检索 |
| 文档数据库 | MongoDB | 对话历史多轮记忆存储 |
| 对象存储 | MinIO | 原始文件 + 图片持久化 |
| PDF 解析 | MinerU | 支持公式 / 表格 / 复杂排版提取 |
| 流式协议 | SSE (Server-Sent Events) | 流式问答输出 |
| 联网搜索 | MCP (Model Context Protocol) | DashScope WebSearch |
| 包管理 | uv | Python 项目依赖管理 |

---

## 5. 目录结构

```
KnowledgeBase/
├── processor/                      # 核心业务逻辑
│   ├── import_processor/         # 导入流程工作流
│   │   ├── nodes/                # LangGraph 节点实现
│   │   │   ├── entry.py          # 入口路由
│   │   │   ├── pdf_to_md.py      # MinerU PDF → Markdown
│   │   │   ├── md_img.py         # 图片处理 + VLM 摘要 (滑动窗口限流保护)
│   │   │   ├── doc_split.py      # 语义切片
│   │   │   ├── item_name_extract.py  # 商品名识别
│   │   │   ├── bge_embed.py      # BGE-M3 向量化
│   │   │   └── milvus_store.py   # Milvus 入库
│   │   ├── main_graph.py         # 导入工作流图
│   │   ├── state.py              # 状态定义
│   │   └── base.py               # 节点基类
│   └── query_processor/          # 查询流程工作流
│       ├── nodes/                # 查询节点实现
│       │   ├── item_name_confirm.py    # 商品名确认 + 历史上下文 + 问题改写
│       │   ├── retrieve_vecs.py        # Milvus 向量检索
│       │   ├── retrieve_hyde.py        # HyDE 假设性检索
│       │   ├── web_search_mcp.py       # MCP 联网搜索
│       │   ├── rrf.py                  # RRF 倒数秩融合
│       │   ├── rerank.py               # Rerank 精排 + 断崖截断
│       │   └── answer_output.py        # LLM 生成 + SSE 推送
│       ├── main_graph.py        # 查询工作流图
│       └── state.py             # 状态定义
├── utils/                          # 工具函数
│   ├── llm.py                   # LLM 调用封装
│   ├── embedding.py             # Embedding 封装
│   ├── rerank.py                # Rerank 封装
│   ├── milvus.py / mongo.py / minio.py   # 中间件连接工具
│   ├── sse.py                   # SSE 会话队列管理 + 异步生成器
│   ├── task_trace.py            # 节点进度追踪 + SSE progress 推送
│   ├── prompt.py                # Prompt 模板集中管理
│   └── logger.py                # 日志
├── web/                           # 接口 & 前端页面
│   ├── api/
│   │   ├── import_service.py    # 导入 API (端口 8000，含 /upload /status)
│   │   └── query_service.py     # 查询 + SSE API (端口 8001，含 /query /stream /history)
│   └── page/
│       ├── chat.html            # 聊天前端页面 (EventSource SSE 接收)
│       └── import.html          # 导入前端页面
├── docs/                          # 项目文档
├── main.py
├── pyproject.toml                 # uv 项目配置 & 依赖声明
├── uv.lock
├── .env.example                   # 环境变量示例
└── .gitignore
```

---

## 6. 快速开始

### 6.1 环境要求

- Python **3.11+**
- [uv](https://github.com/astral-sh/uv) 包管理器
- 中间件服务：
  - Milvus 3.0+
  - MongoDB 6.0+
  - MinIO 最新版
- 阿里云 DashScope API Key

### 6.2 安装依赖

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd KnowledgeBase

# 2. 用 uv 同步依赖（推荐，自动解析 uv.lock）
uv sync

# 或手动使用 pip
# pip install -r <(uv pip compile pyproject.toml)
```

### 6.3 启动中间件（推荐 Docker Compose 一键启动）

参考 docker-compose.yml 示例

### 6.4 配置环境变量

```bash
# 复制示例文件并填入真实配置
cp .env.example .env
```


### 6.5 下载 BGE 模型（首次使用）

```bash
# 可选：ModelScope 下载（国内网络友好，需提前 pip install modelscope）
# python utils/download_models.py
```

### 6.6 启动服务

```bash
# ===== 终端 1：启动导入服务（端口 8000）=====
uv run python -m web.api.import_service
# 或 uvicorn web.api.import_service:app --host 0.0.0.0 --port 8000 --reload

# ===== 终端 2：启动查询服务（端口 8001）=====
uv run python -m web.api.query_service
# 或 uvicorn web.api.query_service:app --host 0.0.0.0 --port 8001 --reload
```

启动后访问：

| 入口 | 地址 |
| --- | --- |
| 文档导入页面 | http://127.0.0.1:8000/import.html |
| 聊天问答页面 | http://127.0.0.1:8001/chat.html |
| 导入 API 文档 (Swagger) | http://127.0.0.1:8000/docs |
| 查询 API 文档 (Swagger) | http://127.0.0.1:8001/docs |
| 导入服务健康检查 | http://127.0.0.1:8000/health |
| 查询服务健康检查 | http://127.0.0.1:8001/health |

---

## 7. 使用流程

### 7.1 文档导入

1. 打开 **导入页面** http://127.0.0.1:8000/import.html
2. 上传 PDF / Markdown 文件（支持多文件批量上传）
3. 系统为每个文件生成独立 `task_id`，后台异步执行：
   - MinerU 解析 PDF → Markdown
   - 图片处理（VLM 摘要 + 限流保护）+ MinIO 上传
   - 语义切片 + 商品名识别
   - BGE-M3 向量化 → Milvus 入库
4. 前端自动轮询 `/status/{task_id}` 展示处理进度

### 7.2 智能问答

1. 打开 **聊天页面** http://127.0.0.1:8001/chat.html
2. 输入自然语言问题
3. 前端自动维护 `session_id`（保存在 localStorage），实现多轮对话记忆连续
4. 阶段进度实时展示：✅ 已完成节点 + ⏳ 进行中节点
5. LLM 答案逐字流式输出，答案附带来源图片与引用信息

---

## 8. 设计亮点

1. **LangGraph 双工作流解耦**：导入与查询流程各自独立 StateGraph，节点模块化封装，可单独调试扩展
2. **SSE + 内存队列 生产者-消费者模式**：后台 LangGraph 执行流与 SSE 响应通道通过 `session_id` 绑定的内存队列通信，跨线程解耦
3. **滑动窗口 API 限流保护**：VLM 图片摘要阶段实现 60RPM 滑动窗口计数器，自动 sleep 压平速率不触发第三方限流
4. **三路召回 + RRF 融合 + Rerank 精排 + 断崖截断**：在召回率与精准率之间取得平衡，控制 LLM 上下文噪声
5. **消息先落库后回填模式**：用户消息先插入 MongoDB 获取 `_id`，商品名识别完成后再回填 `item_names` 与 `rewritten_query`，保证用户提问历史 100% 不丢失
6. **多用户三维隔离**：`session_id` 贯穿 SSE 内存队列、MongoDB 查询过滤、历史记录读写三个维度，多用户数据天然隔离

---

## 9. License

本项目仅用于学习与研究用途。接入使用请遵循第三方 API 与模型许可协议。
