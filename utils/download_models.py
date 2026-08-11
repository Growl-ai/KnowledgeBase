from modelscope import snapshot_download

model_path = snapshot_download(
    model_id="BAAI/bge-m3",
    local_dir="/Users/lyinlu/ai_models",
    # 忽略指定文件夹/文件，通配匹配
    ignore_patterns=[
        "1_Pooling/*",
        "imgs/*",
        "onnx/*",
        ".gitattributes"
    ]
)
print(f"模型下载完成，路径：{model_path}")