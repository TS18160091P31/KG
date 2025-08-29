你当前的项目由以下三个核心文件组成：

- `app.py`：FastAPI 服务接口定义；
- `config.py`：配置项管理（模型地址、输出路径等）；
- `kg_pipeline.py`：知识图谱抽取主流程。

为了将其封装为一个完整的 **Docker 服务**，暴露三个功能（模型配置、文件上传、结果返回），你可以按照以下步骤操作：

------

### ✅ 一、目录结构建议

```
kg-service/
├── app.py
├── config.py
├── kg_pipeline.py
├── requirements.txt
├── Dockerfile
└── steps/
    ├── _1_get_entities.py
    ├── _1b_entity_typing.py
    ├── _1c_entity_attributes.py
    └── _2_get_relations.py
```

------

### ✅ 二、添加 `requirements.txt`

```txt
fastapi
uvicorn
pydantic
pydantic-settings
dspy
ollama
```

如果你用的是本地 DeepSeek 或 Qwen 模型，通过 `ollama` 调用，那么还需要确保 `ollama` 服务已在宿主机上运行并开放端口。

------

### ✅ 三、编写 `Dockerfile`

```dockerfile
# 基础镜像（确保支持 Python 和 Ollama 请求）
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 拷贝文件
COPY . /app

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建必要输出目录
RUN mkdir -p outputs/entities outputs/relations

# 开放端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

------

### ✅ 四、运行 Docker 服务

在你的项目根目录下，执行：

```bash
# 构建镜像
docker build -t kg-service-offline .

# 运行容器（假设 Ollama 运行在宿主机）
docker run -d -p 8000:8000 \
  -e KG_OLLAMA_BASE=http://host.docker.internal:11434 \
  -e KG_OLLAMA_MODEL=qwen3 \
  --name kg-service-container \
  kg-service
  
 #一行版
 docker run -d --name kg-test -p 8000:8000 -e KG_OLLAMA_BASE=http://host.docker.internal:11434 -e KG_OLLAMA_MODEL=ollama_chat/deepseek-r1:14b -e KG_EMBED_BASE=http://host.docker.internal:11434 kg-service-offline 
```

如果在 Linux 上部署，则需使用宿主机 IP（例如 `172.17.0.1` 或 `localhost`）代替 `host.docker.internal`。

 停止容器：`docker stop kg-test`

 删除容器：`docker rm kg-test`

 或者使用一条命令强制删除：`docker rm -f kg-test`

------

### ✅ 五、接口说明

- **上传文本抽取**
  - `POST /extract`：传入 `text: str`
- **上传文件抽取**
  - `POST /extract_file`：传入 `file: UploadFile`（UTF-8文本）
- **返回格式示例**

```json
{
  "entity_attr_file": "outputs/entities/xxxx.json",
  "relations_file": "outputs/relations/xxxx.json",
  "entities_extracted": 12,
  "quads_extracted": 9,
  "elapsed_sec": 3.57
}
```

------

如需添加“模型配置查看”功能，还可在 `app.py` 增加接口：

```python
@app.get("/config")
def get_config():
    return {
        "ollama_base": settings.OLLAMA_BASE,
        "ollama_model": settings.OLLAMA_MODEL,
        "output_dir": str(settings.OUTPUT_DIR)
    }
```

http://localhost:8000/docs

