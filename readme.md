# KG‑Service — 实体属性 & 关系抽取 API

> **一句话**：输入任何中文新闻/报告，后台自动调用本地 Ollama+DeepSeek 模型，生成
>
> 1. 实体+属性 2.事件四元组；结果落地到 `outputs/`.

------

### 环境要求

- **Python**：`==3.10`（推荐使用虚拟环境）
- **Ollama**：已安装并运行（本地或局域网）
- **模型**：如 `deepseek-coder`, `llama3` 等已通过 `ollama pull` 下载

------

### 目录结构

```
kg-service/
├── app.py              # FastAPI 入口
├── kg_pipeline.py      # 抽取流水线（实体→属性→关系）
├── config.py           # Ollama URL / 模型 / 输出路径
├── requirements.txt    # 精简依赖
├── wheelhouse/         # 离线安装用 *.whl（可选）
└── outputs/            # 运行时生成
    ├── entities/
    └── relations/
```

------

## 1. 快速启动（联网开发机）

```bash
python3.10 -m venv venv
或者
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1       
pip install -r requirements.txt    # 如离线安装见 §3
uvicorn app:app --reload --port 8000
```

> - 默认 Ollama 地址：`http://127.0.0.1:11434`
> - 默认模型名：`deepseek`
>    可通过环境变量临时覆盖：
>
> ```bash
> set KG_OLLAMA_BASE=http://192.168.1.8:11434
> set KG_OLLAMA_MODEL=llama3
> ```

------

## 2. 接口文档

### 2.1 POST `/extract`

| 参数   | 类型   | 说明                       |
| ------ | ------ | -------------------------- |
| `text` | string | 原始中文文本（UTF‑8 编码） |

**返回示例**：

```json
{
  "entity_attr_file": "outputs/entities/9f4c7d2a.json",
  "relations_file":   "outputs/relations/9f4c7d2a.json",
  "entities_extracted": 16,
  "quads_extracted": 22,
  "elapsed_sec": 4.13
}
```

------

### 2.2 POST `/extract_file`

支持上传任意格式的 `.txt` 文本文件，自动识别编码转为 UTF‑8。

```bash
curl.exe -X POST http://127.0.0.1:8000/extract_file ^
  -F "file=@sample.txt"
curl.exe -X POST http://127.0.0.1:8000/extract_file -F "file=@sample.txt"
```

------

## 3.  内网部署（离线安装）

### 3.1 在联网机器打包 `.whl`

```bash
pip install pip-tools
pip download --dest wheelhouse -r requirements.txt  
#或者
pip download --dest wheelhouse -r requirements.txt --only-binary=:all:
pip install --no-index --find-links=wheelhouse -r requirements.txt
python -m build --wheel     # 如果有 Git 包（如 dspy）需手动构建并加入 wheelhouse/
```

压缩并拷贝到服务器：

```bash
tar czf wheelhouse.tgz wheelhouse
scp wheelhouse.tgz user@server:/opt/kg-service
```

------

### 3.2 内网服务器安装

```bash
tar xzf wheelhouse.tgz
python3.10 -m venv venv && source venv/bin/activate
pip install --no-index --find-links=wheelhouse -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

> 保证 `ollama` 和 `deepseek` 模型已本地部署。

------

## 4. 常见问题

| 问题                       | 原因 / 解决                                    |
| -------------------------- | ---------------------------------------------- |
| `No matching distribution` | `.whl` 不兼容平台。确认 Python 版本 / 架构一致 |
| `文件必须为 UTF‑8`         | 后端已自动转码，若仍出错请检查是否是二进制文件 |
| `GPU 显存不足`             | 可切换至 CPU 版模型，或减少 batch / max_tokens |

------

## 5. 二次开发指南

- **更换 Prompt 逻辑** → 修改 `steps/*.py` 内模板字符串
- **接入数据库 / Neo4j** → 在 `kg_pipeline.py` 内增加存储逻辑
- **开放 WebSocket / 多模型接口** → 用 FastAPI `@websocket` / `@app.get("/model")` 扩展即可