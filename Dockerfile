# 使用官方 Python 精简镜像作为基础
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 拷贝所有项目文件
COPY . /app

# 安装依赖（优先离线兜底联网）
# 解释：先尝试本地安装，若缺失则自动联网下载
RUN pip install torch==2.1.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip install torch_geometric -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# 安装其余依赖：官方为主，多个国内源作为备用
RUN pip install --no-cache-dir \
    --retries 5 --timeout 120 \
    --index-url https://pypi.org/simple \
    --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://pypi.mirrors.ustc.edu.cn/simple/ \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple/ \
    -r requirements.txt
# 创建输出目录（用于保存抽取结果）
RUN mkdir -p outputs/entities outputs/relations

# 暴露 FastAPI 服务端口
EXPOSE 8000

# 启动服务（通过 uvicorn 运行 FastAPI 应用）
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
