# ============================================================
# Dockerfile — Mitta AI 智能助理后端服务
# 构建：docker build -t mitta-ai .
# 运行：docker run -p 8000:8000 --env-file .env mitta-ai
# ============================================================

# 使用官方 Python 3.13 slim 镜像（减小体积）
FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

# 安装系统依赖：
# - gcc/libpq-dev：psycopg 编译需要
# - curl：健康检查
# - nodejs/npm：MCP stdio 服务器需要 npx（filesystem/git/sequential-thinking 等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 缓存层
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8000

# 健康检查（使用 curl，slim 镜像已安装）
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
# PYTHONPATH=/app/src 已在 ENV 中设置，uvicorn main:app 可直接找到 src/ 下的模块
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
