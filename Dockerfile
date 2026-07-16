FROM python:3.13-slim

WORKDIR /app

# 只装 server 需要的依赖（不含 mlx-lm——Apple Silicon only，Docker 跑不了）
# embedding 走 HTTP 连宿主机的 embed_server
RUN pip install --no-cache-dir \
    anthropic==0.111.0 \
    python-dotenv==1.2.2 \
    fastapi==0.139.0 \
    uvicorn==0.50.2 \
    pydantic==2.13.4 \
    openai==2.43.0 \
    requests==2.34.2 \
    tiktoken==0.13.0

# 源码
COPY src/ src/

# 数据目录（挂载 volume 持久化）
RUN mkdir -p data

# embedding 走宿主机（Docker 容器内不能跑 MLX）
ENV EMBED_ENDPOINT=http://host.docker.internal:8765

EXPOSE 8000
CMD ["python", "-m", "src.server"]
