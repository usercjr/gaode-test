# agent_service 镜像。构建上下文是项目根目录（需要 config.yaml）：
#   docker build -f agent_service/Dockerfile -t wanderai-agent .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf

WORKDIR /app

# 先装依赖再拷代码：代码改动不会让依赖层失效，重新构建只需几秒
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY agent_service/requirements.txt agent_service/requirements.txt
RUN pip install -r agent_service/requirements.txt

COPY config.yaml ./
COPY agent_service/ agent_service/

# 不用 root 运行
RUN useradd --create-home --uid 1000 app && mkdir -p /app/logs /models/hf && chown -R app /app /models
USER app

WORKDIR /app/agent_service
EXPOSE 8002
# 容器内必须监听 0.0.0.0，端口是否对外暴露由 compose 决定
ENV SERVER__HOST=0.0.0.0
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8002/healthz', timeout=3).status == 200 else 1)"
CMD ["python", "app.py"]
