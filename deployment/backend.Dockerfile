FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# 实验四 S4：宿主机本地大模型（LM Studio，OpenAI 兼容接口）。容器内经 host.docker.internal 访问宿主机。
ENV LLM_BASE_URL=http://host.docker.internal:1234/v1 LLM_API_KEY=lm-studio LLM_MODEL=qwen2.5-7b-instruct-1m LLM_TIMEOUT=90
WORKDIR /app
COPY backend/requirements.lock.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && useradd --uid 10001 --create-home course
COPY deployment/requirements.lock.txt /tmp/deployment-requirements.txt
RUN pip install --no-cache-dir -r /tmp/deployment-requirements.txt
COPY backend/ backend/
COPY recommender/ recommender/
COPY scripts/student_package.py scripts/student_package.py
COPY deployment/*.py deployment/
USER 10001
EXPOSE 8000
ENTRYPOINT ["python", "-m", "deployment.serve"]
