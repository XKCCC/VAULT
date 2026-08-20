# VAULT AML adapter image
# torch 2.6.0 + CUDA 12.4 runtime (matches requirements.txt pin)
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Local models baked into the image (~3.5 GB). To skip, mount them at run time:
#   -v /host/models:/app/emo/models
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('BAAI/bge-m3', local_dir='emo/models/bge-m3'); \
    snapshot_download('BAAI/bge-reranker-v2-m3', local_dir='emo/models/bge-reranker-v2-m3')"

COPY emo ./emo

EXPOSE 8000
ENV AML_DREAM_BATCH=32

# Required at run time: AML_LLM_KEY_ENV's referenced key (e.g. OPENAI_API_KEY),
# AML_LLM_BASE / AML_LLM_MODEL for the internal dream LLM (gpt-4o-mini for
# official AML runs), and AML_API_KEY when auth is enabled.
CMD ["python", "emo/aml/server.py", "--host", "0.0.0.0", "--port", "8000"]
