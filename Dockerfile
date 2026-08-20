# Container image for the public retrieval-only demo.
#
# Targets Hugging Face Spaces (Docker SDK), but nothing here is HF-specific
# beyond the default port -- it runs on any container host.
#
# Two decisions worth understanding:
#
#   1. CPU-ONLY TORCH. `pip install torch` pulls the CUDA build by default:
#      roughly 2.5GB of GPU libraries that are dead weight on a CPU host and
#      will blow past image size limits. The CPU index below is ~200MB.
#
#   2. BAKE THE MODELS AND INDEX AT BUILD TIME. Otherwise every cold start
#      downloads ~180MB of model weights and re-embeds the corpus before it
#      can serve a single request. Doing it here makes startup near-instant.

FROM python:3.11-slim

# Hugging Face Spaces runs containers as UID 1000. Create that user up front
# and give the ML libraries a home directory they can actually write caches to
# -- otherwise sentence-transformers fails on a read-only default cache path.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

USER user

# Pre-download both models, then build the index into the image.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
 && python cli.py ingest

# DEMO_MODE=1 disables answer generation. The server enforces this itself, so
# removing the UI button is not the security boundary -- see server.py.
# HOST=0.0.0.0 is required in a container; never set that on a laptop.
ENV DEMO_MODE=1 \
    HOST=0.0.0.0 \
    PORT=7860

EXPOSE 7860

CMD ["python", "server.py"]
