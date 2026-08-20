# Deploying the public demo

The deployed build is **retrieval-only**. Answer generation is disabled because
the server has no authentication or rate limiting, and a public link with a live
API key means anyone on the internet can spend your Anthropic credits. Retrieval
runs entirely locally on the host and costs nothing.

`DEMO_MODE=1` is enforced **server-side** in `server.py` — hiding the button in
the UI is cosmetic, so `/api/ask` returns `403` regardless of what the browser
sends.

---

## Why Hugging Face Spaces

The app needs PyTorch plus two transformer models — roughly 1–1.5 GB of RAM.

| Platform | Free tier | Verdict |
|---|---|---|
| **HF Spaces** | 16 GB RAM, 2 vCPU | ✅ built for ML demos |
| Render | 512 MB | ❌ OOMs on torch |
| Vercel / Netlify | serverless bundle caps | ❌ torch is far too large |
| Fly.io / Railway | needs a card | ⚠️ works, but not free |

---

## Steps

### 1. Create the Space

Go to **[huggingface.co/new-space](https://huggingface.co/new-space)**:

- **Space name:** `hr-rag-assistant`
- **License:** MIT
- **SDK:** **Docker** → *Blank*
- **Hardware:** CPU basic (free)
- **Visibility:** Public

### 2. Clone it locally

```bash
git clone https://huggingface.co/spaces/YOUR-HF-USERNAME/hr-rag-assistant ~/Desktop/hf-space
```

### 3. Copy the project in

```bash
rsync -a --exclude '.git' --exclude 'data/index' --exclude '.env' --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.vscode' --exclude 'WALKTHROUGH.md' ~/Desktop/Projects/hr-rag-assistant/ ~/Desktop/hf-space/
```

### 4. Add the Space header to its README

Spaces need YAML frontmatter to know how to build. Prepend **exactly this** to
the top of `~/Desktop/hf-space/README.md`, above the `# Northwind HR Assistant`
line:

```
---
title: HR RAG Assistant
emoji: 📋
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
```

This block only exists in the Space's copy. Don't add it to the GitHub README —
GitHub renders it as an ugly table.

### 5. Push

```bash
cd ~/Desktop/hf-space && git add . && git commit -m "Deploy retrieval-only demo" && git push
```

The first build takes **5–10 minutes** — it installs PyTorch, downloads both
models, and builds the index. Watch the **Logs** tab. Later pushes are faster
thanks to layer caching.

### 6. Verify the deployed app, not localhost

Your URL will be:

```
https://YOUR-HF-USERNAME-hr-rag-assistant.hf.space
```

Check all of these against that URL, not `127.0.0.1`:

- [ ] Page loads and the status line shows `51 passages · 7 docs · dense on`
- [ ] **No "Ask Claude" button** and the demo notice is visible
- [ ] Clicking an example chip returns passages with scores
- [ ] Switching bm25 / dense / hybrid changes the ranking
- [ ] The rerank toggle changes results
- [ ] **Compare all strategies** renders five columns
- [ ] Generation really is blocked, not just hidden:

```bash
curl -s -X POST https://YOUR-HF-USERNAME-hr-rag-assistant.hf.space/api/ask -d '{"question":"pto"}'
```

That must return the `403` demo-mode message. If it returns an answer, `DEMO_MODE`
did not take effect — check the Dockerfile `ENV` block.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Build fails on `pip install torch` | wrong index URL | keep `--index-url https://download.pytorch.org/whl/cpu` |
| Build succeeds, app never starts | port mismatch | `app_port: 7860` in frontmatter must match `PORT` in the Dockerfile |
| `Permission denied` writing cache | running as root | keep the `USER user` line and `HF_HOME` |
| App starts but returns 500 | index missing | confirm `python cli.py ingest` ran in the build log |
| Cold start is slow | models downloading at runtime | confirm the pre-download `RUN` step is in the build log |

---

## Running the full version with generation

Locally, `DEMO_MODE` is unset, so everything works:

```bash
python server.py
```

To deploy a private instance *with* generation, set `DEMO_MODE=0` and add
`ANTHROPIC_API_KEY` as a Space secret (**Settings → Variables and secrets**).
Only do this on a **private** Space, or add authentication first.
