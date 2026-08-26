# Deploying the public demo

The public build is **retrieval-only**. Answer generation is disabled because
the demo has no authentication or rate limiting, and a public URL with a live
API key is money anyone on the internet can spend. Retrieval runs entirely on
the host CPU, costs nothing, and is the half of the project worth showing.

---

## Streamlit Community Cloud (recommended)

Free, deploys **straight from this GitHub repo** — no Docker, no second git
remote — and redeploys automatically on every push.

### 1. Deploy

Go to **[share.streamlit.io](https://share.streamlit.io)** → **Create app** →
**Deploy a public app from GitHub**, then:

| Field | Value |
|---|---|
| Repository | `VikhyatKoppalgithub/HR_Chatbot_Assistant` |
| Branch | `main` |
| Main file path | `app.py` |
| App URL | `hr-rag-assistant` (or whatever is free) |

Click **Deploy**. The first build takes **5–10 minutes**: it installs PyTorch,
downloads two ~90MB models, and builds the index. Watch the build log.

No secrets to configure. The demo never calls the Anthropic API.

### 2. Verify the deployed URL — not localhost

Your app will be at `https://<your-app-name>.streamlit.app`. Check:

- [ ] Page loads; sidebar shows **51 passages** and *Dense search: on*
- [ ] The blue "Public demo — retrieval only" notice is visible
- [ ] Clicking an example chip returns ranked passages with scores
- [ ] Switching strategy in the sidebar changes the ranking
- [ ] The **Cross-encoder rerank** toggle changes results
- [ ] **Compare all strategies** tab renders five columns
- [ ] Ask *"Can I expense a gym membership?"* and compare: `bm25` should put
      **Non-Reimbursable Expenses** first (wrong), `dense` should put
      **Wellness Stipend** first (right). That contrast is the demo.

### 3. Add the link to your portfolio

In `vikhyat-portfolio/content/projects.ts`, find `slug: "hr-rag-assistant"` and
add one line:

```ts
links: {
  github: "https://github.com/VikhyatKoppalgithub/HR_Chatbot_Assistant",
  demo: "https://<your-app-name>.streamlit.app",
},
```

`ProjectCard.tsx` already renders a **Live demo** button with
`target="_blank" rel="noopener noreferrer"` whenever `demo` is set — no
component changes needed. Commit and push; Vercel redeploys automatically.

---

## The one line that decides whether the build succeeds

At the top of `requirements.txt`:

```
--extra-index-url https://download.pytorch.org/whl/cpu
```

On Linux, plain `pip install torch` pulls the **CUDA** build — an ~800MB wheel
that unpacks to ~2.5GB of GPU libraries that are entirely unused on a CPU host,
and which exceeds every free tier's limits. The CPU index publishes versions
tagged `+cpu`, which sort above the plain release under PEP 440, so pip prefers
them. On macOS there is no separate CPU build, so this is a no-op locally.

**If the Streamlit build fails on disk or memory, this line is the first thing
to check.**

---

## Alternative: Hugging Face Spaces (Docker)

Use this if Streamlit's ~1GB memory ceiling turns out to be too tight. Spaces
gives 16GB free and runs the richer stdlib UI (`server.py`) unchanged via the
included `Dockerfile`.

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   — **SDK: Docker**, blank template, CPU basic, public.
2. Clone it and copy the project in:

```bash
git clone https://huggingface.co/spaces/YOUR-HF-USERNAME/hr-rag-assistant ~/Desktop/hf-space
```

```bash
rsync -a --exclude '.git' --exclude 'data/index' --exclude '.env' --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.vscode' --exclude 'WALKTHROUGH.md' ~/Desktop/Projects/hr-rag-assistant/ ~/Desktop/hf-space/
```

3. Prepend this frontmatter to the **Space's** `README.md` only (not the GitHub
   one — GitHub renders it as an ugly table):

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

4. `git add . && git commit -m "Deploy" && git push`

The Dockerfile sets `DEMO_MODE=1`, `HOST=0.0.0.0`, `PORT=7860`, installs
CPU-only torch, and bakes both models plus the index into the image so cold
starts are fast.

Verify generation is genuinely blocked, not just hidden:

```bash
curl -s -X POST https://YOUR-HF-USERNAME-hr-rag-assistant.hf.space/api/ask -d '{"question":"pto"}'
```

That must return the **403** demo-mode message.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Build fails, out of space/memory | CUDA torch | keep the `--extra-index-url` line first in `requirements.txt` |
| App boots then errors on first query | index missing | `get_engine()` in `app.py` builds it on boot — check the log for that step |
| Very slow first load | models downloading | normal on Streamlit (~1–2 min); the Docker path bakes them in instead |
| Streamlit app sleeps | free tier idles after inactivity | it wakes on the next visit; takes ~30s |
| HF: `Permission denied` on cache | running as root | keep `USER user` and `HF_HOME` in the Dockerfile |
| HF: builds but never serves | port mismatch | `app_port: 7860` must match `PORT` in the Dockerfile |

---

## Running the full version, with generation

Locally, `DEMO_MODE` is unset so everything works:

```bash
python server.py
```

To deploy privately *with* generation, set `DEMO_MODE=0` and add
`ANTHROPIC_API_KEY` as a platform secret — but only on a **private**
deployment, or behind authentication.
