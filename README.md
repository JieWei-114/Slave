# Slave

A local-first AI assistant — built as a learning project to explore full-stack AI application development.

This is a self-hosted chat assistant that runs on Ollama, with retrieval-augmented generation (RAG), semantic memory, local voice input/output, and an optional desktop shell. I built it as a solo project to learn how the pieces of a modern AI application fit together: streaming inference, context assembly, vector search, provider abstractions, and a reactive frontend. It works, but it is experimental: single-user by design, no automated test suite yet, and many of the "smart" behaviors are heuristics I'm still iterating on.

Stack: Angular 21 (signals, standalone components) + FastAPI + MongoDB, with Ollama (or any OpenAI-compatible server) for inference.

## What this project explores

### RAG and context assembly

The chat pipeline assembles context from multiple sources before calling the model: uploaded files (PDF/DOCX/TXT/MD/CSV/JSON/YAML/code, extracted with PyPDF2 and python-docx), semantic memory, web search, and conversation history. Each source carries a confidence weight (files 0.99, memory 0.85, web 0.65, history 0.0) so factual claims are grounded in retrievable sources rather than chat history. Web search fans out across SearXNG, DuckDuckGo, Serper, and Tavily with fallback and quota tracking.

### Provider abstraction

Instead of hardcoding Ollama, inference goes through an `LLMProvider` interface (`backend/app/providers/`). `LLM_PROVIDER=ollama` is the default; `LLM_PROVIDER=openai_compat` points at any OpenAI-compatible endpoint — vLLM, llama.cpp server, LM Studio, or the OpenAI API itself. This was an exercise in designing a clean seam between the app and the model runtime.

### Vector search

Memory embeddings go through a `VectorStore` abstraction (`backend/app/vector/`): the default is a naive cosine-similarity scan over MongoDB documents (simple, zero extra infrastructure), and `VECTOR_STORE=qdrant` switches to Qdrant for a real ANN index. Startup handles reindexing/backfill so you can switch stores on an existing database.

### Streaming

Server-Sent Events end-to-end: FastAPI async generators emit `token`, `reasoning_token`, `metadata`, `done`, and `error` events, and the Angular side parses the stream with a hand-rolled fetch-based SSE parser (needed for POST bodies and stop-generation control). Reasoning tokens render separately from the answer.

### Hallucination-mitigation heuristics (experimental)

An attempt to reduce fabricated answers from small local models. These are heuristics I'm exploring, not guarantees:

- **Entity validation** — extract entities from the answer (spaCy if installed, regex fallback) and fuzzy-match them against source documents (substring, stem, acronym expansion). Many unverified entities cap the reported confidence.
- **Source separation** — history and follow-up context are marked non-factual (confidence 0.0), so the model is prompted to draw facts only from files/memory/web.
- **Reasoning veto** — if the model's internal reasoning contains phrases like "cannot confirm" or "no reliable source", the answer is refused or its confidence capped.

The confidence numbers this produces are indicative, not rigorous — see limitations below.

### Local voice pipeline

Speech-to-text via faster-whisper and text-to-speech via Piper, both running locally through `/voice` endpoints. Models are lazily downloaded on first use (~150MB Whisper base, ~60MB Piper voice) and cached in a Docker volume.

### Hugging Face model management

Search the HF hub for GGUF models and list per-repo quant files (`GET /models/search`, `GET /models/search/{repo}/files`), then pull them into Ollama with SSE progress streaming via `hf.co/{repo}:{quant}` names (`POST /models/pull`). Installed models can be listed and deleted.

### Desktop shell

A Tauri v2 shell (`frontend/src-tauri/`) wraps the Angular SPA into a native desktop app that talks to the local backend.

### Topic management

Conversations can be split into topics: a topic break closes the current thread and the LLM generates an overview summary of it, keeping long sessions navigable.

### Other things built along the way

- Centralized configuration: all limits and confidence weights live in `backend/app/config/settings.py`, overridable via environment variables — no magic numbers scattered in code.
- Per-session rules: JSON rules control search providers, follow-up mode, reasoning mode, custom system instructions, and retrieval limits.
- Frontend state entirely on Angular signals (no RxJS state), CSS-variable design system, collapsible reasoning/validation panels, per-response metadata display.

## Technical Stack

**Backend**

- FastAPI 0.128 — async REST API, SSE streaming
- MongoDB 4.6+ with Motor (async driver)
- Ollama (or any OpenAI-compatible server) for inference
- spaCy 3.7+ for entity extraction (optional, regex fallback)
- Pydantic v2, PyPDF2, python-docx
- faster-whisper (STT), Piper (TTS)
- Qdrant (optional vector store)

**Frontend**

- Angular 21 — standalone components, signal-based state (no RxJS)
- Fetch-based SSE client for streaming
- Marked.js for markdown rendering
- Angular CLI build with SSR support; Tauri v2 for desktop

**Infrastructure**

- Node.js 20+ / Python 3.10+
- Docker Compose profiles for dev/prod, CPU/GPU Ollama, and SearXNG

## Quick Start

### Docker (recommended)

Everything runs in containers — Docker is the only host dependency.

**Requirements**

1. Docker Desktop (Windows/Mac) or Docker Engine + Compose (Linux)
   - Windows: [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)
   - Mac: [Docker Desktop](https://docs.docker.com/desktop/install/mac-install/)
   - Linux: [Docker Engine](https://docs.docker.com/engine/install/) + [Compose](https://docs.docker.com/compose/install/)
2. (Optional) NVIDIA GPU: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

**Step 1: Choose a profile**

```
Profile                    Frontend                 Backend             Ollama  Use case
-------------------------  -----------------------  ------------------  ------  --------------
`dev` + `ollama-cpu`       `ng serve` (hot reload)  `uvicorn --reload`  CPU     Development
`prod-spa` + `ollama-cpu`  Static (Nginx)           Production          CPU     Static SPA
`prod-ssr` + `ollama-cpu`  SSR (Node)               Production          CPU     SSR
`dev` + `ollama-gpu`       `ng serve`               `uvicorn --reload`  GPU     Dev with GPU
`prod-spa` + `ollama-gpu`  Static (Nginx)           Production          GPU     SPA with GPU
```

**Step 2: Start all services**

From the project root:

```bash
# Static SPA (good default)
docker compose --profile prod-spa --profile ollama-cpu up --build

# Development mode (hot reload)
docker compose --profile dev --profile ollama-cpu up --build

# SSR
docker compose --profile prod-ssr --profile ollama-cpu up --build
```

With NVIDIA GPU (requires NVIDIA Container Toolkit):

```bash
# Dev with GPU
OLLAMA_URL=http://ollama-gpu:11434 docker compose --profile dev --profile ollama-gpu up --build

# SPA with GPU
OLLAMA_URL=http://ollama-gpu:11434 docker compose --profile prod-spa --profile ollama-gpu up --build
```

Windows PowerShell (GPU example):

```powershell
$env:OLLAMA_URL="http://ollama-gpu:11434"; docker compose --profile prod-spa --profile ollama-gpu up --build
```

**Step 3: Wait for build and startup**

First startup takes 5-15 minutes depending on connection speed: it downloads base images (Python, Node, MongoDB, Nginx, Ollama), installs Python and Node dependencies, downloads the spaCy model (`en_core_web_sm`), and builds the frontend (prod profiles only).

Success indicators in the logs:

```
mongo       - Waiting for connections on port 27017
searxng     - [uwsgi] spawned uWSGI worker
backend     - INFO:     Application startup complete
ollama      - Listening...
frontend    - Compiled successfully
```

**Step 4: Pull an Ollama model**

The Ollama container starts with no models. Pull one:

```bash
# Enter the Ollama container
docker exec -it slave-ollama-1 bash

# Inside container: pull a model (choose one)
ollama pull qwen2.5:3b
ollama pull gemma3:1b
ollama pull <model>

# Exit container
exit
```

Or from the host (find the container name first if it differs):

```bash
docker ps | grep ollama

docker exec -it slave-ollama-1 ollama pull qwen2.5:3b
docker exec -it slave-ollama-1 ollama pull gemma3:1b
docker exec -it slave-ollama-1 ollama pull <model>
```

Or via the API (no `docker exec` needed — `POST /models/pull` streams progress via SSE and accepts a plain Ollama name or a Hugging Face GGUF reference):

```bash
curl -N -X POST http://localhost:8000/models/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "qwen2.5:3b"}'

# Or a Hugging Face GGUF model (discover via GET /models/search)
curl -N -X POST http://localhost:8000/models/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"}'
```

**Step 5: Access the application**

```
Service         URL                          Description
--------------  ---------------------------  ---------------------
Frontend (SPA)  http://localhost:4173        Main UI (prod SPA)
Frontend (SSR)  http://localhost:4000        Main UI (prod SSR)
Frontend (Dev)  http://localhost:4200        Main UI (dev mode)
Backend API     http://localhost:8000/docs   FastAPI Swagger UI
SearXNG         http://localhost:8080        Private search engine
MongoDB         mongodb://localhost:27017    Database (direct)
Ollama          http://localhost:11434       LLM API
```

**Step 6: Verify everything works**

```bash
# API health
curl http://localhost:8000/health
# Expected: { "status": "healthy", "database": "connected", "version": "1.0.0" }

# Installed Ollama models
curl http://localhost:11434/api/tags
```

Then open the frontend (4173/4000/4200 depending on profile), create a chat session, and send a message.

### Common commands

```bash
# Stop all services
docker compose --profile prod-spa --profile ollama-cpu down

# Restart (without rebuild)
docker compose --profile prod-spa --profile ollama-cpu up

# Rebuild after code changes
docker compose --profile prod-spa --profile ollama-cpu up --build

# View logs
docker compose logs -f backend
docker compose logs -f frontend-spa
docker compose logs -f ollama

# Clean up (remove volumes, fresh start)
docker compose down -v
```

### Data persistence

All data lives in Docker named volumes and survives `docker compose down`:

- `mongo_data` — MongoDB (conversations, memories, rules)
- `ollama_data` — Ollama models (multi-GB)
- `qdrant_data` — Qdrant vector store (when `VECTOR_STORE=qdrant`)
- `voice_models` — Whisper STT + Piper TTS model files

To delete everything:

```bash
docker volume rm slave_mongo_data slave_ollama_data slave_qdrant_data slave_voice_models
```

### First-run downloads

After these one-time downloads, inference runs fully offline (web search still goes out to the internet when triggered):

- Embedding model (~90MB)
- Whisper `base` STT model (~150MB, on first voice use)
- Piper TTS voice (~60MB)
- Ollama models (GB-scale)

### API keys (optional)

Only needed for the paid web search providers (Serper, Tavily). Skip this if you only use the free providers (DuckDuckGo, SearXNG).

Register at [serper.dev](https://serper.dev/) and/or [tavily.com](https://www.tavily.com/), then:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
SERPER_API_KEY=your_serper_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Then restart:

```bash
docker compose --profile prod-spa --profile ollama-cpu up --build
```

### Troubleshooting

**Port already in use**

```bash
# Windows
netstat -ano | findstr :4173

# Linux/Mac
lsof -i :4173

# Kill the process or change the port in docker-compose.yml
```

**Ollama GPU not working**

```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
# If this fails, the NVIDIA Container Toolkit isn't installed correctly
```

**Backend can't connect to MongoDB**

- Wait 10-20 seconds after `docker compose up` for MongoDB to initialize
- Check logs: `docker compose logs mongo`

**MongoDB connection failed (manual setup)**

```bash
mongosh --eval "db.adminCommand('ping')"
# Default connection string: mongodb://127.0.0.1:27017

Start-Service MongoDB          # Windows
sudo systemctl start mongod    # Linux
docker start mongo             # Docker
```

**Ollama not responding**

```bash
curl http://localhost:11434/api/tags
ollama serve
ollama pull <model>
```

**Web search returns nothing**

- Check SearXNG is running if using local search
- Review quotas in `serper_quota` / `tavily_quota`
- Check backend logs for API errors
- Check provider toggles in the Rules UI and API keys in `.env`

**Entity extraction errors**

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

**First build is slow** — normal; subsequent builds use the Docker layer cache. Use `--build` only after code changes.

### Manual setup (without Docker)

**1. Prerequisites**

```bash
node --version    # v20.19+ or v22.12+
python --version  # 3.10+
mongod --version  # 4.6+

# Install Ollama (https://ollama.ai)
ollama pull <model>
```

**2. Backend**

```bash
cd backend

python -m venv venv
source venv/bin/activate  # Windows: venv/Scripts/activate

pip install -r requirements.txt

# Optional: spaCy for NLP entity extraction (Python 3.10-3.13)
pip install spacy
python -m spacy download en_core_web_sm

cp .env.example .env   # edit with your settings

uvicorn app.main:app --reload
```

API: http://127.0.0.1:8000 — docs at http://127.0.0.1:8000/docs

**3. Frontend**

```bash
cd frontend
npm install
npx ng serve
```

Frontend: http://localhost:4200

**4. Database**

```powershell
Start-Service MongoDB                 # Windows service
```

```bash
docker run -d --name mongo -p 27017:27017 -v /data/db:/data/db mongo:7   # Docker
mongod --dbpath /data/db                                                 # manual
```

**5. Local web search (SearXNG)**

```bash
cd searxng
# Edit settings.yml — set a strong secret_key
docker compose up -d
```

SearXNG: http://localhost:8080

## Configuration

- Backend: copy `.env.example` to `.env` and fill in your config. All limits and confidence weights live in `backend/app/config/settings.py` and can be overridden via environment variables.
- Models shown in the UI: edit `backend/app/config/ai_models.py` to add any Ollama-compatible model.

## Desktop (Tauri)

The Angular frontend also runs as a native desktop app via [Tauri v2](https://v2.tauri.app/).

Prerequisites: [Rust](https://rustup.rs/) (stable) and Node 20+.

```bash
cd frontend

# Dev: opens a desktop window loading the ng serve dev server (localhost:4200)
npm run tauri:dev

# Build: bundles the static SPA into a native app (frontend/src-tauri/target/release/bundle/)
npm run tauri:build
```

The desktop app talks to the backend at `http://127.0.0.1:8000`, so the backend must be running (e.g. `docker compose --profile dev up`).

## Development

**Backend** (Ruff for linting/formatting):

```bash
cd backend
ruff check --fix .
ruff format .
```

**Frontend** (Prettier + ESLint):

```bash
cd frontend
npm run format:check
npm run format:fix
npm run lint
npm run lint:fix
```

## Design decisions

- **Mongo as source of truth, Qdrant optional.** MongoDB already stores everything else, so the default vector search is a plain cosine scan over Mongo docs — zero extra infrastructure, fine at small scale. Qdrant is a config switch (`VECTOR_STORE=qdrant`) for when a real ANN index matters; startup reindex/backfill makes switching painless.
- **Provider abstraction instead of hardcoding Ollama.** Partly to learn interface design, partly practical: `openai_compat` means the same app runs against vLLM, llama.cpp, LM Studio, or OpenAI without code changes.
- **API-key auth is optional and off by default.** This is a local-first, single-user app; adding accounts and auth flows would be complexity without benefit for the intended use. An optional API key exists for exposing the backend beyond localhost.
- **Messages embedded in session documents.** Simplicity first: one read gets a whole conversation. The tradeoff is MongoDB's 16MB document limit, which caps very long sessions — a known constraint I'd revisit with a separate messages collection.
- **No test suite yet.** I prioritized prototype speed while the architecture was churning. This is the project's biggest gap and the first thing on the list to fix.

## Known limitations

- No automated tests.
- Single-user design: optional API key, no user accounts or multi-tenancy.
- The confidence numbers are heuristic and indicative, not rigorous — the hallucination-mitigation pipeline reduces obvious fabrication but guarantees nothing.
- Messages embedded in session docs limit very long conversations (16MB document cap).
- First run downloads models from Hugging Face, so setup is not fully offline.
- Small local models (1B-7B) produce noticeably weaker answers than hosted frontier models; the retrieval pipeline helps but doesn't close the gap.

## Privacy notes

All inference, storage, and voice processing run locally; there is no telemetry. Web search (including DuckDuckGo/SearXNG queries) goes out to the internet when triggered, and Serper/Tavily are only used if you configure and enable them. If you expose the backend beyond localhost, put it behind HTTPS (reverse proxy), restrict `CORS_ORIGINS`, enable MongoDB auth, and set a strong SearXNG `secret_key`.

## System requirements

Minimum: 8GB RAM, 4-core CPU, ~40GB disk (Docker images + one 7B model + databases). Recommended: 16GB+ RAM, SSD, and a GPU for faster inference. Rough model memory: 7B ≈ 6-8GB, 13B ≈ 10-12GB, 33B+ ≈ 20GB+.

## Ideas to explore next

1. GraphRAG — graph-enhanced retrieval, possibly with Tree-sitter AST parsing for code
2. Image generation via an external ComfyUI instance (models from Civitai)
3. Separate messages collection to remove the session-size cap

## Built with

[Ollama](https://ollama.ai) · [FastAPI](https://fastapi.tiangolo.com) · [Angular](https://angular.dev) · [MongoDB](https://www.mongodb.com) · [spaCy](https://spacy.io) · [SearXNG](https://docs.searxng.org) · [Serper](https://serper.dev/) · [Tavily](https://www.tavily.com/) · [Tauri](https://v2.tauri.app/) · [Qdrant](https://qdrant.tech/)

## Screenshot

![App screenshot](/frontend/src/assets/images/app_screenshot.png)
