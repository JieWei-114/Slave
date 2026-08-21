# Slave – Advanced Local AI Assistant Platform

**Your data. Your AI. Your rules.** A AI assistant platform engineered for privacy, reliability, and transparency. Built on Angular 21 + FastAPI + MongoDB with intelligent multi-source reasoning, real-time streaming, and zero-hallucination architecture.


## Why Slave?

Slave implements a controlled, source-aware Retrieval-Augmented Generation (RAG) pipeline to ground AI responses in verifiable data and prevent hallucinations.
Instead of relying solely on the language model’s internal knowledge, the system retrieves relevant information from multiple sources and injects it into the prompt with explicit confidence weighting.

**100% Transparent Decision-Making** — See exactly why the AI answered what it did  
**Privacy-First Architecture** — Everything runs locally, no telemetry, no cloud dependencies  
**Anti-Hallucination System** — Multi-layer validation prevents fabricated responses  
**Real-Time Streaming** — Token-by-token streaming with stop-generation control  
**Intelligent Context Management** — Multi-source context fusion with confidence tracking  
**Production-Ready** — Clean architecture, comprehensive error handling, fully typed


## Core Capabilities

### **1. Multi-Source Intelligent Context - RAG**

The AI doesn't just answer—it **intelligently assembles context** from multiple sources:

#### **File Intelligence**

- **Supported Formats**: PDF, DOCX, TXT, MD, CSV, JSON, YAML, code files
- **Smart Extraction**: PyPDF2, python-docx, fallback encoding (UTF-8 → Latin-1)
- **Attachment Persistence**: Files stored in MongoDB with 30-day expiry
- **Context Injection**: File content automatically prioritized as authoritative source

#### **Multi-Provider Web Search**

- **Providers**: SearXNG (privacy), DuckDuckGo (free), Serper (Google proxy), Tavily (research)
- **Smart Routing & Fallback Chain**: Auto-detects research queries with graceful degradation
- **Advance Search Mode**: Distributes queries across all enabled providers, deduplicates results
- **Quota Tracking**: Real-time monitoring of API limits with automatic provider switching
- **Result Ranking**: Sentence-based scoring prioritizes most relevant snippets

#### **URL Content Extraction**

- **Smart Truncation**: Extracts key points for web search enrichment
- **Source Attribution**: Full URL tracking in metadata

#### **Semantic Memory System**

- **Embedding-Based Search**: Cosine embedding similarity matching with configurable threshold (0.3 default)
- **Categorized Storage**: `preference/fact`, `important`, `other`
- **Auto-Memory**: Automatically saves important exchanges for future context
- **Confidence Scoring**: Each memory tracks confidence level (0.95 default)

#### **Conversation History**

- **Configurable Depth**: Default 10 messages, customizable per-session
- **Follow-Up Mode**: Resolves pronouns/references against previous assistant answer
- **Smart Truncation**: Per-message character limits prevent context overflow

### **2. Anti-Hallucination Validation Pipeline (fabricate facts)**

**Solution**: Multi-layer validation and model limitation

#### **Source Layer Separation**

Slave separates **factual sources** and **contextual sources**:

**Factual (High Confidence)**

- Files (0.99) — User-uploaded, authoritative
- Memory (0.85) — Verified past knowledge
- Web (0.65) — External grounding

**Contextual (Supporting Only)**

- History (0.0) — Continuity, not facts
- Follow-Up (0.0) — Reference resolution

**Why this matters**: The AI **cannot cite conversation history as a fact source**. It must draw facts from verifiable sources.

#### **Entity Validation Service**

**Dual-Mode Extraction**:

1. **NLP Mode** (spaCy): Extracts PERSON, ORG, GPE, DATE, MONEY, etc.
2. **Pattern Mode** (Fallback): Regex-based with intelligent common-word filtering

**Strategy Fuzzy Matching**:

- **Exact Match**: Direct substring matching
- **Partial Match**: Lower-case fuzzy matching
- **Stem Match**: Handles plurals/tenses ("company" ↔ "companies")
- **Acronym Expansion**: "FBI" ↔ "Federal Bureau Investigation"

**Factual Guard** - Risk Levels:

- HIGH: 6+ Unverified entities → Cap confidence to 0.4
- MED: 3-5 Unverified entities → Cap confidence to 0.5
- LOW: <3 Unverified entities → Cap confidence to 0.6
- NONE: All entities verified → No cap

#### **Source Conflict Detection**: 

- Identifies contradictions between information sources

#### **Reasoning Veto System**  (Shows why answers were refused or confidence capped)

**Hard Vetoes** 

- "cannot confirm", "no reliable source", "conflicting sources"
- "no access to", "not covered in context", "outside my knowledge"

**Soft Vetoes** 

- "uncertain", "speculative", "probably", "assuming", "might"
- "not sure", "unclear", "low confidence"

#### **Confidence Calculation**

```
Initial Confidence (from source type) : 85%
    ↓
Hard Veto? → REFUSE ANSWER
Soft Veto? → Cap at 0.6
    ↓
Factual Guard (unverified entities)? → Cap by risk level
    ↓
Final Confidence
```

All stages tracked in metadata for full transparency.

### **3. Streaming Architecture**

**Real-Time Event Streaming** via Server-Sent Events (SSE):

```typescript
Event Types:
├─ token             // Final answer generation
├─ reasoning_token   // AI's thinking process generation
├─ metadata          // Confidence, sources, validation results
├─ done              // Stream complete
└─ error             // Error with recovery suggestions
```

**Benefits**:

- Token-by-token rendering for perceived speed
- Stop-generation control (user can interrupt)
- Transparent reasoning display (see AI's thought process)


### **4. Centralized Configuration System**

**Single Source of Truth**: All limits defined in [`settings.py`](backend/app/config/settings.py)

```python
# Example: Web search limits
WEB_SEARCH_RESULTS_PER_PROVIDER: int = 10
CHAT_WEB_SNIPPET_MAX_CHARS: int = 800
CHAT_WEB_TOTAL_MAX_CHARS: int = 6000

# Confidence levels
CONFIDENCE_FILE: float = 0.99
CONFIDENCE_MEMORY: float = 0.85
CONFIDENCE_WEB: float = 0.65
```

**No magic numbers in code** — everything configurable via environment variables.


### **5. Custom Rules Engine**

Per-session:

```json
{
   "searxng": true,
   "duckduckgo": true,
   "tavily": false,
   "serper": false,
   "advanceSearch": false,
   "followUpEnabled": true,
   "reasoningEnabled": false,
   "customInstructions": "You are a Python expert...",
   "webSearchLimit": 10,
   "memorySearchLimit": 10,
   "historyLimit": 10
}
```

**Use Cases**:

- Define AI behavior and personality via system prompts
- Pre-configured rule templates for common use cases
- Multiple rules per session with priority management
- Dynamic rule injection into AI context
- Disable web search for sensitive conversations
- Enable reasoning mode for complex problem-solving
- Custom system instructions per session (personality, expertise)

### **6. Hugging Face Model Integration**

- Search the Hugging Face hub for GGUF models and list per-repo quant files (`GET /models/search`, `GET /models/search/{repo}/files`)
- Pull models into Ollama with SSE progress streaming via `hf.co/{repo}:{quant}` names (`POST /models/pull`), plus list/delete installed models

### **7. Modern UI/UX**

- Clean, responsive design with CSS design system
- Skeleton loaders and smooth animations
- Collapsible reasoning/validation panels
- Color-coded confidence and risk indicators
- Real-time typing indicators
- Displays comprehensive metadata for each AI response


## Technical Stack

### **Backend**

- **FastAPI** 0.128.0 — Async-first, type-safe REST API
- **MongoDB** 4.16+ — Document database with Motor async driver
- **Ollama** — Local LLM inference
- **spaCy** 3.7+ — NLP for entity extraction (optional, graceful fallback)
- **Pydantic** v2 — Strict data validation
- **PyPDF2** — PDF text extraction
- **Streaming**: Server-Sent Events (SSE)
- **python-docx** — DOCX text extraction

### **Frontend**

- **Angular** 21.1.0 — Standalone components
- **Vite** — Fast build tool
- **Signal-based State** — Signal-based reactive state management (no RxJS)
- **EventSource** — SSE for real-time streaming
- **Marked.js** — Markdown rendering
- **Styling**: CSS Variables design system
- **Build**: Angular CLI with SSR support

### **Infrastructure**

- **Node.js** 20+ / **Python** 3.10+
- **MongoDB** 4.6+ (local/Docker/cloud)
- **Docker** (for MongoDB + SearXNG)

## Quick Start

### **Docker One-Command Startup**
**Zero host dependencies.** Everything runs in containers — no heavy installation required on your machine.

#### **What You Need on Your Computer**
1. **Docker Desktop** (Windows/Mac) or **Docker Engine + Docker Compose** (Linux)
   - Windows: [Download Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)
   - Mac: [Download Docker Desktop](https://docs.docker.com/desktop/install/mac-install/)
   - Linux: [Install Docker Engine](https://docs.docker.com/engine/install/) + [Docker Compose](https://docs.docker.com/compose/install/)

2. **(Optional) NVIDIA GPU Support**
   - Windows/Linux with NVIDIA GPU: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

#### **Step-by-Step First-Time Setup**

**Step 1: Choose Your Profile**

Pick the frontend + backend + Ollama combination that fits your use case:

```
Profile                    Frontend                 Backend             Ollama  Use Case       
-------------------------  -----------------------  ------------------  ------  -------------- 
`dev` + `ollama-cpu`       `ng serve` (hot reload)  `uvicorn --reload`  CPU     Development    
`prod-spa` + `ollama-cpu`  Static (Nginx)           Production          CPU     Production SPA 
`prod-ssr` + `ollama-cpu`  SSR (Node)               Production          CPU     Production SSR 
`dev` + `ollama-gpu`       `ng serve`               `uvicorn --reload`  GPU     Dev with GPU   
`prod-spa` + `ollama-gpu`  Static (Nginx)           Production          GPU     Prod SPA + GPU 
```

**Step 2: Start All Services**

Open a terminal in the project root directory and run:

**Production SPA (Recommended for First-Time Users)**

```bash
# find where the project is located
cd ~/Slave
```

```bash
docker compose --profile prod-spa --profile ollama-cpu up --build
```

**Development Mode (Hot Reload)**

```bash
docker compose --profile dev --profile ollama-cpu up --build
```

**Production SSR**

```bash
docker compose --profile prod-ssr --profile ollama-cpu up --build
```

**With NVIDIA GPU** (requires NVIDIA Container Toolkit)

```bash
# Dev with GPU
OLLAMA_URL=http://ollama-gpu:11434 docker compose --profile dev --profile ollama-gpu up --build

# Prod SPA with GPU
OLLAMA_URL=http://ollama-gpu:11434 docker compose --profile prod-spa --profile ollama-gpu up --build
```

**Windows PowerShell** (GPU example)

```powershell
$env:OLLAMA_URL="http://ollama-gpu:11434"; docker compose --profile prod-spa --profile ollama-gpu up --build
```

---

**Step 3: Wait for Build & Startup**

First-time startup will take **5-15 minutes** depending on your internet speed:

- Downloads base images (Python, Node, MongoDB, Nginx, Ollama)
- Installs Python dependencies (FastAPI, spaCy, PyPDF2, etc.)
- Downloads spaCy NLP model (`en_core_web_sm`)
- Installs Node dependencies (Angular, etc.)
- Builds frontend (dev: skipped, prod: full build)

Watch for these success indicators in logs:

```
mongo       - Waiting for connections on port 27017
searxng     - [uwsgi] spawned uWSGI worker
backend     - INFO:     Application startup complete
ollama      - Listening...
frontend    - Compiled successfully
```


**Step 4: Pull an Ollama Model**
The Ollama container is running but **has no models yet**. Pull one:

```bash
# Enter the Ollama container
docker exec -it slave-ollama-1 bash

# Inside container: Pull a model (choose one)
ollama pull qwen2.5:3b
ollama pull gemma3:1b
ollama pull <model>

# Exit container
exit
```

**Alternative: Pull from host** (if Ollama container name is different)

```bash
# Find container name:
docker ps | grep ollama

#Pull a model
docker exec -it slave-ollama-1 ollama pull qwen2.5:3b
docker exec -it slave-ollama-1 ollama pull gemma3:1b
docker exec -it slave-ollama-1 ollama pull <model>

```

**Alternative: Pull via the API (Hugging Face GGUF or Ollama names)**

No `docker exec` needed — `POST /models/pull` streams progress via SSE and accepts either a plain Ollama name or a Hugging Face GGUF reference:

```bash
curl -N -X POST http://localhost:8000/models/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "qwen2.5:3b"}'

# Or a Hugging Face GGUF model (discover via GET /models/search)
curl -N -X POST http://localhost:8000/models/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"}'
```

**Step 5: Access the Application**

Open your browser:

```
Service             URL                          Description           
------------------  ---------------------------  --------------------- 
**Frontend (SPA)**  http://localhost:4173        Main UI (prod SPA)    
**Frontend (SSR)**  http://localhost:4000        Main UI (prod SSR)    
**Frontend (Dev)**  http://localhost:4200        Main UI (dev mode)    
**Backend API**     http://localhost:8000/docs   FastAPI Swagger UI    
**SearXNG**         http://localhost:8080        Private search engine 
**MongoDB**         `mongodb://localhost:27017`  Database (direct)     
**Ollama**          http://localhost:11434       LLM API               
```

**Step 6: Verify Everything Works**

1. **Check API Health**

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{ "status": "healthy", "database": "connected", "version": "1.0.0" }
```

2. **Check Ollama Models**

```bash
curl http://localhost:11434/api/tags
```

3. **Open the Frontend**  

Navigate to http://localhost:4173 (or 4000/4200 depending on profile).  
Create a chat session and test a message.

#### **Common Commands**

**Stop all services**
```bash
docker compose --profile prod-spa --profile ollama-cpu down
```

**Restart services** (without rebuild)
```bash
docker compose --profile prod-spa --profile ollama-cpu up
```

**Rebuild after code changes**
```bash
docker compose --profile prod-spa --profile ollama-cpu up --build
```

**View logs**
```bash
docker compose logs -f backend
docker compose logs -f frontend-spa
docker compose logs -f ollama
```

**Clean up (remove volumes, fresh start)**
```bash
docker compose down -v
```

#### **Data Persistence**

All data persists in Docker named volumes:

- `mongo_data`: MongoDB database (conversations, memories, rules)
- `ollama_data`: Ollama models (multi-GB, survives restarts)
- `qdrant_data`: Qdrant vector store (memory embeddings when `VECTOR_STORE=qdrant`)
- `voice_models`: Whisper STT + Piper TTS model files

Even after `docker compose down`, your data remains. To delete:

```bash
docker volume rm slave_mongo_data slave_ollama_data slave_qdrant_data slave_voice_models
```

#### **Troubleshooting**

**Port already in use**

```bash
# Check what's using port 4173 (or 8000, 11434, etc.)
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

# If this fails, NVIDIA Container Toolkit isn't installed correctly
```

**Backend can't connect to MongoDB**
- Wait 10-20 seconds after `docker compose up` for MongoDB to initialize
- Check logs: `docker compose logs mongo`

**First build is slow**
- Normal. Subsequent builds use Docker layer cache (much faster)
- Use `--build` only when you change code

---

#### **API Keys**

To enable paid web search providers (Serper, Tavily)

**Skip this step** if you only ant to use free providers (DuckDuckgo, SearXNG), create a `.env`

- Register your account to get our API key
**SerperDev** - (https://serper.dev/)
**Tavily** - (https://www.tavily.com/)

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
SERPER_API_KEY=your_serper_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**Then restart**:

```bash
docker compose --profile prod-spa --profile ollama-cpu up --build
```

### **Alternative: Manual Setup (Without Docker)**
If you prefer to run services directly on your host machine (not recommended for beginners):

### **1. Prerequisites**

```bash
# Check versions
node --version    # v20.19+ or v22.12+
python --version  # 3.10+
mongod --version  # 4.6+

# Install Ollama (https://ollama.ai)
ollama pull llama2:7b
ollama pull <model>
```

### **2. Backend Setup**

```bash
cd backend

# Clean up (optional but recommended)
rm -rf venv
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Install spaCy for NLP entity extraction (Python 3.10-3.13 only)
pip install spacy
python -m spacy download en_core_web_sm

# Create .env
cp .env.example .env
Edit .env with your settings

# Start server
uvicorn app.main:app --reload
```

**API**: http://127.0.0.1:8000  
**API Docs**: http://127.0.0.1:8000/docs

### **3. Frontend Setup**

```bash
cd frontend

# Clean up (optional but recommended)
rm -rf node_modules .angular package-lock.json
npm cache clean --force

# Install dependencies
npm install

# Start development server
npx ng serve # --verbose /to see logs
```

**Frontend URL**: http://localhost:4200

### **4. Database Setup**

**Option A: MongoDB Service (Windows)**

```powershell
Start-Service MongoDB
```

**Option B: Docker**

```bash
docker run -d --name mongo -p 27017:27017 -v C:\data\db:/data/db mongo:7
```

**Option C: Manual Start**

```bash
mongod --dbpath /data/db
```

### **5. Local Web Search (SearXNG)**

```bash
cd searxng

# Edit settings.yml - set a strong secret_key
docker compose up -d
```

**SearXNG URL**: http://localhost:8080

## Configuration

### **Backend Environment (.env)**

Copy .env.example to .env and fill in your config

### **AI Models Configuration**

Edit `backend/app/config/ai_models.py`:
```python
AVAILABLE_MODELS = [
    {
        "id": "llama2:13b",
        "name": "Llama 2 13B",
        "description": "Powerful reasoning",
        'size': '13B'
    },
    # Add custom models
]
```

## Development

### **Backend Development**

- Backend: Ruff for linting/formatting

**Code Quality**

```bash
cd backend

# Lint and format with Ruff
ruff check --fix .
ruff format .
```

**Testing**

```bash
pytest
pytest --cov=app
pytest -v tests/....
pytest -v
```

### **Frontend Development**

- Frontend: Prettier + ESLint

**Code Quality**

```bash
cd frontend

# Format with Prettier
npm run format:check
npm run format:fix

# Linting (ESLint)
npm run lint
npm run lint:fix
```

**Testing**

```bash
npm test
npm run test:coverage
```

## Desktop (Tauri)

The Angular frontend can also run as a native desktop app via [Tauri v2](https://v2.tauri.app/).

**Prerequisites:** [Rust](https://rustup.rs/) (stable) and Node 20+.

```bash
cd frontend

# Dev: opens a desktop window loading the ng serve dev server (localhost:4200)
npm run tauri:dev

# Build: bundles the static SPA build into a native app (frontend/src-tauri/target/release/bundle/)
npm run tauri:build
```

The desktop app talks to the backend at `http://127.0.0.1:8000`, so the backend must be running via Docker Compose (e.g. `docker compose --profile dev up`).

## Troubleshooting

### **MongoDB Connection Failed**

```bash
# Check MongoDB running
mongosh --eval "db.adminCommand('ping')"

# Verify connection string in settings
# Default: mongodb://127.0.0.1:27017

# Start MongoDB service
Start-Service MongoDB  # Windows
sudo systemctl start mongod  # Linux

# Or use Docker
docker start mongo

# Create database (auto-created on first run)
mongosh slave
```

### **Ollama Not Responding**

```bash
# Check Ollama running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Pull missing models
ollama pull <model>
```

### **Web search no results**

- Check SearXNG is running if using local search
- Review quotas in: serper_quota, tavily_quota
- Check backend logs for API errors
- Check provider settings in Rules UI. Verify API keys in `.env` are set correctly.

### **Entity extraction errors**

```bash
# Install spaCy: 
pip install spacy 
python -m spacy download en_core_web_sm
```

## Privacy & Security

### **Privacy Guarantees**

**100% Local Processing**: All AI inference runs on your machine  
**No Telemetry**: Zero tracking, analytics, or data collection  
**Local Data Storage**: All conversations stay in your MongoDB  
**Local-First**: After the first-run downloads below, inference runs fully offline. Web search (including DuckDuckGo and SearXNG queries) still goes out to the internet when triggered  
**First run downloads**: embedding model (~90MB), Whisper `base` STT model (~150MB, on first voice use), Piper TTS voice (~60MB), and Ollama models (GB-scale)  
**Optional Web** — Serper/Tavily only when you enable/trigger search  
**Open Source**: Full code transparency, audit at any time
**No Third-Party CDNs** — All assets bundled

### **External Services**

**Web Search Providers** (Serper, Tavily):
- Only used when explicitly requested
- Can be disabled in settings
- Local alternative: SearXNG (fully private)

### **Security Best Practices**

1. **MongoDB**

```bash
# Enable MongoDB authentication
mongod --auth --bind_ip 127.0.0.1
# Set strong MongoDB passwords
```

2. **Reverse Proxy** (Nginx/Caddy)

```nginx
# Use HTTPS with reverse proxy (nginx/Caddy)
location /api {
    proxy_pass http://127.0.0.1:8000;
}
```

3. **Firewall Rules**

```bash
ufw allow 4200/tcp  # Frontend
ufw allow 8000/tcp  # Backend (localhost only recommended)
```

4. **CORS Restrictions**

```env
CORS_ORIGINS=["http://localhost:4200"]  # Whitelist only
```

5. **Set strong `secret_key` for SearXNG**

```
<random 32-64 chars>
```

6. **Update dependencies regularly**

## Performance

### **System Requirements**

**Minimum**:

- 8GB RAM
- 4-core CPU
- ~40GB disk space (Docker images + one 7B model + databases)
- MongoDB 4.6+

**Recommended**:

- 16GB+ RAM (for larger models)
- 8-core CPU
- SSD for MongoDB
- GPU for faster inference (optional)

### **Model Memory Usage (minimum)**

- 7B parameters: ~6-8GB RAM
- 13B parameters: ~10-12GB RAM
- 33B+ parameters: 20GB+ RAM

### **Optimizations**

- MongoDB connection pooling
- Async I/O throughout
- Efficient embedding caching
- Lazy loading in frontend
- SSE streaming reduces memory
- Signal-based reactivity (minimal rerenders)
- Configure Ollama GPU acceleration (CUDA/ROCm)
- Limit concurrent sessions (1-3 for 7B models on 8GB RAM)

## Documentation

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc
- **README**: files in major directories

## Contributing

Built with best-in-class open-source tools:

- [**Ollama**](https://ollama.ai) — Local LLM inference engine
- [**FastAPI**](https://fastapi.tiangolo.com) — Modern Python web framework
- [**Angular**](https://angular.dev) — Enterprise web framework
- [**MongoDB**](https://www.mongodb.com) — Document database
- [**spaCy**](https://spacy.io) — Industrial-strength NLP
- [**SearXNG**](https://docs.searxng.org) — Privacy-respecting metasearch
- [**SerperDev**](https://serper.dev/) - Google-search results in real-time 
- [**Tavily**](https://www.tavily.com/) - Tavily is the real‑time search engine for AI agents and RAG workflows

## FAQ 

**Q: Is my data truly private?**  
A: Yes. All LLM inference runs locally via Ollama. Web search is optional and only triggers when you enable it. MongoDB is local. Zero telemetry.

**Q: Do I need a GPU?**  
A: No, but it helps. Ollama works on CPU, just slower.

**Q: Can I use custom models?**  
A: Yes! Any Ollama-compatible model. Add to `ai_models.py`.

**Q: Why is reasoning separate from answer?**  
A: Separating internal thoughts from user-facing response improves quality and enables validation.

**Q: What's the veto system?**  
A: AI checks its own reasoning. If it said "cannot confirm" internally but generated a confident answer externally, veto system catches this inconsistency.

**Q: How does entity validation work?**  
A: Extracts names/places from answer, checks if they appear in source documents. Flags unverified entities.

**Q: Why fuzzy matching?**  
A: Exact matching fails on plurals ("APIs" vs "API"), partial phrases ("John Smith" in "Dr. John Smith"), etc.

**Q: How does follow-up mode work?**  
A: When enabled, the AI treats the **previous assistant answer** as the **primary context**. Pronouns/references resolve against it first, then fall back to conversation history.

**Q: Do I need spaCy?**  
A: Recommended. System falls back to pattern-based extraction.

## Roadmap

### **Planned Features**
1. ~~Plugin system for custom model providers~~ ✅ Implemented — LLM provider abstraction (`LLM_PROVIDER=ollama` or `LLM_PROVIDER=openai_compat` for vLLM / llama.cpp server / LM Studio / OpenAI)
2. ~~Local Model Runtime Layer (ollama => lama.cpp or vLLM)~~ ✅ Supported via `LLM_PROVIDER=openai_compat`
3. Hugging Face integration - AI model repository
4. ~~Vector database abstraction layer (Qdrant)~~ ✅ Implemented — `VECTOR_STORE=mongo` (default) or `VECTOR_STORE=qdrant`
5. Tauri Desktop (UI)
6. ~~Voice input & output (speech → text → AI → speech)~~ ✅ Implemented — local STT (faster-whisper) + TTS (Piper) via `/voice` endpoints
7. Advanced GraphRAG + Tree-sitter AST + Tooling (Graph-enhanced retrieval / GraphRAG - experimental)
8. Image & Video Generation - Stable Diffusion integration (From: Civitai, Run: ComfyUI)

## App screenshot
![App screenshot](/frontend/src/assets/images/app_screenshot.png)

**Built with ❤️ for privacy-conscious developers**
