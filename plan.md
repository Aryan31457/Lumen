# Photo Memory: Complete Implementation Plan

**Project:** A self-hosted "Google Photos"-style app that groups faces, captions every image, and answers questions like *"Show me the picture where Aryan is swimming"*, built with GenAI, RAG, vector embeddings, LangChain and LangGraph.

**Core design rule:** captions and embeddings never contain names. They contain stable `person_id` tokens (`{{p:1}}`). Names are resolved at display time, so renaming a person is a single-row update.

> **Note on "JEV":** I treated Feature 4 as an *evaluation + observability + optimisation* layer (LangSmith tracing, retrieval metrics, LLM-as-judge). If JEV means something else, only Chapter 14 changes.

---

## Table of Contents

1. [Decisions and assumptions](#1-decisions-and-assumptions)
2. [Architecture](#2-architecture)
3. [Repository structure](#3-repository-structure)
4. [Setup guide](#4-setup-guide)
5. [Database schema](#5-database-schema)
6. [Vector store setup](#6-vector-store-setup)
7. [Core modules (config, models, tokens)](#7-core-modules)
8. [Prompt library (all prompts)](#8-prompt-library)
9. [Feature 1: Face detection, assignment and clustering](#9-feature-1-faces)
10. [Feature 2: Captioning](#10-feature-2-captioning)
11. [Ingestion graph (LangGraph)](#11-ingestion-graph)
12. [Feature 3: Query graph (RAG)](#12-feature-3-query-graph)
13. [People management: rename, merge, split, delete](#13-people-management)
14. [Feature 4: Evaluation, tracing and optimisation](#14-feature-4-evaluation)
15. [API and UI](#15-api-and-ui)
16. [Testing](#16-testing)
17. [Privacy and ethics](#17-privacy-and-ethics)
18. [Week-by-week milestones](#18-milestones)
19. [Troubleshooting](#19-troubleshooting)
20. [Learning checklist, product scope, future work](#20-learning-and-scope)

---

## 1. Decisions and assumptions

| Topic | Decision | Why |
|---|---|---|
| Face model | InsightFace `buffalo_l` (SCRFD detector + ArcFace 512-d) | Best open-source accuracy/effort ratio |
| Clustering | HDBSCAN (batch) + nearest-neighbour assignment (incremental) | No need to guess the number of people |
| Captioning | Hosted vision LLM **or** local VLM via Ollama, behind LangChain `init_chat_model` | Swap providers by changing `.env` |
| Image embeddings | OpenCLIP ViT-B-32 (512-d) | Text-to-image search signal |
| Text embeddings | `BAAI/bge-small-en-v1.5` (384-d) | Small, fast, good quality |
| Vector DB | Qdrant (named vectors, payload filters, built-in RRF fusion) | Filter-by-person plus semantic search in one query |
| Relational DB | PostgreSQL | People, faces, images, templates |
| Orchestration | LangGraph (ingestion graph and query graph) | Fan-out, loops, human-in-the-loop, checkpoints |
| API / UI | FastAPI / Streamlit | Fastest path to a demo |
| Tracing / eval | LangSmith + custom metrics | See Chapter 14 |

**Identity model (one ID per person):**

| Concept | ID | Mutable? |
|---|---|---|
| A face box in one image | `face_id` | No |
| One real individual (a cluster) | `person_id` ("userId") | **Never** changes or gets reused |
| Display name | `people.display_name` | Yes, `NULL` means render "User {person_id}" |

**Token formats stored in captions:**

| Token | Meaning |
|---|---|
| `{{p:7}}` | Person 7 (resolved to the name or "User 7" at render time) |
| `{{f:88}}` | Face 88, not yet assigned to any person (resolved to "an unidentified person"; rewritten to `{{p:N}}` when clustering assigns it) |

---

## 2. Architecture

```
                    ┌─────────────────────────────────┐
                    │  Streamlit UI (gallery, people, │
                    │  search chat, eval dashboard)   │
                    └───────────────┬─────────────────┘
                                    │ REST
                    ┌───────────────▼─────────────────┐
                    │           FastAPI               │
                    └──────┬─────────────────┬────────┘
                           │                 │
              ┌────────────▼───────┐   ┌─────▼──────────────────┐
              │  INGESTION GRAPH   │   │  QUERY GRAPH           │
              └────────────┬───────┘   └─────┬──────────────────┘
                           │                 │
    ┌──────────────────────▼─────────────────▼───────────────────┐
    │ InsightFace · VLM · OpenCLIP · bge-small · LLM (parse/grade)│
    └──────────────────────┬─────────────────┬───────────────────┘
                           │                 │
            ┌──────────────▼──┐   ┌──────────▼───────┐  ┌──────────────┐
            │ Qdrant          │   │ PostgreSQL       │  │ Image files  │
            │  faces, photos  │   │ people/faces/... │  │ data/images  │
            └─────────────────┘   └──────────────────┘  └──────────────┘
                      └────────── LangSmith traces ──────────┘
```

**Ingestion graph**

```
START → load ─(duplicate)→ END
          └→ detect_faces ─┬→ caption ───────┐
                           └→ clip_embed ────┴→ build_templates → index → END
```

**Query graph**

```
START → parse_query → resolve_persons ─(unknown name)──────────────→ respond → END
                          │  (ambiguous: interrupt → user picks)          ▲
                          ▼                                               │
                      retrieve → grade ─(enough good results)─────────────┤
                          ▲        └─(too few, attempts left)→ rewrite    │
                          └──────────────────────────────────────┘        │
                                   └─(attempts exhausted)─────────────────┘
```

**Batch jobs (outside the graphs):** `cluster_job` runs HDBSCAN over unassigned faces, creates or matches people, and rewrites `{{f:ID}}` to `{{p:ID}}` in captions.

---

## 3. Repository structure

```
photo-memory/
├── README.md
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── sql/
│   └── schema.sql
├── data/                      # gitignored
│   ├── images/
│   ├── thumbs/
│   ├── qdrant/
│   └── pg/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── db.py
│   ├── models.py              # lazy model loaders
│   ├── qdrant_store.py
│   ├── tokens.py              # render, sanitize, join
│   ├── prompts.py             # ALL prompts live here
│   ├── faces.py
│   ├── captioner.py
│   ├── clustering.py
│   ├── people_service.py      # rename / merge / split / delete / assign
│   ├── graphs/
│   │   ├── __init__.py
│   │   ├── ingest_graph.py
│   │   └── query_graph.py
│   ├── api.py
│   └── cli.py
├── ui/
│   └── streamlit_app.py
├── eval/
│   ├── queries.jsonl
│   ├── face_labels.csv
│   ├── run_eval.py
│   └── metrics.py
└── tests/
    ├── test_tokens.py
    ├── test_people_service.py
    └── test_query_parsing.py
```

---

## 4. Setup guide

### 4.1 Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.11 | 3.12 also works for most packages; use 3.11 if InsightFace wheels fail |
| Docker + Docker Compose | recent | For Qdrant and Postgres |
| Git | any | |
| C++ build tools | Windows only | Install "Microsoft C++ Build Tools" before `pip install insightface` |
| NVIDIA GPU (optional) | 8 GB+ VRAM | CPU works for a few hundred photos |

### 4.2 Create the project and environment

```bash
mkdir photo-memory && cd photo-memory
git init
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
mkdir -p app/graphs ui eval tests sql data/{images,thumbs,qdrant,pg}
touch app/__init__.py app/graphs/__init__.py
printf "data/\n.venv/\n.env\n__pycache__/\n" > .gitignore
```

### 4.3 `requirements.txt`

```text
# API / UI
fastapi
uvicorn[standard]
python-multipart
streamlit
httpx

# Config / DB
pydantic>=2
pydantic-settings
psycopg[binary,pool]
qdrant-client

# LangChain / LangGraph
langchain
langchain-core
langgraph
langgraph-checkpoint-postgres
langsmith
langchain-anthropic        # or: langchain-openai
langchain-ollama           # only if using local models

# Vision / embeddings
insightface
onnxruntime                # GPU: onnxruntime-gpu instead
opencv-python-headless
pillow
numpy
scikit-learn
torch                      # GPU build: see pytorch.org
open_clip_torch
sentence-transformers

# Dev / eval
pytest
ragas                      # optional
```

```bash
pip install -r requirements.txt
pip freeze > requirements.lock.txt    # pin versions once it works
```

### 4.4 `docker-compose.yml`

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333", "6334:6334"]
    volumes: ["./data/qdrant:/qdrant/storage"]
    restart: unless-stopped

  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: photos
      POSTGRES_PASSWORD: photos
      POSTGRES_DB: photos
    ports: ["5432:5432"]
    volumes:
      - ./data/pg:/var/lib/postgresql/data
      - ./sql/schema.sql:/docker-entrypoint-initdb.d/001_schema.sql
    restart: unless-stopped
```

```bash
docker compose up -d
docker compose ps                       # both "running"
curl http://localhost:6333/healthz      # Qdrant health check
```

### 4.5 `.env.example` (copy to `.env`)

```bash
DATABASE_URL=postgresql://photos:photos@localhost:5432/photos
QDRANT_URL=http://localhost:6333
IMAGE_DIR=data/images

# --- Pick ONE provider block ---
# A) Hosted (Anthropic)
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-5-5
VLM_PROVIDER=anthropic
VLM_MODEL=claude-sonnet-5-5
ANTHROPIC_API_KEY=sk-ant-...

# B) Hosted (OpenAI)
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-4o-mini
# VLM_PROVIDER=openai
# VLM_MODEL=gpt-4o-mini
# OPENAI_API_KEY=sk-...

# C) Fully local (Ollama): run `ollama pull qwen2.5vl:7b` first
# LLM_PROVIDER=ollama
# LLM_MODEL=qwen2.5vl:7b
# VLM_PROVIDER=ollama
# VLM_MODEL=qwen2.5vl:7b

# --- Tuning (see Chapter 14) ---
MIN_DET_SCORE=0.6
MIN_FACE_PX=40
MIN_BLUR_VAR=40
ASSIGN_THRESHOLD=0.45
CENTROID_MATCH_THRESHOLD=0.55
MIN_CLUSTER_SIZE=3
CAPTION_RETRIES=2
MAX_QUERY_ATTEMPTS=2

# --- Tracing (optional) ---
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=photo-memory
```

### 4.5.1 Model downloads (first run)

InsightFace, OpenCLIP and bge download automatically on first use. To warm the cache:

```bash
python - <<'EOF'
from insightface.app import FaceAnalysis
FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"]).prepare(ctx_id=-1)
import open_clip
open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
from sentence_transformers import SentenceTransformer
SentenceTransformer("BAAI/bge-small-en-v1.5")
print("models ready")
EOF
```

### 4.6 Run order (after all code is written)

```bash
docker compose up -d
python -m app.cli init                         # creates Qdrant collections
python -m app.cli ingest ./sample_photos       # runs the ingestion graph on a folder
python -m app.cli cluster                      # HDBSCAN: creates people, rewrites tokens
uvicorn app.api:app --reload --port 8000       # terminal 1
streamlit run ui/streamlit_app.py              # terminal 2
```

### 4.7 Smoke test

1. `python -m app.cli ingest ./sample_photos` finishes with `indexed: N`.
2. `python -m app.cli cluster` prints the number of people created.
3. Open Streamlit, People tab: rename "User 1" to a real name.
4. Search tab: *"photos of <name>"* returns images, and captions show the new name.

---

## 5. Database schema

`sql/schema.sql` (auto-applied on first `docker compose up`):

```sql
CREATE TABLE people (
  person_id     SERIAL PRIMARY KEY,
  display_name  TEXT CHECK (display_name IS NULL OR char_length(display_name) <= 50),
  merged_into   INT REFERENCES people(person_id),
  deleted       BOOLEAN NOT NULL DEFAULT FALSE,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE images (
  image_id          SERIAL PRIMARY KEY,
  path              TEXT UNIQUE NOT NULL,
  sha256            TEXT UNIQUE NOT NULL,
  width             INT,
  height            INT,
  taken_at          TIMESTAMPTZ,
  gps_lat           DOUBLE PRECISION,
  gps_lon           DOUBLE PRECISION,
  caption_template  TEXT,          -- "{{p:1}} and {{p:7}} are eating pizza on a beach."
  presence_template TEXT,          -- "{{p:1}} and {{p:7}} are present in this image."
  activity_text     TEXT,          -- name-free: "eating pizza on a beach"
  scene             TEXT,
  tags              JSONB NOT NULL DEFAULT '[]',
  token_map         JSONB NOT NULL DEFAULT '[]',  -- [{"alias":"P1","face_id":88,"person_id":1}]
  needs_recaption   BOOLEAN NOT NULL DEFAULT FALSE,
  status            TEXT NOT NULL DEFAULT 'new',  -- new|faces_done|indexed|failed
  error             TEXT,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE faces (
  face_id      SERIAL PRIMARY KEY,
  image_id     INT NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
  person_id    INT REFERENCES people(person_id),      -- NULL = unassigned
  bbox         JSONB NOT NULL,                        -- [x1,y1,x2,y2]
  det_score    REAL,
  blur_var     REAL,
  quality_ok   BOOLEAN NOT NULL DEFAULT TRUE,
  assign_sim   REAL,
  assigned_by  TEXT                                   -- auto | cluster | manual
);

CREATE TABLE image_people (
  image_id   INT NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
  person_id  INT NOT NULL REFERENCES people(person_id),
  PRIMARY KEY (image_id, person_id)
);

CREATE INDEX idx_faces_person   ON faces(person_id);
CREATE INDEX idx_faces_image    ON faces(image_id);
CREATE INDEX idx_ip_person      ON image_people(person_id);
CREATE INDEX idx_images_taken   ON images(taken_at);
CREATE INDEX idx_people_name    ON people(lower(display_name));
```

---

## 6. Vector store setup

Two Qdrant collections:

| Collection | Point ID | Vectors | Payload |
|---|---|---|---|
| `faces` | `face_id` | 512-d ArcFace, cosine | `person_id` (0 = unassigned), `image_id` |
| `photos` | `image_id` | named `activity` (384-d), `clip` (512-d), cosine | `image_id`, `person_ids[]`, `tags[]`, `activity_text`, `taken_at_ts`, `path` |

`app/qdrant_store.py`:

```python
from functools import lru_cache
from qdrant_client import QdrantClient, models
from .config import settings


@lru_cache
def client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collections() -> None:
    c = client()
    if not c.collection_exists("faces"):
        c.create_collection(
            "faces",
            vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
        )
        c.create_payload_index("faces", "person_id", models.PayloadSchemaType.INTEGER)

    if not c.collection_exists("photos"):
        c.create_collection(
            "photos",
            vectors_config={
                "activity": models.VectorParams(size=384, distance=models.Distance.COSINE),
                "clip": models.VectorParams(size=512, distance=models.Distance.COSINE),
            },
        )
        c.create_payload_index("photos", "person_ids", models.PayloadSchemaType.INTEGER)
        c.create_payload_index("photos", "taken_at_ts", models.PayloadSchemaType.FLOAT)
        c.create_payload_index("photos", "tags", models.PayloadSchemaType.KEYWORD)
```

---

## 7. Core modules

### 7.1 `app/config.py`

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://photos:photos@localhost:5432/photos"
    qdrant_url: str = "http://localhost:6333"
    image_dir: str = "data/images"

    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-5-5"
    vlm_provider: str = "anthropic"
    vlm_model: str = "claude-sonnet-5-5"

    min_det_score: float = 0.6
    min_face_px: int = 40
    min_blur_var: float = 40.0
    assign_threshold: float = 0.45
    centroid_match_threshold: float = 0.55
    min_cluster_size: int = 3
    caption_retries: int = 2
    max_query_attempts: int = 2
    min_good_results: int = 3


settings = Settings()
```

### 7.2 `app/db.py`

```python
from contextlib import contextmanager
import psycopg
from psycopg.rows import dict_row
from .config import settings


@contextmanager
def conn():
    """Commits on normal exit, rolls back on exception."""
    with psycopg.connect(settings.database_url, row_factory=dict_row) as c:
        yield c
```

### 7.3 `app/models.py` (lazy model loaders)

```python
from functools import lru_cache
import numpy as np
from langchain.chat_models import init_chat_model
from .config import settings


@lru_cache
def face_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


@lru_cache
def clip_bundle():
    import open_clip, torch
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tok = open_clip.get_tokenizer("ViT-B-32")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(dev).eval(), preprocess, tok, dev


def clip_image_vec(pil_image) -> list[float]:
    import torch
    model, preprocess, _, dev = clip_bundle()
    with torch.no_grad():
        x = preprocess(pil_image).unsqueeze(0).to(dev)
        v = model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
    return v[0].cpu().numpy().astype(float).tolist()


def clip_text_vec(text: str) -> list[float]:
    import torch
    model, _, tok, dev = clip_bundle()
    with torch.no_grad():
        v = model.encode_text(tok([text]).to(dev))
        v = v / v.norm(dim=-1, keepdim=True)
    return v[0].cpu().numpy().astype(float).tolist()


@lru_cache
def text_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed_doc(text: str) -> list[float]:
    return text_embedder().encode(text, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    return text_embedder().encode(BGE_QUERY_PREFIX + text, normalize_embeddings=True).tolist()


@lru_cache
def get_llm():
    return init_chat_model(settings.llm_model, model_provider=settings.llm_provider, temperature=0)


@lru_cache
def get_vlm():
    return init_chat_model(settings.vlm_model, model_provider=settings.vlm_provider, temperature=0)
```

### 7.4 `app/tokens.py` (render, sanitize, join)

```python
import re
from .db import conn

TOKEN = re.compile(r"\{\{([pf]):(\d+)\}\}")
ALIAS = re.compile(r"\bP\d+\b")


def sanitize_name(name: str) -> str:
    """Strip anything that could be mistaken for a token or markup."""
    name = re.sub(r"[{}<>\\`]", "", name or "").strip()
    name = re.sub(r"\s+", " ", name)
    return name[:50]


def load_people() -> dict[int, dict]:
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT person_id, display_name, merged_into, deleted FROM people")
        return {r["person_id"]: r for r in cur.fetchall()}


def root_id(pid: int, people: dict) -> int:
    seen = set()
    while people.get(pid, {}).get("merged_into") and pid not in seen:
        seen.add(pid)
        pid = people[pid]["merged_into"]
    return pid


def display(pid: int, people: dict) -> str:
    pid = root_id(pid, people)
    p = people.get(pid)
    if not p or p["deleted"]:
        return "a person"
    return p["display_name"] or f"User {pid}"


def render(template: str | None, people: dict) -> str:
    if not template:
        return ""

    def sub(m):
        kind, i = m.group(1), int(m.group(2))
        return "an unidentified person" if kind == "f" else display(i, people)

    return TOKEN.sub(sub, template)


def join_names(names: list[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def presence_template(tokens: list[str]) -> str:
    if not tokens:
        return "No identified people in this image."
    verb = "is" if len(tokens) == 1 else "are"
    return f"{join_names(tokens)} {verb} present in this image."


def summarise_group(names: list[str], max_names: int = 4) -> str:
    """For big group photos: 'User 1, User 3 and 3 others'."""
    if len(names) <= max_names:
        return join_names(names)
    shown = names[: max_names - 1]
    return join_names(shown + [f"{len(names) - len(shown)} others"])
```

---

## 8. Prompt library

All prompts live in `app/prompts.py`. Placeholders use `<<UPPER_CASE>>` markers and are filled with `.replace()` so the text can contain braces and quotes safely. Structured output (Pydantic models) enforces JSON shape, so prompts describe *content rules* rather than JSON syntax.

| # | Prompt | Used by | Model |
|---|---|---|---|
| P1 | `CAPTION_SYSTEM` | caption node | VLM |
| P2 | `CAPTION_USER` / `CAPTION_USER_NO_PEOPLE` | caption node | VLM |
| P3 | `CAPTION_REPAIR` | caption retry | VLM |
| P4 | `QUERY_PARSE_SYSTEM` | parse_query node | LLM |
| P5 | `GRADE_SYSTEM` | grade node | LLM |
| P6 | `REWRITE_SYSTEM` | rewrite node | LLM |
| P7 | `ANSWER_SYSTEM` (optional) | respond node | LLM |
| P8 | `CAPTION_JUDGE_SYSTEM` | evaluation | LLM |
| P9 | `EVAL_QUERY_GEN_SYSTEM` | eval-set bootstrapping | LLM |

```python
# app/prompts.py

# ---------- P1: caption system ----------
CAPTION_SYSTEM = """You are a precise photo-captioning assistant for a private photo library.
You describe only what is visibly present in the image.
You never guess or invent names, ages, nationalities, relationships, or occupations.
When unsure about a detail, you leave it out or stay generic.
Your captions are short, factual, and written in present continuous tense."""

# ---------- P2: caption user prompt (people marked) ----------
CAPTION_USER = """The image shows people marked with numbered boxes. The labels present are: <<LABELS>>.

Rules:
1. Refer to a marked person ONLY by their label (for example P1, P2). Never replace a label with a name, pronoun-based description, or any other identifier.
2. People who have NO box must be called "a person", "another person", or "a group of people". Never give them a label.
3. Write ONE sentence for `caption` describing what the marked people are doing together or separately, naming every labelled person who is visibly taking part. Include the setting if clear. Example style: "P1 and P2 are eating pizza on a beach."
4. `activity` is the same description with ALL person references removed. It starts with a verb phrase and has at most 12 words, lowercase. Example: "eating pizza on a beach".
5. `scene` is the place or setting in 1 to 5 words, lowercase. Example: "beach at sunset".
6. `tags` is 3 to 8 lowercase single concepts (activities, objects, scene, weather, time of day, mood). Never include labels, names, or the words "person" or "people".
7. `people_visible` lists the labels you can actually see taking part.
8. Do not mention boxes, labels as drawings, or image quality. Do not guess gender or age.

Describe this image."""

# ---------- P2b: caption user prompt (no marked people) ----------
CAPTION_USER_NO_PEOPLE = """No people are marked in this image (there may be none, or they may be too small or blurry to identify).

Rules:
1. Do not use labels such as P1 or P2. Refer to any visible people as "a person" or "a group of people".
2. Write ONE sentence for `caption` describing what is happening and where.
3. `activity` is the same description with any person references removed, starting with a verb phrase (at most 12 words, lowercase). If there is no activity, describe the scene, for example "mountain lake at sunrise".
4. `scene` is the setting in 1 to 5 words, lowercase.
5. `tags` is 3 to 8 lowercase single concepts. No names.
6. `people_visible` must be an empty list.

Describe this image."""

# ---------- P3: repair prompt appended on retry ----------
CAPTION_REPAIR = """

Your previous answer was rejected for these reasons:
<<ERRORS>>
Allowed labels: <<LABELS>>. Rewrite the full answer and fix every problem."""

# ---------- P4: query parsing ----------
QUERY_PARSE_SYSTEM = """You convert a user's request about their personal photo library into a structured search plan.
Today's date is <<TODAY>>.

Field rules:
- person_names: names or labels exactly as the user wrote them (for example "Aryan", "User 7"). Include only people the user explicitly names. Ignore pronouns such as "me", "him", "her". Never invent names.
- person_match: "all" when the user wants the people together in the same photo ("Aryan and Riya", "together", "with"); "any" when they say "or" or "either". Default "all".
- activity: the action in a short verb phrase with NO person names, for example "swimming". Empty string if none is stated.
- scene_terms: places, objects, or settings mentioned (for example "beach", "birthday cake"). Empty list if none.
- date_from, date_to: ISO dates (YYYY-MM-DD) when the user gives a time range. Resolve relative phrases using today's date ("last month", "in 2024", "last summer"). Use null when no time is mentioned.
- semantic_query: a short, name-free phrase combining activity and scene that should be used for semantic search, for example "swimming in a pool". Use an empty string when the user only wants photos of a person or a date range with no activity or scene.
- limit: the number of photos requested if stated, otherwise 12.

Examples:
Request: Show me the picture where Aryan is swimming
-> person_names=["Aryan"], person_match=all, activity="swimming", scene_terms=[], semantic_query="swimming", limit=12

Request: photos of Aryan and Riya eating pizza on the beach
-> person_names=["Aryan","Riya"], person_match=all, activity="eating pizza", scene_terms=["beach"], semantic_query="eating pizza on a beach", limit=12

Request: all pictures of User 7
-> person_names=["User 7"], person_match=all, activity="", scene_terms=[], semantic_query="", limit=12

Request: sunset pictures from last month
-> person_names=[], activity="", scene_terms=["sunset"], date_from=first day of last month, date_to=last day of last month, semantic_query="sunset", limit=12

Request: show 5 photos of Riya or Aryan hiking
-> person_names=["Riya","Aryan"], person_match=any, activity="hiking", semantic_query="hiking", limit=5"""

# ---------- P5: relevance grader ----------
GRADE_SYSTEM = """You grade photo search results for relevance.
The database has ALREADY guaranteed the people constraints (person filters). Do NOT judge people. Judge ONLY whether the activity or scene described by each candidate matches what the user asked for.

Scoring:
3 = clearly matches the requested activity/scene
2 = likely matches (related or partially described)
1 = weak match
0 = does not match

Be strict. A candidate that merely contains the right person but a different activity gets 0 or 1.
Return one grade per candidate, using the candidate's image_id."""

# ---------- P6: query rewrite ----------
REWRITE_SYSTEM = """A photo search returned too few good matches.
Write ONE alternative search phrase that broadens the original using synonyms or closely related scenes (for example "swimming" -> "people in a pool or sea, water, swimwear").
Rules: no person names, no more than 15 words, do not repeat any previously tried phrase.
Return only the phrase."""

# ---------- P7: optional natural-language answer ----------
ANSWER_SYSTEM = """You are a friendly photo assistant. Using ONLY the provided data, write 1 to 2 sentences telling the user what you found.
Mention how many photos matched and the main people or activity. If confidence is low, say these are the closest matches. Do not invent details and do not mention internal IDs."""

# ---------- P8: caption judge (evaluation) ----------
CAPTION_JUDGE_SYSTEM = """You are an impartial evaluator of image captions. You will see an image and a caption that uses labels like User 1 instead of real names.
Score each criterion from 1 (very poor) to 5 (excellent):
- accuracy: everything stated is visibly true
- completeness: the main subjects and activity are covered
- hallucination: 5 means NO invented details, 1 means many invented details
- label_use: people are referred to only with the given labels; no invented names
Return the scores and a one-sentence rationale. Be critical."""

# ---------- P9: eval query generation ----------
EVAL_QUERY_GEN_SYSTEM = """You help build a test set for a photo search system.
Given a name-free caption and its tags for one photo, write 3 different natural-language search queries a user might type to find exactly this photo. Vary phrasing: one short, one detailed, one using a synonym. Do not include person names. Return the queries only."""
```

---

## 9. Feature 1: Faces

### 9.1 Pipeline

1. Decode the image (OpenCV applies EXIF rotation).
2. Detect faces and compute 512-d ArcFace embeddings with InsightFace.
3. Quality gate: detector score, minimum face size, blur (variance of Laplacian). Faces that fail get `quality_ok=false`, receive **no identity**, and are described as "a person".
4. Store faces in Postgres and embeddings in Qdrant `faces`.
5. **Incremental assignment:** find nearest assigned neighbours. If the best similarity is at least `ASSIGN_THRESHOLD`, vote for the person and assign. Otherwise leave `person_id = NULL` (pending).
6. **Batch clustering (`cluster_job`):** HDBSCAN over pending faces. Each cluster is matched to an existing person by centroid or becomes a new person. Faces are assigned and `{{f:ID}}` tokens become `{{p:ID}}`.

### 9.2 `app/faces.py`

```python
import cv2
import numpy as np
from qdrant_client import models
from .config import settings
from .models import face_app
from .qdrant_store import client


def _blur_var(img, bbox) -> float:
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect(path: str):
    """Returns (bgr_image, faces sorted left-to-right)."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    out = []
    for f in face_app().get(img):
        x1, y1, x2, y2 = [float(v) for v in f.bbox]
        blur = _blur_var(img, f.bbox)
        ok = (
            float(f.det_score) >= settings.min_det_score
            and min(x2 - x1, y2 - y1) >= settings.min_face_px
            and blur >= settings.min_blur_var
        )
        out.append(
            {
                "bbox": [x1, y1, x2, y2],
                "det_score": float(f.det_score),
                "blur_var": blur,
                "quality_ok": ok,
                "embedding": f.normed_embedding.astype(float).tolist(),
            }
        )
    out.sort(key=lambda d: d["bbox"][0])
    return img, out


def try_assign(vec: list[float]) -> tuple[int | None, float]:
    """Nearest-neighbour vote among assigned faces. Returns (person_id|None, best_sim)."""
    res = client().query_points(
        "faces",
        query=vec,
        limit=7,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="person_id", range=models.Range(gt=0))]
        ),
        with_payload=True,
    ).points
    if not res or res[0].score < settings.assign_threshold:
        return None, (res[0].score if res else 0.0)
    votes: dict[int, float] = {}
    for h in res:
        if h.score >= settings.assign_threshold:
            pid = h.payload["person_id"]
            votes[pid] = votes.get(pid, 0.0) + h.score
    best = max(votes, key=votes.get)
    return best, res[0].score
```

### 9.3 `app/clustering.py`

```python
import numpy as np
from sklearn.cluster import HDBSCAN
from qdrant_client import models
from .config import settings
from .db import conn
from .qdrant_store import client
from . import people_service as ps


def run_cluster_job() -> dict:
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT face_id FROM faces WHERE person_id IS NULL AND quality_ok")
        ids = [r["face_id"] for r in cur.fetchall()]

    if len(ids) < settings.min_cluster_size:
        return {"pending": len(ids), "new_people": 0, "matched_existing": 0}

    X = []
    for i in range(0, len(ids), 256):
        pts = client().retrieve("faces", ids=ids[i : i + 256], with_vectors=True)
        X.extend([p.vector for p in pts])
    X = np.asarray(X, dtype=np.float32)       # normalised -> euclidean ranks like cosine
    ids = [p for p in ids][: len(X)]

    labels = HDBSCAN(
        min_cluster_size=settings.min_cluster_size, min_samples=2, metric="euclidean"
    ).fit_predict(X)

    new_people = matched = 0
    for lab in sorted(set(labels) - {-1}):
        idx = np.where(labels == lab)[0]
        face_ids = [ids[i] for i in idx]
        centroid = X[idx].mean(axis=0)
        centroid /= np.linalg.norm(centroid)

        hit = client().query_points(
            "faces",
            query=centroid.tolist(),
            limit=1,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="person_id", range=models.Range(gt=0))]
            ),
            with_payload=True,
        ).points
        if hit and hit[0].score >= settings.centroid_match_threshold:
            pid = hit[0].payload["person_id"]
            matched += 1
        else:
            pid = ps.create_person()
            new_people += 1
        ps.assign_faces(face_ids, pid, by="cluster")

    return {
        "pending": int((labels == -1).sum()),
        "new_people": new_people,
        "matched_existing": matched,
    }
```

---

## 10. Feature 2: Captioning

**Flow:** draw numbered boxes (P1, P2, ...) on a copy of the image (only for quality faces, one box per person), send it to the VLM with the P1/P2 prompt, validate that only allowed labels are used, then substitute labels with `{{p:ID}}` or `{{f:ID}}` tokens.

### `app/captioner.py`

```python
import base64
import re
import cv2
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from .config import settings
from .models import get_vlm
from . import prompts

LABEL_RE = re.compile(r"\bP\d+\b")


class CaptionOut(BaseModel):
    caption: str = Field(description="One sentence; marked people referred to only by labels")
    activity: str = Field(description="Same meaning, all person references removed")
    scene: str = ""
    tags: list[str] = []
    people_visible: list[str] = []


def annotate(img_bgr, labelled: dict[str, dict]):
    """labelled = {"P1": {"bbox": [...]}, ...}"""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    scale = max(1.0, max(h, w) / 1000)
    for label, f in labelled.items():
        x1, y1, x2, y2 = [int(v) for v in f["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), max(2, int(3 * scale)))
        cv2.putText(out, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9 * scale, (0, 255, 0), max(2, int(2 * scale)))
    return out


def to_b64_jpeg(img_bgr, max_side: int = 1024) -> str:
    h, w = img_bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


def validate(out: CaptionOut, labels: list[str]) -> list[str]:
    errs = []
    allowed = set(labels)
    used = set(LABEL_RE.findall(out.caption))
    if not out.caption.strip():
        errs.append("caption is empty")
    if used - allowed:
        errs.append(f"caption uses labels that are not allowed: {sorted(used - allowed)}")
    if LABEL_RE.search(out.activity):
        errs.append("activity must not contain any labels")
    if any(LABEL_RE.search(t) for t in out.tags):
        errs.append("tags must not contain labels")
    if not out.activity.strip():
        errs.append("activity is empty")
    return errs


def caption_image(img_bgr, labelled: dict[str, dict]) -> CaptionOut:
    labels = list(labelled)
    shown = annotate(img_bgr, labelled) if labels else img_bgr
    b64 = to_b64_jpeg(shown)

    user_text = (
        prompts.CAPTION_USER.replace("<<LABELS>>", ", ".join(labels))
        if labels
        else prompts.CAPTION_USER_NO_PEOPLE
    )
    llm = get_vlm().with_structured_output(CaptionOut)

    repair = ""
    for _ in range(settings.caption_retries + 1):
        msg = [
            SystemMessage(content=prompts.CAPTION_SYSTEM),
            HumanMessage(
                content=[
                    {"type": "text", "text": user_text + repair},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]
            ),
        ]
        out: CaptionOut = llm.invoke(msg)
        errs = validate(out, labels)
        if not errs:
            return out
        repair = prompts.CAPTION_REPAIR.replace("<<ERRORS>>", "\n".join(f"- {e}" for e in errs)).replace(
            "<<LABELS>>", ", ".join(labels) or "none"
        )

    # Fallback: generic, name-free caption so ingestion never blocks
    return CaptionOut(
        caption="A photo.", activity=(out.activity or "a photo").strip(), scene=out.scene,
        tags=[t for t in out.tags if not LABEL_RE.search(t)], people_visible=[],
    )
```

---

## 11. Ingestion graph

### `app/graphs/ingest_graph.py`

```python
import hashlib, shutil
from datetime import datetime
from pathlib import Path
from typing import TypedDict
from PIL import Image
from langgraph.graph import StateGraph, START, END
from qdrant_client import models

from ..config import settings
from ..db import conn
from ..qdrant_store import client
from ..models import clip_image_vec, embed_doc
from .. import faces as F
from ..captioner import caption_image
from ..tokens import presence_template, ALIAS


class IngestState(TypedDict, total=False):
    path: str
    image_id: int
    skip: bool
    labelled: dict            # {"P1": {"face_id":88,"person_id":1|None,"bbox":[...]}}
    caption: dict
    clip_vec: list
    caption_template: str
    presence_template: str
    token_map: list
    person_ids: list


# ---------- helpers ----------
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _exif(path: str):
    taken = None
    with Image.open(path) as im:
        w, h = im.size
        ex = im.getexif()
        raw = ex.get_ifd(0x8769).get(0x9003) or ex.get(0x0132)
        if raw:
            try:
                taken = datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
    if taken is None:
        taken = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return w, h, taken


# ---------- nodes ----------
def load(state: IngestState) -> IngestState:
    src = state["path"]
    sha = _sha256(src)
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT image_id FROM images WHERE sha256=%s", (sha,))
        if cur.fetchone():
            return {"skip": True}
        dest = Path(settings.image_dir) / f"{sha[:16]}{Path(src).suffix.lower()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(src).resolve() != dest.resolve():
            shutil.copy2(src, dest)
        w, h, taken = _exif(str(dest))
        cur.execute(
            "INSERT INTO images(path, sha256, width, height, taken_at) VALUES (%s,%s,%s,%s,%s) RETURNING image_id",
            (str(dest), sha, w, h, taken),
        )
        return {"image_id": cur.fetchone()["image_id"], "path": str(dest), "skip": False}


def route_after_load(state: IngestState):
    return END if state.get("skip") else "detect_faces"


def detect_faces(state: IngestState) -> IngestState:
    img, faces = F.detect(state["path"])
    labelled: dict[str, dict] = {}
    seen_persons: set[int] = set()
    n = 0

    with conn() as c, c.cursor() as cur:
        for f in faces:
            pid, sim = (F.try_assign(f["embedding"]) if f["quality_ok"] else (None, 0.0))
            cur.execute(
                """INSERT INTO faces(image_id, person_id, bbox, det_score, blur_var, quality_ok, assign_sim, assigned_by)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING face_id""",
                (state["image_id"], pid, str(f["bbox"]).replace("'", ""), f["det_score"], f["blur_var"],
                 f["quality_ok"], sim, "auto" if pid else None),
            )
            face_id = cur.fetchone()["face_id"]
            client().upsert("faces", points=[models.PointStruct(
                id=face_id, vector=f["embedding"],
                payload={"person_id": pid or 0, "image_id": state["image_id"]})])

            # Only quality faces get a label; one label per person (dedupe)
            if f["quality_ok"] and (pid is None or pid not in seen_persons):
                n += 1
                labelled[f"P{n}"] = {"face_id": face_id, "person_id": pid, "bbox": f["bbox"]}
                if pid:
                    seen_persons.add(pid)
        cur.execute("UPDATE images SET status='faces_done' WHERE image_id=%s", (state["image_id"],))

    # keep the decoded image out of state; captioner re-reads it (cheap) for simplicity
    return {"labelled": labelled}


def caption(state: IngestState) -> IngestState:
    import cv2
    img = cv2.imread(state["path"])
    out = caption_image(img, state["labelled"])
    return {"caption": out.model_dump()}


def clip_embed(state: IngestState) -> IngestState:
    with Image.open(state["path"]) as im:
        return {"clip_vec": clip_image_vec(im.convert("RGB"))}


def build_templates(state: IngestState) -> IngestState:
    labelled = state["labelled"]
    alias_to_token, token_map, person_ids, tokens = {}, [], [], []
    for alias, f in labelled.items():
        tok = f"{{{{p:{f['person_id']}}}}}" if f["person_id"] else f"{{{{f:{f['face_id']}}}}}"
        alias_to_token[alias] = tok
        tokens.append(tok)
        token_map.append({"alias": alias, "face_id": f["face_id"], "person_id": f["person_id"]})
        if f["person_id"]:
            person_ids.append(f["person_id"])

    cap = state["caption"]["caption"]
    template = ALIAS.sub(lambda m: alias_to_token.get(m.group(0), "a person"), cap)
    return {
        "caption_template": template,
        "presence_template": presence_template(tokens),
        "token_map": token_map,
        "person_ids": sorted(set(person_ids)),
    }


def index(state: IngestState) -> IngestState:
    cap = state["caption"]
    tags = [t.lower().strip() for t in cap.get("tags", [])]
    activity = cap["activity"].strip()
    scene = (cap.get("scene") or "").strip()
    search_text = f"{activity}. Scene: {scene}. Tags: {', '.join(tags)}"

    import json
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT taken_at, path FROM images WHERE image_id=%s", (state["image_id"],))
        row = cur.fetchone()
        cur.execute(
            """UPDATE images SET caption_template=%s, presence_template=%s, activity_text=%s, scene=%s,
                                 tags=%s, token_map=%s, status='indexed' WHERE image_id=%s""",
            (state["caption_template"], state["presence_template"], activity, scene,
             json.dumps(tags), json.dumps(state["token_map"]), state["image_id"]),
        )
        for pid in state["person_ids"]:
            cur.execute("INSERT INTO image_people VALUES (%s,%s) ON CONFLICT DO NOTHING",
                        (state["image_id"], pid))

    client().upsert("photos", points=[models.PointStruct(
        id=state["image_id"],
        vector={"activity": embed_doc(search_text), "clip": state["clip_vec"]},
        payload={
            "image_id": state["image_id"],
            "person_ids": state["person_ids"],
            "tags": tags,
            "activity_text": activity,
            "taken_at_ts": row["taken_at"].timestamp() if row["taken_at"] else None,
            "path": row["path"],
        },
    )])
    return {}


# ---------- graph ----------
def build_ingest_graph():
    g = StateGraph(IngestState)
    g.add_node("load", load)
    g.add_node("detect_faces", detect_faces)
    g.add_node("caption", caption)
    g.add_node("clip_embed", clip_embed)
    g.add_node("build_templates", build_templates)
    g.add_node("index", index)

    g.add_edge(START, "load")
    g.add_conditional_edges("load", route_after_load, {"detect_faces": "detect_faces", END: END})
    g.add_edge("detect_faces", "caption")        # fan-out
    g.add_edge("detect_faces", "clip_embed")
    g.add_edge(["caption", "clip_embed"], "build_templates")   # join: waits for both
    g.add_edge("build_templates", "index")
    g.add_edge("index", END)
    return g.compile()


def ingest_folder(folder: str) -> dict:
    graph = build_ingest_graph()
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in exts]
    ok = failed = 0
    for f in files:
        try:
            graph.invoke({"path": f})
            ok += 1
        except Exception as e:                      # mark and continue
            failed += 1
            print("FAILED", f, e)
    return {"files": len(files), "indexed": ok, "failed": failed}
```

**Why a fan-out/join:** captioning (slow, network) and CLIP embedding (GPU) are independent, so LangGraph runs them in parallel. The two parallel nodes write different state keys (`caption`, `clip_vec`), so no reducer is needed.

**Scaling later:** run many images concurrently with `graph.batch(inputs, config={"max_concurrency": 4})`, and swap the loop for a Celery/Redis queue.

---

## 12. Feature 3: Query graph

### 12.1 Behaviour

*"Show me the picture where Aryan is swimming"*

1. **parse_query (LLM, structured output):** `person_names=["Aryan"]`, `semantic_query="swimming"`.
2. **resolve_persons:** `"Aryan"` → `person_id`. Unknown name gives a helpful message. Several matches pause the graph with `interrupt()` so the user can choose.
3. **retrieve:** one Qdrant call that filters by `person_ids` (and dates) and fuses two vector searches (caption text and CLIP) with Reciprocal Rank Fusion.
4. **grade (LLM):** scores activity relevance only (the person constraint is guaranteed by the filter).
5. **route:** enough good results → respond. Too few and attempts left → **rewrite** the query and retrieve again. Out of attempts → respond with the best candidates flagged as low confidence.
6. **respond:** render each caption with **current** names.

### 12.2 `app/graphs/query_graph.py`

```python
import re
from datetime import date, datetime, timezone
from difflib import get_close_matches
from typing import Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from qdrant_client import models

from ..config import settings
from ..db import conn
from ..qdrant_store import client
from ..models import get_llm, embed_query, clip_text_vec
from ..tokens import load_people, render, display, root_id, summarise_group
from .. import prompts


# ---------- structured outputs ----------
class SearchPlan(BaseModel):
    person_names: list[str] = []
    person_match: Literal["all", "any"] = "all"
    activity: str = ""
    scene_terms: list[str] = []
    date_from: str | None = None
    date_to: str | None = None
    semantic_query: str = ""
    limit: int = Field(default=12, ge=1, le=50)


class Grade(BaseModel):
    image_id: int
    score: int = Field(ge=0, le=3)
    reason: str = ""


class GradeList(BaseModel):
    grades: list[Grade]


class QueryState(TypedDict, total=False):
    question: str
    plan: dict
    person_ids: list[int]
    status: str                  # ok | no_person
    unknown: list[str]
    suggestions: list[str]
    semantic_query: str
    tried: list[str]
    attempts: int
    candidates: list[dict]
    graded: list[dict]
    answer: dict


# ---------- helpers ----------
USER_LABEL = re.compile(r"^\s*user\s*(\d+)\s*$", re.I)


def find_people(name: str) -> list[dict]:
    m = USER_LABEL.match(name)
    with conn() as c, c.cursor() as cur:
        if m:
            cur.execute("SELECT person_id, display_name FROM people "
                        "WHERE person_id=%s AND merged_into IS NULL AND NOT deleted", (int(m.group(1)),))
            return cur.fetchall()
        cur.execute("""SELECT person_id, display_name FROM people
                       WHERE merged_into IS NULL AND NOT deleted AND lower(display_name)=lower(%s)""", (name,))
        rows = cur.fetchall()
        if not rows:   # prefix fallback
            cur.execute("""SELECT person_id, display_name FROM people
                           WHERE merged_into IS NULL AND NOT deleted AND lower(display_name) LIKE lower(%s)""",
                        (name + "%",))
            rows = cur.fetchall()
        return rows


def _date_ts(s: str | None, end: bool = False):
    if not s:
        return None
    try:
        d = datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return None
    if end:
        d = d.replace(hour=23, minute=59, second=59)
    return d.replace(tzinfo=timezone.utc).timestamp()


# ---------- nodes ----------
def parse_query(state: QueryState) -> QueryState:
    system = prompts.QUERY_PARSE_SYSTEM.replace("<<TODAY>>", date.today().isoformat())
    plan: SearchPlan = get_llm().with_structured_output(SearchPlan).invoke(
        [SystemMessage(content=system), HumanMessage(content=state["question"])]
    )
    return {"plan": plan.model_dump(), "semantic_query": plan.semantic_query,
            "tried": [plan.semantic_query] if plan.semantic_query else [], "attempts": 0}


def resolve_persons(state: QueryState) -> QueryState:
    ids, unknown = [], []
    for name in state["plan"]["person_names"]:
        cands = find_people(name)
        if not cands:
            unknown.append(name)
        elif len(cands) == 1:
            ids.append(cands[0]["person_id"])
        else:
            # Human-in-the-loop: pause until the UI resumes with a chosen person_id
            chosen = interrupt({
                "type": "disambiguate",
                "name": name,
                "options": [{"person_id": c["person_id"],
                             "label": c["display_name"] or f"User {c['person_id']}"} for c in cands],
            })
            ids.append(int(chosen))
    if unknown:
        with conn() as c, c.cursor() as cur:
            cur.execute("SELECT person_id, display_name FROM people "
                        "WHERE merged_into IS NULL AND NOT deleted")
            known = [r["display_name"] or f"User {r['person_id']}" for r in cur.fetchall()]
        sugg = [s for u in unknown for s in get_close_matches(u, known, n=2, cutoff=0.5)]
        return {"status": "no_person", "unknown": unknown, "suggestions": sugg}
    return {"status": "ok", "person_ids": ids}


def route_after_resolve(state: QueryState):
    return "retrieve" if state["status"] == "ok" else "respond"


def retrieve(state: QueryState) -> QueryState:
    plan, pids = state["plan"], state.get("person_ids", [])
    must = []
    if pids:
        if plan["person_match"] == "all":
            must += [models.FieldCondition(key="person_ids", match=models.MatchValue(value=p)) for p in pids]
        else:
            must.append(models.FieldCondition(key="person_ids", match=models.MatchAny(any=pids)))
    lo, hi = _date_ts(plan.get("date_from")), _date_ts(plan.get("date_to"), end=True)
    if lo or hi:
        must.append(models.FieldCondition(key="taken_at_ts", range=models.Range(gte=lo, lte=hi)))
    flt = models.Filter(must=must) if must else None

    q = state.get("semantic_query", "").strip()
    fetch = max(30, plan["limit"] * 3)

    if not q:  # person/date-only request: no semantic ranking needed
        pts, _ = client().scroll("photos", scroll_filter=flt, limit=fetch, with_payload=True)
        cands = [{"image_id": p.id, "score": 0.0, "payload": p.payload} for p in pts]
        cands.sort(key=lambda c: c["payload"].get("taken_at_ts") or 0, reverse=True)
        return {"candidates": cands}

    res = client().query_points(
        "photos",
        prefetch=[
            models.Prefetch(query=embed_query(q), using="activity", filter=flt, limit=fetch),
            models.Prefetch(query=clip_text_vec(q), using="clip", filter=flt, limit=fetch),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=fetch,
        with_payload=True,
    ).points
    return {"candidates": [{"image_id": p.id, "score": p.score, "payload": p.payload} for p in res]}


def grade(state: QueryState) -> QueryState:
    cands = state.get("candidates", [])
    if not state.get("semantic_query", "").strip() or not cands:
        return {"graded": [dict(c, grade=3, reason="filter-only match") for c in cands]}

    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT image_id, activity_text, scene, tags FROM images WHERE image_id = ANY(%s)",
                    ([c["image_id"] for c in cands[:30]],))
        info = {r["image_id"]: r for r in cur.fetchall()}

    lines = [f"image_id={i} | activity: {r['activity_text']} | scene: {r['scene']} | tags: {', '.join(r['tags'])}"
             for i, r in info.items()]
    prompt = f"User request: {state['question']}\nSearch phrase: {state['semantic_query']}\n\nCandidates:\n" + "\n".join(lines)
    result: GradeList = get_llm().with_structured_output(GradeList).invoke(
        [SystemMessage(content=prompts.GRADE_SYSTEM), HumanMessage(content=prompt)]
    )
    gmap = {g.image_id: g for g in result.grades}
    graded = []
    for c in cands:
        g = gmap.get(c["image_id"])
        if g and g.score >= 2:
            graded.append(dict(c, grade=g.score, reason=g.reason))
    graded.sort(key=lambda c: (-c["grade"], -c["score"]))
    return {"graded": graded}


def route_after_grade(state: QueryState):
    good = len(state.get("graded", []))
    if good >= settings.min_good_results or state.get("attempts", 0) >= settings.max_query_attempts:
        return "respond"
    if not state.get("semantic_query", "").strip():
        return "respond"
    return "rewrite"


def rewrite(state: QueryState) -> QueryState:
    prompt = (f"Original request: {state['question']}\nCurrent phrase: {state['semantic_query']}\n"
              f"Previously tried: {state.get('tried', [])}")
    out = get_llm().invoke([SystemMessage(content=prompts.REWRITE_SYSTEM), HumanMessage(content=prompt)])
    new_q = out.content.strip().strip('"')
    return {"semantic_query": new_q, "tried": state.get("tried", []) + [new_q],
            "attempts": state.get("attempts", 0) + 1}


def respond(state: QueryState) -> QueryState:
    if state.get("status") == "no_person":
        names = ", ".join(state["unknown"])
        msg = f"I don't know anyone named {names} yet."
        if state.get("suggestions"):
            msg += f" Did you mean: {', '.join(state['suggestions'])}?"
        else:
            msg += " Open the People tab to name a face group first."
        return {"answer": {"message": msg, "results": [], "low_confidence": False}}

    people = load_people()
    limit = state["plan"]["limit"]
    good = state.get("graded", [])
    low_conf = len(good) < settings.min_good_results and bool(state.get("semantic_query", "").strip())
    chosen = good if good else state.get("candidates", [])
    chosen = chosen[:limit]

    ids = [c["image_id"] for c in chosen]
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT image_id, path, caption_template, presence_template, taken_at "
                    "FROM images WHERE image_id = ANY(%s)", (ids,))
        rows = {r["image_id"]: r for r in cur.fetchall()}

    results = []
    for c in chosen:
        r = rows.get(c["image_id"])
        if not r:
            continue
        names = [display(p, people) for p in c["payload"].get("person_ids", [])]
        results.append({
            "image_id": r["image_id"],
            "path": r["path"],
            "caption": render(r["caption_template"], people),          # names resolved NOW
            "people": summarise_group(names),
            "taken_at": r["taken_at"].isoformat() if r["taken_at"] else None,
            "confidence": "low" if low_conf else "high",
            "why": c.get("reason", ""),
        })

    if not results:
        msg = "I couldn't find matching photos."
    elif low_conf:
        msg = f"I found {len(results)} possible photo(s). These are the closest matches, so some may not fit."
    else:
        msg = f"Found {len(results)} photo(s)."
    return {"answer": {"message": msg, "results": results, "low_confidence": low_conf}}


def build_query_graph(checkpointer=None):
    g = StateGraph(QueryState)
    for name, fn in [("parse_query", parse_query), ("resolve_persons", resolve_persons),
                     ("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite), ("respond", respond)]:
        g.add_node(name, fn)
    g.add_edge(START, "parse_query")
    g.add_edge("parse_query", "resolve_persons")
    g.add_conditional_edges("resolve_persons", route_after_resolve,
                            {"retrieve": "retrieve", "respond": "respond"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade, {"respond": "respond", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())
```

**Calling it (with human-in-the-loop):**

```python
from langgraph.types import Command
graph = build_query_graph()
cfg = {"configurable": {"thread_id": "session-123"}}

out = graph.invoke({"question": "Show me Aryan swimming"}, cfg)
if "__interrupt__" in out:                      # ambiguous name
    options = out["__interrupt__"][0].value["options"]
    out = graph.invoke(Command(resume=options[0]["person_id"]), cfg)
print(out["answer"])
```

For production, replace `MemorySaver` with `PostgresSaver` (from `langgraph-checkpoint-postgres`) so paused conversations survive restarts.

**Optional natural-language answer:** to use prompt P7, call the LLM in `respond` with `ANSWER_SYSTEM` and a compact JSON of `{count, people, activity, low_confidence}`. Keep the templated message as the fallback, since it is free and deterministic.

---

## 13. People management

All operations touch only **Postgres rows and Qdrant payloads**. No operation re-runs the VLM or recomputes an embedding, with one exception flagged below.

| Action | DB changes | Qdrant changes | Caption changes | VLM call? |
|---|---|---|---|---|
| Rename | `people.display_name` | none | none (resolved at render) | No |
| Merge B into A | faces, image_people, `merged_into` | `faces.person_id`, `photos.person_ids` | rewrite `{{p:B}}` → `{{p:A}}` | No |
| Split faces out of A | new person, faces, image_people | same | rewrite token only for affected images | No (flags image if ambiguous) |
| Assign from cluster job | faces, image_people | same | `{{f:ID}}` → `{{p:N}}` | No |
| Delete person | faces deleted, `deleted=true` | face vectors removed, payloads synced | tokens → "a person" | No |

### `app/people_service.py`

```python
import json
from qdrant_client import models
from .db import conn
from .qdrant_store import client
from .tokens import sanitize_name


# ---------- helpers ----------
def create_person() -> int:
    with conn() as c, c.cursor() as cur:
        cur.execute("INSERT INTO people DEFAULT VALUES RETURNING person_id")
        return cur.fetchone()["person_id"]


def _sync_image(cur, image_id: int) -> list[int]:
    """Single source of truth: person_ids = distinct person_id of the image's faces."""
    cur.execute("SELECT DISTINCT person_id FROM faces WHERE image_id=%s AND person_id IS NOT NULL", (image_id,))
    pids = sorted(r["person_id"] for r in cur.fetchall())
    cur.execute("DELETE FROM image_people WHERE image_id=%s", (image_id,))
    for p in pids:
        cur.execute("INSERT INTO image_people VALUES (%s,%s)", (image_id, p))
    return pids


def _push_payloads(image_pids: dict[int, list[int]]):
    for image_id, pids in image_pids.items():
        try:
            client().set_payload("photos", payload={"person_ids": pids}, points=[image_id])
        except Exception:
            pass  # photo may not be indexed yet


def _replace_token(cur, image_id: int, old: str, new: str, face_ids: set[int] | None = None, person_id=None):
    cur.execute("SELECT caption_template, presence_template, token_map FROM images WHERE image_id=%s", (image_id,))
    r = cur.fetchone()
    cap = (r["caption_template"] or "").replace(old, new)
    pres = (r["presence_template"] or "").replace(old, new)
    tmap = r["token_map"]
    for e in tmap:
        if face_ids is None or e["face_id"] in face_ids:
            e["person_id"] = person_id
    cur.execute("UPDATE images SET caption_template=%s, presence_template=%s, token_map=%s WHERE image_id=%s",
                (cap, pres, json.dumps(tmap), image_id))


# ---------- operations ----------
def rename(person_id: int, new_name: str | None) -> None:
    name = sanitize_name(new_name) if new_name else None
    with conn() as c, c.cursor() as cur:
        cur.execute("UPDATE people SET display_name=%s WHERE person_id=%s", (name or None, person_id))


def assign_faces(face_ids: list[int], person_id: int, by: str = "manual") -> None:
    """Assign pending faces to a person and rewrite their {{f:ID}} tokens."""
    affected: dict[int, list[int]] = {}
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT face_id, image_id FROM faces WHERE face_id = ANY(%s)", (face_ids,))
        rows = cur.fetchall()
        cur.execute("UPDATE faces SET person_id=%s, assigned_by=%s WHERE face_id = ANY(%s)",
                    (person_id, by, face_ids))
        for r in rows:
            _replace_token(cur, r["image_id"], f"{{{{f:{r['face_id']}}}}}", f"{{{{p:{person_id}}}}}",
                           {r["face_id"]}, person_id)
        for image_id in {r["image_id"] for r in rows}:
            affected[image_id] = _sync_image(cur, image_id)
    client().set_payload("faces", payload={"person_id": person_id}, points=face_ids)
    _push_payloads(affected)


def merge(keep: int, drop: int) -> None:
    if keep == drop:
        return
    affected: dict[int, list[int]] = {}
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT person_id, display_name, merged_into FROM people WHERE person_id IN (%s,%s)", (keep, drop))
        ppl = {r["person_id"]: r for r in cur.fetchall()}
        if any(p["merged_into"] for p in ppl.values()):
            raise ValueError("Both people must be active (not already merged)")
        if not ppl[keep]["display_name"] and ppl[drop]["display_name"]:
            cur.execute("UPDATE people SET display_name=%s WHERE person_id=%s", (ppl[drop]["display_name"], keep))

        cur.execute("SELECT DISTINCT image_id FROM faces WHERE person_id=%s", (drop,))
        images = [r["image_id"] for r in cur.fetchall()]
        cur.execute("UPDATE faces SET person_id=%s, assigned_by='manual' WHERE person_id=%s", (keep, drop))
        cur.execute("UPDATE people SET merged_into=%s WHERE person_id=%s", (keep, drop))
        for img in images:
            _replace_token(cur, img, f"{{{{p:{drop}}}}}", f"{{{{p:{keep}}}}}")
            cur.execute("SELECT token_map FROM images WHERE image_id=%s", (img,))
            tmap = cur.fetchone()["token_map"]
            for e in tmap:
                if e["person_id"] == drop:
                    e["person_id"] = keep
            cur.execute("UPDATE images SET token_map=%s WHERE image_id=%s", (json.dumps(tmap), img))
            affected[img] = _sync_image(cur, img)

    client().set_payload("faces", payload={"person_id": keep},
                         points=models.Filter(must=[models.FieldCondition(
                             key="person_id", match=models.MatchValue(value=drop))]))
    _push_payloads(affected)


def split(person_id: int, face_ids: list[int]) -> int:
    """Move selected faces of `person_id` into a brand-new person."""
    new_id = create_person()
    affected: dict[int, list[int]] = {}
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT face_id, image_id FROM faces WHERE face_id = ANY(%s) AND person_id=%s",
                    (face_ids, person_id))
        moving = cur.fetchall()
        cur.execute("UPDATE faces SET person_id=%s, assigned_by='manual' WHERE face_id = ANY(%s)",
                    (new_id, [r["face_id"] for r in moving]))
        for r in moving:
            cur.execute("SELECT 1 FROM faces WHERE image_id=%s AND person_id=%s LIMIT 1",
                        (r["image_id"], person_id))
            still_there = cur.fetchone() is not None
            if still_there:        # same person has another face in this image: token is ambiguous
                cur.execute("UPDATE images SET needs_recaption=TRUE WHERE image_id=%s", (r["image_id"],))
            else:
                _replace_token(cur, r["image_id"], f"{{{{p:{person_id}}}}}", f"{{{{p:{new_id}}}}}",
                               {r["face_id"]}, new_id)
        for img in {r["image_id"] for r in moving}:
            affected[img] = _sync_image(cur, img)
    client().set_payload("faces", payload={"person_id": new_id}, points=[r["face_id"] for r in moving])
    _push_payloads(affected)
    return new_id


def delete_person(person_id: int) -> None:
    """Right-to-erasure: removes biometric data (faces + vectors) for this person."""
    affected: dict[int, list[int]] = {}
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT face_id, image_id FROM faces WHERE person_id=%s", (person_id,))
        rows = cur.fetchall()
        face_ids = [r["face_id"] for r in rows]
        for img in {r["image_id"] for r in rows}:
            _replace_token(cur, img, f"{{{{p:{person_id}}}}}", "a person")
        cur.execute("DELETE FROM faces WHERE person_id=%s", (person_id,))
        cur.execute("UPDATE people SET deleted=TRUE, display_name=NULL WHERE person_id=%s", (person_id,))
        for img in {r["image_id"] for r in rows}:
            affected[img] = _sync_image(cur, img)
    if face_ids:
        client().delete("faces", points_selector=models.PointIdsList(points=face_ids))
    _push_payloads(affected)
```

**Re-captioning flagged images:** a small CLI command (`python -m app.cli recaption`) finds `needs_recaption = TRUE`, re-runs the `caption → build_templates → index` portion for those images, and clears the flag. This is the only case in which a VLM call follows a people-management action.

---

## 14. Feature 4: Evaluation, tracing and optimisation

### 14.1 What to measure

| Layer | Metric | Target (MVP) |
|---|---|---|
| Face clustering | Pairwise precision, recall, B-cubed F1, ARI | precision ≥ 0.98, recall ≥ 0.85 |
| Captions | LLM-judge scores (accuracy, completeness, hallucination, label use); label-validation failure rate | mean ≥ 4/5; failure rate < 3% |
| Retrieval | Recall@5, Recall@10, MRR, nDCG@10 | Recall@10 ≥ 0.80 |
| Latency | p50 / p95 end-to-end query | p95 < 3 s (without rewrite loop) |
| Ingestion | seconds per image, cost per 1k images | track, then optimise |

### 14.2 Build the eval sets first

**Faces:** label ~150 faces from ~10 people in `eval/face_labels.csv`:

```csv
face_id,true_person
1,aryan
2,aryan
3,riya
```

**Queries:** 50 queries in `eval/queries.jsonl` with hand-verified ground truth:

```json
{"query": "Aryan swimming", "expected_image_ids": [12, 45, 88], "type": "person+activity"}
{"query": "photos of Riya", "expected_image_ids": [3, 4, 9, 22], "type": "person-only"}
{"query": "Aryan and Riya on the beach", "expected_image_ids": [31, 77], "type": "multi-person"}
{"query": "sunset pictures", "expected_image_ids": [5, 6, 19], "type": "scene-only"}
```

To bootstrap queries faster, run prompt **P9** over captions, then **manually verify** the expected IDs. Never trust LLM-generated ground truth without checking.

### 14.3 `eval/metrics.py`

```python
import itertools
import numpy as np
from sklearn.metrics import adjusted_rand_score


def pairwise_pr(true_labels, pred_labels):
    tp = fp = fn = 0
    for (t1, p1), (t2, p2) in itertools.combinations(zip(true_labels, pred_labels), 2):
        same_t, same_p = t1 == t2, p1 == p2
        tp += same_t and same_p
        fp += (not same_t) and same_p
        fn += same_t and (not same_p)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    return prec, rec


def ari(true_labels, pred_labels):
    return adjusted_rand_score(true_labels, pred_labels)


def recall_at_k(ranked, expected, k):
    return len(set(ranked[:k]) & set(expected)) / max(1, len(expected))


def mrr(ranked, expected):
    for i, r in enumerate(ranked, 1):
        if r in expected:
            return 1 / i
    return 0.0


def ndcg_at_k(ranked, expected, k):
    dcg = sum(1 / np.log2(i + 2) for i, r in enumerate(ranked[:k]) if r in expected)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(expected))))
    return dcg / idcg if idcg else 0.0
```

### 14.4 `eval/run_eval.py` (outline)

```python
import json, time, statistics
from app.graphs.query_graph import build_query_graph
from eval.metrics import recall_at_k, mrr, ndcg_at_k

graph = build_query_graph()
rows, lat = [], []
for i, line in enumerate(open("eval/queries.jsonl")):
    q = json.loads(line)
    t0 = time.time()
    out = graph.invoke({"question": q["query"]}, {"configurable": {"thread_id": f"eval-{i}"}})
    lat.append(time.time() - t0)
    ranked = [r["image_id"] for r in out["answer"]["results"]]
    rows.append((recall_at_k(ranked, q["expected_image_ids"], 10),
                 mrr(ranked, q["expected_image_ids"]),
                 ndcg_at_k(ranked, q["expected_image_ids"], 10)))

print("Recall@10", statistics.mean(r[0] for r in rows))
print("MRR      ", statistics.mean(r[1] for r in rows))
print("nDCG@10  ", statistics.mean(r[2] for r in rows))
print("p50 / p95 latency", statistics.median(lat), sorted(lat)[int(0.95 * len(lat)) - 1])
```

Ambiguous-name queries pause on `interrupt()`; exclude them from automated runs or pass a fixed `Command(resume=...)`.

### 14.5 LangSmith tracing

Setting `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` in `.env` traces every LangChain/LangGraph run automatically. Use it to:
- see which node dominates latency
- inspect parse failures (wrong `person_names`)
- compare prompt versions on the same dataset (upload `queries.jsonl` as a LangSmith dataset and run experiments)

### 14.6 Caption judge (prompt P8)

```python
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from app.models import get_llm
from app.captioner import to_b64_jpeg
from app import prompts
import cv2

class JudgeScores(BaseModel):
    accuracy: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    hallucination: int = Field(ge=1, le=5)
    label_use: int = Field(ge=1, le=5)
    rationale: str

def judge(image_path: str, rendered_caption: str) -> JudgeScores:
    b64 = to_b64_jpeg(cv2.imread(image_path))
    llm = get_llm().with_structured_output(JudgeScores)
    return llm.invoke([
        SystemMessage(content=prompts.CAPTION_JUDGE_SYSTEM),
        HumanMessage(content=[
            {"type": "text", "text": f"Caption: {rendered_caption}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]),
    ])
```

Use a **different model** than the captioner as judge when possible, and spot-check 20 judgments by hand. LLM judges are biased toward fluent text.

### 14.7 Optimisation playbook

| Problem | Experiment | Measure with |
|---|---|---|
| Two people merged in one cluster | Raise `ASSIGN_THRESHOLD`; raise `MIN_BLUR_VAR`; cluster only high-quality faces | Pairwise precision |
| One person split in many clusters | Lower `CENTROID_MATCH_THRESHOLD`; merge-suggestion UI using centroid similarity | Pairwise recall |
| Wrong activities in captions | Tighten prompt P2; compare 2 VLMs; add a second pass on low-confidence images | Caption judge |
| Poor retrieval | Compare activity-only vs CLIP-only vs RRF; try a cross-encoder reranker; tune `limit` before grading | Recall@10, MRR |
| Slow queries | Cache embeddings for repeated queries; lower prefetch limit; skip grading for person-only queries (already done); smaller grading model | p95 latency |
| High cost | Local VLM for bulk captioning, hosted LLM for parse/grade; cache by SHA-256; downscale images to 768 px | Cost per 1k images |
| Qdrant slow at scale | Enable scalar quantisation; tune HNSW `m` / `ef_construct` | p95 latency, recall |

Change **one variable at a time** and record results in a table. This table is your "optimisation" story for portfolio and interviews.

---

## 15. API and UI

### 15.1 `app/api.py`

```python
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langgraph.types import Command

from .db import conn
from .qdrant_store import ensure_collections
from .tokens import load_people, display
from . import people_service as ps
from .clustering import run_cluster_job
from .graphs.ingest_graph import build_ingest_graph
from .graphs.query_graph import build_query_graph

app = FastAPI(title="Photo Memory")
ingest_graph = build_ingest_graph()
query_graph = build_query_graph()


@app.on_event("startup")
def _startup():
    ensure_collections()


class SearchIn(BaseModel):
    question: str
    thread_id: str | None = None

class ResumeIn(BaseModel):
    thread_id: str
    person_id: int

class RenameIn(BaseModel):
    name: str | None

class MergeIn(BaseModel):
    keep: int
    drop: int

class SplitIn(BaseModel):
    face_ids: list[int]


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    import tempfile, shutil, os
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
    ingest_graph.invoke({"path": tmp.name})
    return {"ok": True}


@app.post("/cluster/run")
def cluster():
    return run_cluster_job()


@app.get("/people")
def people():
    ppl = load_people()
    with conn() as c, c.cursor() as cur:
        cur.execute("""SELECT p.person_id, count(DISTINCT f.image_id) AS photos,
                              min(f.face_id) AS cover_face_id
                       FROM people p JOIN faces f ON f.person_id=p.person_id
                       WHERE p.merged_into IS NULL AND NOT p.deleted
                       GROUP BY p.person_id ORDER BY photos DESC""")
        rows = cur.fetchall()
    return [{"person_id": r["person_id"], "label": display(r["person_id"], ppl),
             "named": bool(ppl[r["person_id"]]["display_name"]),
             "photos": r["photos"], "cover_face_id": r["cover_face_id"]} for r in rows]


@app.patch("/people/{pid}")
def rename(pid: int, body: RenameIn):
    ps.rename(pid, body.name)
    return {"ok": True}

@app.post("/people/merge")
def merge(body: MergeIn):
    try:
        ps.merge(body.keep, body.drop)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}

@app.post("/people/{pid}/split")
def split(pid: int, body: SplitIn):
    return {"new_person_id": ps.split(pid, body.face_ids)}

@app.delete("/people/{pid}")
def delete(pid: int):
    ps.delete_person(pid)
    return {"ok": True}


def _pack(out, thread_id):
    if "__interrupt__" in out:
        return {"thread_id": thread_id, "needs_choice": out["__interrupt__"][0].value}
    return {"thread_id": thread_id, "answer": out["answer"]}

@app.post("/search")
def search(body: SearchIn):
    tid = body.thread_id or str(uuid.uuid4())
    out = query_graph.invoke({"question": body.question}, {"configurable": {"thread_id": tid}})
    return _pack(out, tid)

@app.post("/search/resume")
def resume(body: ResumeIn):
    out = query_graph.invoke(Command(resume=body.person_id), {"configurable": {"thread_id": body.thread_id}})
    return _pack(out, body.thread_id)


@app.get("/images/{image_id}/file")
def image_file(image_id: int):
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT path FROM images WHERE image_id=%s", (image_id,))
        r = cur.fetchone()
    if not r:
        raise HTTPException(404)
    return FileResponse(r["path"])
```

Also add `GET /faces/{face_id}/crop` (crop the stored bbox with OpenCV and return a JPEG) for people thumbnails, and `GET /faces/unassigned` for the "unassigned faces" screen.

### 15.2 `app/cli.py`

```python
import sys
from .qdrant_store import ensure_collections
from .graphs.ingest_graph import ingest_folder
from .clustering import run_cluster_job

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "init":
        ensure_collections(); print("collections ready")
    elif cmd == "ingest":
        print(ingest_folder(sys.argv[2]))
    elif cmd == "cluster":
        print(run_cluster_job())
```

### 15.3 Streamlit screens (`ui/streamlit_app.py`)

| Tab | Features |
|---|---|
| **Upload** | multi-file uploader to `/ingest`, button to run `/cluster/run` |
| **People** | grid of face thumbnails with label ("User 1" or real name), photo counts, text box to rename, "Merge into…" select, split tool |
| **Search** | chat input to `/search`; if `needs_choice`, render radio buttons then call `/search/resume`; results grid with rendered captions, people, and confidence badge |
| **Library** | all photos with caption and presence text |
| **Eval** | latest metrics from `eval/results.json` plus a link to LangSmith |

Minimal search tab:

```python
import requests, streamlit as st
API = "http://localhost:8000"

q = st.chat_input("Ask about your photos, e.g. 'Aryan swimming'")
if q:
    r = requests.post(f"{API}/search", json={"question": q}).json()
    st.session_state["last"] = r

r = st.session_state.get("last")
if r and "needs_choice" in r:
    nc = r["needs_choice"]
    pick = st.radio(f"Which '{nc['name']}'?", nc["options"], format_func=lambda o: o["label"])
    if st.button("Confirm"):
        r = requests.post(f"{API}/search/resume",
                          json={"thread_id": r["thread_id"], "person_id": pick["person_id"]}).json()
        st.session_state["last"] = r
        st.rerun()
elif r:
    ans = r["answer"]
    st.write(ans["message"])
    cols = st.columns(3)
    for i, item in enumerate(ans["results"]):
        with cols[i % 3]:
            st.image(f"{API}/images/{item['image_id']}/file")
            st.text(item["caption"])          # st.text avoids markdown injection from names
            st.caption(item["people"])
```

---

## 16. Testing

`tests/test_tokens.py`:

```python
from app.tokens import render, presence_template, sanitize_name, summarise_group

PEOPLE = {
    1: {"person_id": 1, "display_name": None,   "merged_into": None, "deleted": False},
    7: {"person_id": 7, "display_name": None,   "merged_into": None, "deleted": False},
    9: {"person_id": 9, "display_name": "Riya", "merged_into": 7,    "deleted": False},
}

def test_default_labels():
    assert render("{{p:1}} and {{p:7}} are eating pizza.", PEOPLE) == "User 1 and User 7 are eating pizza."

def test_rename_is_reflected():
    PEOPLE[1]["display_name"] = "Aryan"
    assert render("{{p:1}} is swimming.", PEOPLE) == "Aryan is swimming."
    PEOPLE[1]["display_name"] = None

def test_merge_redirect():
    assert render("{{p:9}} waves.", PEOPLE) == "User 7 waves."

def test_pending_face():
    assert render("{{f:88}} smiles.", PEOPLE) == "an unidentified person smiles."

def test_name_cannot_inject_token():
    assert "{{" not in sanitize_name("{{p:3}} <script>")

def test_presence():
    assert presence_template(["{{p:1}}", "{{p:7}}"]) == "{{p:1}} and {{p:7}} are present in this image."

def test_big_group():
    assert summarise_group(["A", "B", "C", "D", "E", "F"]) == "A, B, C and 3 others"
```

**Integration checklist** (run with a 20-photo fixture set):

| # | Scenario | Expected |
|---|---|---|
| 1 | Ingest same file twice | second run is skipped (SHA-256) |
| 2 | Group photo, 5 people | each quality face has a label; caption readable |
| 3 | Blurry or tiny face | no identity; caption says "a person" |
| 4 | Rename User 1 → Aryan | captions and search update instantly; no re-index |
| 5 | Search "Aryan swimming" | only photos with Aryan, ranked by activity match |
| 6 | Search a name that doesn't exist | friendly message plus suggestions |
| 7 | Two people named Aryan | disambiguation interrupt, then results |
| 8 | Merge two clusters | both IDs resolve to one person; no duplicate results |
| 9 | Split a cluster | only chosen faces move; ambiguous images flagged |
| 10 | Delete person | faces and vectors removed; captions say "a person" |
| 11 | Name `{{p:3}}` or `'; DROP TABLE people;--` | rendered harmlessly; no SQL error |
| 12 | VLM returns an invalid label ("P9") | retry, then fallback caption |
| 13 | "Aryan **and** Riya together" | `all-of` filter; only photos with both |
| 14 | "Aryan **or** Riya" | `any-of` filter |
| 15 | "photos of Aryan last month" | date filter applied, no semantic query |

---

## 17. Privacy and ethics

Face embeddings are **biometric data**. Treat them accordingly.

- Use your own photos, or consented/public datasets, for the portfolio build.
- Keep everything local-first. If using a hosted VLM/LLM, tell users that images are sent to that provider; offer the fully local (Ollama) configuration.
- Add a **Delete person** feature (done in Chapter 13) and a **Delete everything** command.
- Never expose the app publicly without authentication.
- Document your legal basis: India's DPDP Act 2023 (consent, purpose limitation, erasure) and GDPR if serving Europe.
- Disclose known limitations: face recognition accuracy varies across demographics and image quality; never use it for surveillance, identification of strangers, or decisions about people.
- Never auto-name anyone from the internet or from social profiles. Names come only from the user.

Add a `PRIVACY.md` to the repo; interviewers notice this.

---

## 18. Milestones

| Week | Deliverable | Acceptance criteria |
|---|---|---|
| 0 | Environment, Docker, schema, 300 test photos, labelled eval faces | `docker compose up` works; `ensure_collections` passes |
| 1 | Face detection, embeddings, quality gate, Qdrant `faces` (notebook then module) | Faces detected on 95%+ of clear faces |
| 2 | HDBSCAN `cluster_job`, incremental assignment, clustering metrics | Pairwise precision ≥ 0.98 on eval set |
| 3 | Captioning with labelled boxes, validation and retry (P1–P3) | < 3% fallback captions; labels always valid |
| 4 | Templates, tokens, `render`, rename/merge/split/delete, unit tests | All tests in Chapter 16 pass |
| 5 | `photos` index with two vectors, hybrid RRF retrieval, person filters | "Aryan swimming" works end to end |
| 6 | Query graph: parse, resolve, interrupt, grade, rewrite loop (P4–P6) | Disambiguation and unknown-name flows work |
| 7 | Evaluation harness, LangSmith tracing, optimisation experiments table | Metrics report generated; 3 documented experiments |
| 8 | FastAPI, Streamlit, README, architecture diagram, demo video, PRIVACY.md | Fresh clone to running demo in under 15 minutes |

---

## 19. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `pip install insightface` fails on Windows | Missing C++ build tools | Install Microsoft C++ Build Tools, retry |
| InsightFace runs on CPU only | `onnxruntime` instead of `onnxruntime-gpu`, or CUDA mismatch | Install `onnxruntime-gpu` matching your CUDA version |
| Everyone lands in one cluster | Threshold too low, low-quality faces included | Raise `ASSIGN_THRESHOLD`, `MIN_BLUR_VAR`, `MIN_FACE_PX` |
| Same person in many clusters | Threshold too high, pose/age variation | Lower `CENTROID_MATCH_THRESHOLD`; add merge suggestions |
| VLM keeps inventing names | Prompt drift or weak model | Keep temperature 0; check validation; try a stronger VLM |
| Caption mentions "P3" but only P1, P2 exist | Model hallucinated label | Validation retries it; check `CAPTION_RETRIES` |
| Qdrant `Wrong input: Not existing vector name` | Query uses wrong named vector | Use `using="activity"` or `using="clip"` exactly |
| Filter on `person_ids` is slow | Missing payload index | Run `ensure_collections()` (creates indexes) |
| Rename not showing in UI | UI cached old captions | Render at request time; don't cache rendered text |
| `interrupt` resume does nothing | No checkpointer or different `thread_id` | Compile with a checkpointer; reuse the same `thread_id` |
| `Bad EXIF` / missing dates | Photos stripped of metadata | Fallback uses file modified time (already implemented) |
| Ollama VLM gives poor JSON | Small model | Use a bigger VLM, or set `format="json"`; keep retries |
| Out of GPU memory | CLIP + InsightFace + local VLM together | Run VLM on a hosted API, or process sequentially |

---

## 20. Learning and scope

### 20.1 Learning checklist (tick as you build)

- [ ] **Embeddings:** cosine similarity, normalisation, ANN, HNSW (`m`, `ef`)
- [ ] **Face recognition:** detection vs alignment vs recognition, ArcFace margin loss, thresholds and ROC
- [ ] **Clustering:** DBSCAN vs HDBSCAN vs agglomerative, evaluating without labels (silhouette) and with labels (ARI, B-cubed)
- [ ] **CLIP / contrastive learning:** joint image-text space, zero-shot retrieval
- [ ] **Vision-language models:** prompting with images, structured outputs, hallucination modes
- [ ] **RAG:** hybrid search, metadata filtering, RRF, reranking, query rewriting, grading, RAG evaluation
- [ ] **LangChain:** `init_chat_model`, structured output, message formats, retrievers
- [ ] **LangGraph:** state, reducers, nodes, conditional edges, fan-out/join, `interrupt`, checkpointers, subgraphs
- [ ] **Databases:** relational modelling for identity (merge/split), transactions, migrations
- [ ] **Evaluation:** Recall@k, MRR, nDCG, LLM-as-judge pitfalls, LangSmith datasets and experiments
- [ ] **Engineering:** FastAPI, async jobs, Docker, pytest, CI
- [ ] **Responsible AI:** biometric privacy, demographic bias, consent, data deletion

### 20.2 Applications and product scope

| Segment | Use |
|---|---|
| Families | Private, self-hosted Google Photos alternative |
| Photographers / event studios | "All photos of this guest" for delivery and selection |
| Journalism / media archives | Natural-language search over large asset libraries |
| Education / corporate events | Album building and attendee-specific galleries (with consent) |
| Accessibility | Automatic alt-text for images |
| E-commerce | Auto-captioning and visual search for catalogues (drop the face module) |

**Positioning:** privacy-first, self-hosted, conversational photo search with customisable pipelines.

**Possible monetisation:** SaaS for studios, on-prem licence, or API (captioning plus face grouping).

### 20.3 Future roadmap

1. **Merge suggestions:** show cluster pairs with centroid similarity above a threshold
2. **Hindi and multilingual queries** (parse step translates to English search phrase)
3. **Auto-albums:** "Trip to Goa 2025" from date, GPS and scene clustering
4. **Video support:** sample frames, track faces across frames
5. **Voice search**
6. **Multi-user shared albums** with per-user permissions
7. **Agentic actions:** "Make an album of Aryan's swimming photos"
8. **Cross-encoder reranker** and **ColPali-style** visual document retrieval for scans
9. **Mobile app** (React Native) with on-device face detection

### 20.4 Definition of done

- Fresh clone to a working demo in under 15 minutes using this setup guide
- Renaming a person updates every caption and search without re-indexing
- Evaluation report with metrics and at least three documented optimisation experiments
- README with architecture diagram, demo GIF, privacy statement and known limitations