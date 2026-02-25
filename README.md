# Whisper2Obsidian

> **Automated voice memo → Obsidian note pipeline using LangGraph, local Whisper transcription, and Groq LLM analysis.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-optimised-black)](https://developer.apple.com/metal/)

---

## Overview

```
iPhone Voice Record Pro → .m4a + metadata sidecar (Google Drive)
        ↓
[watcher_node]           – detects new files by mtime, skips already-processed
        ↓
[transcription_node]     – mlx-whisper large-v3-turbo (Neural Engine + Metal GPU)
        ↓
[vault_indexer_node]     – loads existing tags & links from SQLite vault index
        ↓
[analysis_node]          – Groq Llama-3.3-70b analysis → structured JSON
                           (rate-limited: RPM / TPM / RPD guards)
        ↓
[note_writer_node]       – Jinja2 template (by category) → Obsidian Markdown
        ↓
[file_writer_node]       – writes .md to vault inbox, updates SQLite index
```

---

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.11+**
- **ffmpeg** installed (`brew install ffmpeg`)
- **Groq API key** (free tier is sufficient)
- Google Drive folder mounted locally (no GDrive API needed)

---

## Setup

```bash
# 1. Clone
git clone https://github.com/Michal0lszewski/Whisper2Obsidian.git
cd Whisper2Obsidian

# 2. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install (mlx-whisper will pull the model on first run)
pip install -e ".[dev]"

# 4. Configure
cp .env.example .env
# Edit .env – set AUDIO_FOLDER, VAULT_PATH, GROQ_API_KEY
nano .env

# 5. Index your existing vault (recommended before first run)
w2o-harvest

# 6. Run
whisper2obsidian          # daemon mode (polls every 60s)
whisper2obsidian --once   # process one memo and exit
whisper2obsidian --show-rate-usage   # show Groq token/request counters
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `AUDIO_FOLDER` | _required_ | Path to Google Drive voice memo folder |
| `VAULT_PATH` | _required_ | Obsidian vault root |
| `INBOX_FOLDER` | `00 Inbox` | Sub-folder for new notes |
| `GROQ_API_KEY` | _required_ | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `WHISPER_MODEL` | `mlx-community/whisper-large-v3-turbo` | MLX Whisper model |
| `GROQ_RPM_LIMIT` | `28` | Max requests/min (free cap: 30) |
| `GROQ_TPM_LIMIT` | `11000` | Max tokens/min (free cap: 12 000) |
| `GROQ_RPD_LIMIT` | `950` | Max requests/day (free cap: 1 000) |
| `SHOW_RATE_USAGE` | `false` | Print Groq usage table after each run |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Note Templates (by category)

Voice Record Pro category → Jinja2 template:

| Category | Template | Features |
|---|---|---|
| `meeting` | `meeting.md.j2` | Action items, decisions, IMPORTANT callout |
| `idea` | `idea.md.j2` | Markmind `markmap` codeblock, TIP callout |
| `research` / `lecture` | `research.md.j2` | Concept markmap, status Dataview field |
| _any other_ | `default.md.j2` | Key points, action items, NOTE callout |

All templates include:
- YAML frontmatter with `tags`, `date`, `category`, Dataview inline fields
- `[[wiki-links]]` to related notes
- Optional Mermaid diagram block
- Collapsible raw transcript

---

## Obsidian Plugins Used

| Plugin | Usage |
|---|---|
| **Dataview** | Frontmatter + inline `key:: value` fields |
| **Markmind / Markmap** | Mind-map codeblocks in idea/research notes |
| **Mermaid** | Flowchart diagrams (built into Obsidian) |

---

## Rate Limiting

The `GroqRateLimiter` service guards every Groq API call using a sliding-window algorithm:

- **RPM** and **TPM** tracked via 60-second deque window
- **RPD** tracked via daily counter with midnight reset
- `await_capacity(estimated_tokens)` sleeps automatically if limits would be exceeded
- Configurable via `.env` so you can adjust for a paid Groq tier

---

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Re-index vault
w2o-harvest
```

---

## Project Structure

```
src/whisper2obsidian/
├── config.py              # pydantic-settings Config
├── state.py               # LangGraph TypedDict state
├── graph.py               # compile_graph()
├── main.py                # CLI entry point
├── nodes/
│   ├── watcher.py
│   ├── transcription.py
│   ├── vault_indexer.py
│   ├── analysis.py
│   ├── note_writer.py
│   └── file_writer.py
├── services/
│   ├── groq_rate_limiter.py
│   ├── metadata_parser.py
│   └── vault_index.py
├── scripts/
│   └── vault_harvest.py
└── templates/
    ├── default.md.j2
    ├── meeting.md.j2
    ├── idea.md.j2
    └── research.md.j2
```
