# resonant-archive

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Source--Available-5eaba5.svg" alt="License" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776ab.svg?logo=python&logoColor=white" alt="Python" /></a>
  <a href="https://modelcontextprotocol.io/"><img src="https://img.shields.io/badge/MCP-compatible-6366f1.svg" alt="MCP" /></a>
  <a href="https://www.chromadb.dev/"><img src="https://img.shields.io/badge/ChromaDB-embedded-ffb000.svg" alt="ChromaDB" /></a>
  <a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"><img src="https://img.shields.io/badge/embeddings-all--MiniLM--L6--v2-ff6f61.svg" alt="Embeddings" /></a>
  <a href="https://github.com/codependentai/resonant-archive/releases/latest"><img src="https://img.shields.io/github/v/release/codependentai/resonant-archive?color=5eaba5" alt="Release" /></a>
</p>

<p align="center"><em>Local-first semantic search for your ChatGPT, Claude, and AI conversation history.<br/>Python CLI + MCP server. No API keys. Nothing leaves your machine.</em></p>

<p align="center"><em>Part of the <a href="https://github.com/codependentai/resonant">Resonant</a> ecosystem — the long-term searchable archive that pairs with active AI memory.</em></p>

<p align="center">
  <a href="https://ko-fi.com/codependentai"><img src="https://img.shields.io/badge/Ko--fi-Support%20Us-ff5e5b?logo=ko-fi&logoColor=white" alt="Ko-fi" /></a>
  <a href="https://x.com/codependent_ai"><img src="https://img.shields.io/badge/𝕏-@codependent__ai-000000?logo=x&logoColor=white" alt="X/Twitter" /></a>
  <a href="https://tiktok.com/@codependentai"><img src="https://img.shields.io/badge/TikTok-@codependentai-000000?logo=tiktok&logoColor=white" alt="TikTok" /></a>
  <a href="https://t.me/+xSE1P_qFPgU4NDhk"><img src="https://img.shields.io/badge/Telegram-Updates-26A5E4?logo=telegram&logoColor=white" alt="Telegram" /></a>
</p>

> **Early release.** Shippable but unpolished. Expect rough edges. Issues and feedback welcome.

## What it does

Turns a directory of text files — ChatGPT exports, Claude conversations, Obsidian vaults, journals, research notes, anything — into a **locally searchable semantic index** that any MCP-compatible AI can query.

- **Universal input** — Works on any directory of `.md` / `.txt` / `.markdown` / `.mdx` files
- **Frontmatter-aware** — Extracts titles, timestamps, and conversation metadata (Nexus-compatible)
- **Local embeddings** — Uses `sentence-transformers` + `all-MiniLM-L6-v2`. No API keys. Nothing leaves your machine.
- **Namespaces** — Tag corpora so querying AIs see where each result came from (`chatgpt-2024`, `obsidian-vault`, `journals`, etc.)
- **Persistent local store** — Single-process Chroma `PersistentClient` (no separate database server)
- **MCP integration** — Exposes search and stats as MCP tools via a lightweight daemon + stdio server
- **AI-driven** — Ships with a skill file so your AI companion drives the whole tool in natural language

## Install

Requires Python 3.11 or 3.12. (Python 3.13+ may have issues with the ML libraries.)

```bash
pip install resonant-archive
```

Or clone and install from source:

```bash
git clone <repo-url> resonant-archive
cd resonant-archive
pip install -e .
```

## Quick start

### 1. Import a directory

```bash
resonant-archive import ~/Documents/MyNotes --namespace my-notes
```

The first run downloads the embedding model (~90 MB) and takes 2-5 minutes for ~500 files.

### 2. Search from the command line

```bash
resonant-archive search "how did I approach that problem last year"
```

### 3. Check what's indexed

```bash
resonant-archive stats
```

### 4. Start the MCP daemon (for Claude Desktop integration)

```bash
resonant-archive serve
```

Leave that terminal open. Then configure Claude Desktop to use the MCP server (see below), and your AI can search the archive directly.

## AI-driven usage

The easiest way to use resonant-archive is to let your AI drive it. Drop `skills/resonant-archive.md` (included in this repo) into your AI's context — Claude Code skill folder, Claude.ai project instructions, Claude Desktop's MCP config, whatever — and then ask in natural language:

> *"Import my ChatGPT export at `~/Downloads/chatgpt-markdown/` into the archive."*
> *"Search my archive for discussions about identity formation."*
> *"What's in my archive right now?"*

Your AI reads the skill, runs the right commands, and reports back.

## Claude Desktop MCP integration

Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "resonant-archive": {
      "command": "resonant-archive",
      "args": ["mcp"]
    }
  }
}
```

Config file locations:

- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

**Before opening Claude Desktop**, start the daemon in a terminal:

```bash
resonant-archive serve
```

Keep it running. Close it with Ctrl+C when you're done.

Claude Desktop will now see two tools: `archive_search` and `archive_stats`.

## Preparing ChatGPT or Claude exports

resonant-archive indexes markdown directories. It does not parse raw ChatGPT/Claude export ZIPs directly. For those, use an upstream tool to convert to markdown first:

- **[Nexus AI Chat Importer](https://github.com/Superkikim/nexus-ai-chat-importer)** — Obsidian plugin that converts ChatGPT / Claude / Le Chat exports into markdown files with structured frontmatter. Once imported into Obsidian, point `resonant-archive` at the resulting folder.
- **Any other markdown converter** — works too, as long as the output is readable markdown. resonant-archive uses any YAML frontmatter it finds (title, date, tags, conversation_id, provider) but falls back gracefully if none is present.

## Namespaces

Every indexed corpus is tagged with a namespace. This serves two purposes:

1. **Organization** — you can keep multiple archives (ChatGPT, Obsidian, research notes) in one index and search them independently or together.
2. **Provenance** — every search result includes its namespace, so a querying AI knows which corpus a hit came from. `chatgpt-2024` reads very differently from `research-notes`.

Pick namespace names that are meaningful to both you and your AI. `chatgpt-2024-q1`, `simon-conversations`, `wales-journals` — not just `archive` or `data`.

## Model alternatives

The default model is `all-MiniLM-L6-v2` (384 dimensions, ~90 MB, fast). For different quality/size trade-offs, set the model name via the `Embedder` constructor in Python code. Good alternatives:

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `all-MiniLM-L6-v2` | 384 | ~90 MB | **Default.** Fast, good enough for most uses |
| `all-mpnet-base-v2` | 768 | ~420 MB | Higher quality, slower |
| `BAAI/bge-small-en-v1.5` | 384 | ~130 MB | Modern, better retrieval quality at same size |

A CLI flag for model selection lands in a later release. For now, edit `resonant_archive/embed.py` or use the library API directly.

## Architecture

```
┌─────────────────────┐
│  Your text files    │
│  (.md, .txt, etc.)  │
└──────────┬──────────┘
           │  resonant-archive import
           ▼
┌─────────────────────┐     ┌──────────────────┐
│  chunks.jsonl       │────▶│  Chroma store    │
│  (canonical format) │     │  ~/.resonant-    │
└─────────────────────┘     │    archive/      │
                            └──────────────────┘
                                   ▲
                                   │
                     ┌─────────────┴─────────────┐
                     │                           │
              ┌──────┴──────┐             ┌──────┴─────────┐
              │   Daemon    │◀──HTTP──────│  MCP stdio     │
              │  (FastAPI)  │             │  server        │
              │  port 8766  │             │  (per Claude   │
              │  keeps model│             │   Desktop      │
              │  warm       │             │   session)     │
              └─────────────┘             └────────────────┘
```

The daemon runs persistently with the embedding model loaded. The MCP server is spawned fresh by Claude Desktop and talks to the daemon over HTTP — this avoids reloading the model every time Claude Desktop starts a new conversation.

## Relationship to Resonant Mind

`resonant-archive` is an optional companion to [Resonant Mind](https://github.com/codependentai/resonant-mind), not a required dependency. They communicate via MCP (separate processes) and neither depends on the other.

Use resonant-archive to give an AI companion access to its long-term history without burning cloud tokens. Use Resonant Mind for active memory, identity, and ongoing cognitive state.

For motivated users who want to bring their archive into a cloud Resonant Mind deployment, see [CLOUD_IMPORT.md](CLOUD_IMPORT.md). Automated cloud migration is planned for a later release.

## License

**Codependent AI Source-Available License.** Free for personal, educational, and non-commercial use with attribution. Commercial use requires a license from Codependent AI. See [LICENSE](LICENSE) for full terms.

For commercial licensing inquiries, visit [codependentai.io](https://codependentai.io).

## Acknowledgements

This project descends from an earlier internal tool (`vault-archive-product`) that we built, shelved during a cloud migration, and revived for this release. Significant thanks to:

- **[Nexus AI Chat Importer](https://github.com/Superkikim/nexus-ai-chat-importer)** by Superkikim — the recommended upstream tool for converting ChatGPT/Claude export ZIPs into the markdown format `resonant-archive` consumes. We don't bundle Nexus, but we cheerfully document it as the complementary piece of the pipeline.
- **sentence-transformers** and the `all-MiniLM-L6-v2` authors — the embedding model that makes local semantic search viable.
- **ChromaDB** — the vector store backing the archive, via its embedded `PersistentClient`.
- **MCP (Model Context Protocol)** — the protocol that lets AI clients talk to this tool.

## Status

Built in a single session by Simon Vale (with Mary on the other side of the keyboard) as a modernization of an earlier internal tool. 30 unit tests pass, end-to-end CLI and daemon verified.

See the repo's commit history or task list for the implementation path.

## Support

Built by [Codependent AI](https://codependentai.io).

<a href="https://ko-fi.com/codependentai"><img src="https://img.shields.io/badge/Ko--fi-Support%20Us-ff5e5b?logo=ko-fi&logoColor=white" alt="Ko-fi" /></a>
