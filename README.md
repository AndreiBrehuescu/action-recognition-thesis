# Claude Code Orchestrator

A tiered multi-agent setup for Claude Code. Type `/orchestrate <task>` and Claude
decomposes the task, routes each piece to the **cheapest model that can handle it**, runs
independent pieces in parallel as subagents, and synthesizes one result.

The point is cost discipline: you only pay flagship-tier rates for the parts that genuinely
need deep reasoning. Everything else lands on cheaper, faster models.

> Heads up: this doesn't use *fewer* tokens than a single big call — it usually uses more
> overall. What drops is **cost per task**, because most of those tokens go to cheap models.

---

## What's in here

```
.claude/
├── commands/
│   └── orchestrate.md       # the /orchestrate command (the "brain")
└── agents/
    ├── cheap-worker.md      # Haiku  — formatting, extraction, simple edits
    ├── mid-worker.md        # Sonnet — drafting, standard code, refactors
    └── flagship-worker.md   # Opus   — hard reasoning, reserved for costly mistakes
```

The command is the orchestrator. The three worker agents are what make the tiering real —
in Claude Code, the per-tier model is pinned in each subagent's frontmatter, not in the
command itself.

---

## Setup

### Option A — one project (recommended to start)

Keep the `.claude/` folder in the root of the repo you're working in. The command and
agents are then available in that project, and you can commit them so teammates get them too.

```
your-project/
├── .claude/          ← this folder
└── ... your code ...
```

### Option B — everywhere

To make `/orchestrate` available in every project, put the same `commands/` and `agents/`
folders under your home directory instead:

- **Windows:** `C:\Users\<you>\.claude\commands\` and `...\.claude\agents\`
- **macOS / Linux:** `~/.claude/commands/` and `~/.claude/agents/`

Project-level files win if both exist with the same name.

---

## Usage

In a Claude Code session:

```
/orchestrate Refactor the auth module and write unit tests for it
/orchestrate Summarize these meeting notes and draft a reply to the client
/orchestrate Audit this script for security issues, then fix what you find
```

Everything after `/orchestrate` becomes the task. That's it.

---

## How it works

1. **Triage.** If the task is trivial (a quick edit, a lookup, a one-paragraph answer),
   the orchestrator just does it — no subagents, no coordination overhead.
2. **Plan.** Otherwise it decomposes the task into the smallest set of independent
   subtasks and assigns each the cheapest sufficient tier.
3. **Delegate.** Independent subtasks run in parallel as subagents, each routed to
   `cheap-worker`, `mid-worker`, or `flagship-worker`. Each subagent gets only the context
   it needs (its brief plus any upstream results) — tight context is part of the savings,
   since every subagent also gets its own clean window.
4. **Synthesize.** The orchestrator stitches the outputs into one coherent answer. If a
   single subtask covered the whole task, it returns that directly instead of adding a
   redundant pass.

Workers can also escalate: if a "cheap" task turns out to hide real reasoning, the worker
flags it so the orchestrator can re-route it up a tier instead of guessing.

---

## Tuning

**Model tiers** — each worker pins its own model in frontmatter. Swap any of these for a
different cost/capability split:

| Tier            | File                  | Model (default)              |
|-----------------|-----------------------|------------------------------|
| cheap           | `cheap-worker.md`     | `claude-haiku-4-5-20251001`  |
| mid             | `mid-worker.md`       | `claude-sonnet-4-6`          |
| flagship        | `flagship-worker.md`  | `claude-opus-4-8`            |

**Orchestrator model** — set in `commands/orchestrate.md` frontmatter, currently
`claude-sonnet-4-6`. Bump it to `claude-opus-4-8` if your tasks need heavier planning, or
drop it to Haiku if decomposition is usually simple.

**Routing rules** — what counts as "flagship-worthy" lives in the `description` fields of
the worker files and in the Phase 1 guidance inside `orchestrate.md`. Edit those to change
how aggressively work gets pushed to cheaper tiers, or to add a fourth tier.

---

## Requirements & notes

- Claude Code with custom slash commands and subagents (current versions support both).
- Model strings above are current as of June 2026. If you set this up much later, confirm
  them against the Anthropic docs and update the frontmatter.
- This is a prompt-and-config setup — there's no code to install and nothing to build.
