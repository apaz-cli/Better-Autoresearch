import copy
import difflib
import re
from typing import Any

import config
from models import (
    DECIDE_TOOL, EDIT_TOOL,
    Message, ModelConfig, Tool,
    llm_call,
    PROPOSE_IDEA_CONFIG, IMPLEMENT_IDEA_CONFIG, DIAGNOSE_CRASH_CONFIG, SHOULD_KEEP_CONFIG, COMMIT_MESSAGE_CONFIG,
)

SYSTEM_PROMPT = """\
You may NOT add new package dependencies beyond what's in pyproject.toml.

Simplicity criterion: all else equal, prefer simpler code. A tiny gain that \
adds ugly complexity is not worth it. Removing code and getting equal or \
better results is a great outcome.

VRAM is a constraint. Some increase is acceptable for meaningful val_bpb \
gains, but it should not blow up dramatically. Specifically, peak memory usage must stay \
below 80 GB.\
"""


def ask(messages: list[Message], model: ModelConfig, tools: list[Tool] | None = None, tool_choice: dict | None = None) -> Any:
    return llm_call(messages, model=model, system=SYSTEM_PROMPT, tools=tools, tool_choice=tool_choice, max_agent_turns=config.MAX_AGENT_TURNS)


def quick(prompt: str, model: ModelConfig) -> str:
    return llm_call([{"role": "user", "content": prompt}], model=model, max_tokens=config.QUICK_MAX_TOKENS)


def edit_train(messages: list[Message], model: ModelConfig) -> str:
    return llm_call(messages, model=model, system=SYSTEM_PROMPT, tools=[copy.copy(EDIT_TOOL)], max_agent_turns=config.MAX_AGENT_TURNS)


def propose_idea(train_py: str, results: str) -> str:
    """Return a natural-language description of the next experiment to try (no code).

    Reads the full experiment history and current train.py to propose the single
    most promising change. The returned string is passed directly to implement_idea.
    """
    return ask([{"role": "user", "content": f"""\
You are an autonomous ML researcher. Your goal is to minimize val_bpb \
(validation bits per byte) on a fixed 5-minute training run.

val_bpb = (sum of per-token cross-entropy in nats) / (log(2) * sum of UTF-8 \
bytes of target tokens). Special tokens with byte-length 0 are excluded. \
It is vocabulary-size-independent, so architectural changes that affect model \
size or tokenization are fairly compared. Lower is better.

You edit train.py — the only file you may change. Everything in it is fair \
game: model architecture, optimizer, hyperparameters, batch size, etc.

What is the single most promising change to try next? Think step by step \
about what's likely to improve val_bpb given the results so far. \
Describe your plan in a short paragraph — no code yet.

Experiment history (TSV):
{results}

Current train.py:
{train_py}\
"""}], model=PROPOSE_IDEA_CONFIG)


def implement_idea(train_py: str, idea: str) -> None:
    """Edit train.py in place to implement idea. Side effect only — no return value.

    idea should be the natural-language description from propose_idea. The model
    makes one or more str_replace edits until it signals end_turn.
    """
    edit_train([{"role": "user", "content": f"""\
Use str_replace_based_edit_tool to make edits. The file path is train.py (not repo/train.py). Implement this change to train.py:

{idea}

Current train.py:
{train_py}\
"""}], model=IMPLEMENT_IDEA_CONFIG)


def diagnose_crash(idea: str, original_train_py: str, crashed_train_py: str, log: str) -> str | None:
    """Attempt to fix a crashed train.py in place via the edit tool.

    Returns the give-up reason (str) if the model decides the idea is
    fundamentally broken and cannot be fixed. Returns None if the model
    attempted a fix (train.py may have been modified) or produced no output.

    Note: a None return does not guarantee train.py was changed — the caller
    should diff before committing.
    """
    diff_lines = list(difflib.unified_diff(
        original_train_py.splitlines(keepends=True),
        crashed_train_py.splitlines(keepends=True),
        fromfile="a/train.py", tofile="b/train.py",
    ))
    diff_str = "".join(diff_lines) if diff_lines else "(no changes from original)"
    text = edit_train([{"role": "user", "content": f"""\
The training run crashed. The idea being tested:
{idea}

Log:
{log}

Diff introduced by this experiment (original → crashed train.py):
{diff_str}

Full crashed train.py:
{crashed_train_py}

If this is a simple fixable bug, use str_replace_based_edit_tool to fix train.py (file path is train.py, not repo/train.py).
If the idea is fundamentally broken (OOM with no easy fix, etc.), do NOT use the tool \
— just reply with GIVE_UP: <reason>\
"""}], model=DIAGNOSE_CRASH_CONFIG)
    m = re.search(r"GIVE_UP\s*:\s*(.+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def should_keep(idea: str, old_train_py: str, new_train_py: str, val_bpb: float, memory_gb: float, best: float, results: str) -> bool:
    """Ask the model whether to keep or discard an experiment. Returns True to keep.

    Uses DECIDE_TOOL (terminal) to force a structured response: the model must
    provide a justification and a keep/discard decision. The justification is
    logged; the decision is returned as a bool.
    """
    diff = "".join(difflib.unified_diff(
        old_train_py.splitlines(keepends=True),
        new_train_py.splitlines(keepends=True),
        fromfile="a/train.py",
        tofile="b/train.py",
    )) or "(no changes)"
    result = ask(
        [{"role": "user", "content": f"""\
We just ran an ML experiment. Results:
  val_bpb:   {val_bpb:.6f}  (previous best: {best:.6f}, delta: {val_bpb - best:+.6f})
  memory_gb: {memory_gb:.1f}

Experiment history (TSV):
{results}

The idea:
{idea}

Diff:
{diff}

New train.py:
{new_train_py}

Should we keep this change? Weigh the val_bpb improvement against added complexity and memory usage. \
A small gain that adds ugly or fragile code is not worth keeping. \
A simplification that breaks even on val_bpb is worth keeping.\
"""}],
        model=SHOULD_KEEP_CONFIG,
        tools=[DECIDE_TOOL],
        tool_choice={"type": "tool", "name": "decide"},
    )
    return result["decision"] == "keep"


def commit_message(idea: str, diff: str | None = None) -> str:
    ctx = f"Proposed idea: {idea}\n\nActual diff:\n{diff}" if diff else idea
    return quick(f"""\
Need a one-line git commit message for this ML experiment. \
Output just the message, no command or quotes.

{ctx}\
""", model=COMMIT_MESSAGE_CONFIG)
