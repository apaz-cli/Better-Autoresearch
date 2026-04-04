#!/usr/bin/env python3
"""
program.py — Autonomous research loop for autoresearch.

The LLM proposes ideas and implements them; this script handles the rest.

Usage:
    uv run program.py
"""

import difflib
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import anthropic
from anthropic.types import MessageParam, ToolUseBlock

#############
# Configure #
#############

OPUS = "claude-opus-4-6"
HAIKU = "claude-haiku-4-5-20251001"
BRANCH = "autoresearch/val_bpb-improvement"
LOG_FILE = "program.log"
ERRORS_FILE = "errors.log"
RESULTS_FILE = "results.tsv"
TRAIN_TIMEOUT = 660
MAX_CRASH_FIXES = 2
MAX_AGENT_TURNS = 20
LLM_KEEP_DISCARD = False  # if True, ask the LLM for keep/discard decisions; otherwise use raw val_bpb improvement

SYSTEM_PROMPT = """\
You may NOT add new package dependencies beyond what's in pyproject.toml.

Simplicity criterion: all else equal, prefer simpler code. A tiny gain that \
adds ugly complexity is not worth it. Removing code and getting equal or \
better results is a great outcome.

VRAM is a constraint. Some increase is acceptable for meaningful val_bpb \
gains, but it should not blow up dramatically. Specifically, peak memory usage must stay \
below 80 GB.\
"""

###########
# Logging #
###########


def print_log(*args: object) -> None:
    msg = " ".join(str(a) for a in args)
    tag = re.search(r"\[(\w+)\]", msg)
    color = {
        "propose":   "\033[36m",  # cyan
        "implement": "\033[36m",  # cyan
        "train":     "\033[34m",  # blue
        "crash":     "\033[31m",  # red
        "keep":      "\033[32m",  # green
        "discard":   "\033[33m",  # yellow
        "judge":     "\033[35m",  # magenta
    }.get(tag.group(1) if tag else "", "")
    print(f"{color}{msg}\033[0m" if color else msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def diff_log(old: str, new: str, label: str = "train.py") -> None:
    diff = list(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{label}",
        tofile=f"b/{label}",
    ))
    if not diff:
        print("\033[33m(no changes)\033[0m")
        with open(LOG_FILE, "a") as f:
            f.write("(no changes)\n")
        return
    for line in diff:
        if line.startswith("@@"):
            print(f"\033[36m{line}\033[0m", end="")
        elif line.startswith("+"):
            print(f"\033[32m{line}\033[0m", end="")
        elif line.startswith("-"):
            print(f"\033[31m{line}\033[0m", end="")
        else:
            print(line, end="")
    with open(LOG_FILE, "a") as f:
        f.write("".join(diff))

def log_result(commit: str, val_bpb: float, memory_gb: float, status: str, description: str) -> None:
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")


###############
# LLM HELPERS #
###############

client = anthropic.Anthropic()


@dataclass
class Tool:
    spec: dict
    handler: Callable[[ToolUseBlock], Any]
    terminal: bool = False

    @property
    def name(self) -> str:
        return self.spec["name"]


def make_edit_tool() -> Tool:
    """Create a fresh str_replace_based_edit_tool with its own undo stack."""
    undo_stack: list[str] = []

    def handler(block: ToolUseBlock) -> str:
        cmd = block.input.get("command")
        path = block.input.get("path", "train.py")
        if os.path.realpath(path) != os.path.realpath("train.py"):
            return f"Error: only train.py may be edited (got {path!r})"
        if cmd == "create":
            return "Error: create is not supported"
        content = Path("train.py").read_text()
        if cmd == "view":
            lines = content.splitlines()
            vr = block.input.get("view_range")
            s, e = (vr[0] - 1, vr[1]) if vr else (0, len(lines))
            return "\n".join(f"{s + i + 1}\t{l}" for i, l in enumerate(lines[s:e]))
        elif cmd == "str_replace":
            old, new = block.input["old_str"], block.input["new_str"]
            count = content.count(old)
            if count != 1:
                return f"Error: old_str found {count} times (must be exactly 1)"
            undo_stack.append(content)
            Path("train.py").write_text(content.replace(old, new, 1))
            return "Edit applied."
        elif cmd == "insert":
            new_str = block.input["new_str"]
            lines = content.splitlines(keepends=True)
            lines.insert(
                block.input["insert_line"],
                new_str if new_str.endswith("\n") else new_str + "\n",
            )
            undo_stack.append(content)
            Path("train.py").write_text("".join(lines))
            return "Insert applied."
        elif cmd == "undo_edit":
            if not undo_stack:
                return "Error: nothing to undo"
            Path("train.py").write_text(undo_stack.pop())
            return "Undo applied."
        return f"Error: unknown command {cmd!r}"

    return Tool(
        spec={"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
        handler=handler,
    )


def _with_backoff(fn: Callable) -> Any:
    """Retry fn() with exponential backoff on 5xx server errors, capped at 5 minutes."""
    delay = 1.0
    while True:
        try:
            return fn()
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                print_log(f"[llm_call] Server error {e.status_code}, retrying in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 300.0)
            else:
                raise


def llm_call(messages: list[MessageParam], model: str, system: str | None = None, tools: list[Tool] | None = None, tool_choice: dict | None = None, max_tokens: int = 64000) -> Any:
    """Core LLM call. Two modes depending on whether tools are provided:

    Without tools: streams a single response and returns the text (str).

    With tools: runs an agentic loop — calls the API, dispatches tool_use blocks
    to each Tool's handler, appends results, and repeats until end_turn or a
    terminal tool fires. Non-terminal tool handlers return a str result that is
    fed back to the model. Terminal tool handlers return any value, which is
    immediately returned from this function (bypassing the normal str return).

    Returns:
        str — the model's final text response (no tools, or non-terminal tool loop)
        Any — the terminal Tool handler's return value, if one fired

    All API calls are retried with exponential backoff on 5xx errors.
    """
    kwargs: dict = dict(model=model, max_tokens=max_tokens, messages=list(messages))
    if system:
        kwargs["system"] = system
    if model != HAIKU:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": "medium"}
    if tools:
        tool_map = {t.name: t for t in tools}
        kwargs["tools"] = [t.spec for t in tools]
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        turn = 0
        while True:
            response = _with_backoff(lambda: client.messages.create(**kwargs))
            text = "".join(b.text for b in response.content if b.type == "text")

            if response.stop_reason == "end_turn":
                break
            elif response.stop_reason == "max_tokens":
                print_log(f"[llm_call] Max tokens reached. Time to panic. Final text:\n{text}")
                return text
            elif response.stop_reason == "refusal":
                print_log(f"[llm_call] Refusal. Final text:\n{text}")
                return text
            elif response.stop_reason == "model_context_window_exceeded":
                print_log(f"[llm_call] Context window exceeded. Final text:\n{text}")
                return text

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool = tool_map.get(block.name)
                if tool is None:
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": f"Error: unknown tool {block.name!r}"})
                    continue
                result = tool.handler(block)
                if tool.terminal:
                    return result
                print_log(f"[tool:{block.input.get('command', block.name)}] {result}")
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

            turn += 1
            if turn >= MAX_AGENT_TURNS:
                print_log(f"[llm_call] Max agent turns reached.")
                return text
            kwargs["messages"] += [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ]

    else:
        def _stream() -> str:
            with client.messages.stream(**kwargs) as stream:
                return stream.get_final_text()
        text = _with_backoff(_stream)

    print_log(f"[{model}] {text}")
    return text


def ask(messages: list[MessageParam], tools: list[Tool] | None = None, tool_choice: dict | None = None) -> Any:
    """Opus call with extended thinking and the research system prompt.

    Without tools: returns the model's text response (str).
    With a terminal tool: returns that tool's handler result (e.g. a dict for DECIDE_TOOL).
    With non-terminal tools: runs the agentic loop and returns final text (str).
    """
    return llm_call(messages, model=OPUS, system=SYSTEM_PROMPT, tools=tools, tool_choice=tool_choice)


def quick(prompt: str) -> str:
    """Haiku call for short-output tasks (commit messages, etc.). Returns str."""
    return llm_call([{"role": "user", "content": prompt}], model=HAIKU, max_tokens=512)


def edit_train(messages: list[MessageParam]) -> str:
    """Opus agentic loop that edits train.py in place via str_replace_based_edit_tool.

    The model may call the tool multiple times. Each call is executed immediately
    and the result fed back. Returns the model's final text response once it
    signals end_turn. Side effect: train.py may be modified on disk.
    """
    return llm_call(messages, model=OPUS, system=SYSTEM_PROMPT, tools=[make_edit_tool()])


DECIDE_TOOL = Tool(
    spec={
        "name": "decide",
        "description": "Record your keep/discard decision with reasoning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "justification": {
                    "type": "string",
                    "description": "Reasoning for the decision, considering val_bpb improvement, complexity, and memory.",
                },
                "decision": {
                    "type": "string",
                    "enum": ["keep", "discard"],
                },
            },
            "required": ["justification", "decision"],
        },
    },
    handler=lambda block: block.input,
    terminal=True,
)

###########
# PROMPTS #
###########


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
{train_py}"""
    }])


def implement_idea(train_py: str, idea: str) -> None:
    """Edit train.py in place to implement idea. Side effect only — no return value.

    idea should be the natural-language description from propose_idea. The model
    makes one or more str_replace edits until it signals end_turn.
    """
    edit_train([{"role": "user", "content": f"""\
Use str_replace_based_edit_tool to make edits. Implement this change to train.py:

{idea}

Current train.py:
{train_py}"""}])


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

If this is a simple fixable bug, use str_replace_based_edit_tool to fix train.py.
If the idea is fundamentally broken (OOM with no easy fix, etc.), do NOT use the tool \
— just reply with GIVE_UP: <reason>"""}])
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
A simplification that breaks even on val_bpb is worth keeping."""}],
        tools=[DECIDE_TOOL],
        tool_choice={"type": "tool", "name": "decide"},
    )
    justification = result.get("justification", "")
    decision = result["decision"]
    print_log(f"[judge] {justification}")
    return decision == "keep"


def commit_message(idea: str, diff: str | None = None) -> str:
    ctx = f"Proposed idea: {idea}\n\nActual diff:\n{diff}" if diff else idea
    return quick(f"""\
Need a one-line git commit message for this ML experiment. \
Output just the message, no command or quotes.

{ctx}\
""")


def change_branch(branch: str) -> None:
    """Switch to branch if it exists, create it if not."""
    exists = (subprocess.run(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"]).returncode == 0)
    if exists:
        subprocess.run(["git", "checkout", branch], check=True)
        print_log(f"Switched to branch: {branch}")
    else:
        subprocess.run(["git", "checkout", "-b", branch], check=True)
        print_log(f"Created branch: {branch}")


#############
# MAIN LOOP #
#############

def startup_checks() -> None:
    # API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    # Ensure GPU
    r = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True)
    if r.returncode != 0:
        print("Error: No GPU found (nvidia-smi failed).")
        sys.exit(1)

    # Stay out the way of other users on my shared cluster
    last_gpu = str(len(r.stdout.strip().splitlines()) - 1)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", last_gpu)

    # Git identity
    name = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True).stdout.strip()
    email = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True).stdout.strip()
    if not name or not email:
        print("Error: git user.name and user.email must be configured.")
        sys.exit(1)

    # Data preparation
    subprocess.run(["uv", "run", "prepare.py"], check=True)

def run_training() -> tuple[str, bool]:
    """Run train.py. Returns (log: str, success: bool)."""
    try:
        r = subprocess.run(
            ["uv", "run", "train.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=TRAIN_TIMEOUT,
        )
        return r.stdout, r.returncode == 0
    except subprocess.TimeoutExpired as e:
        return (e.stdout or b"").decode("utf-8") + "\n[TIMEOUT]", False


def run_experiment(original_train_py: str, description: str, idea: str, is_baseline: bool = False) -> tuple[float, float] | None:
    """Run training with crash-fix retries. Returns (val_bpb, memory_gb) or None on give-up."""
    for attempt in range(1 + MAX_CRASH_FIXES):

        # Run the training
        print_log(f"[train] Running... (attempt {attempt + 1}/{1 + MAX_CRASH_FIXES})")
        log, ok = run_training()

        # Read results from stdout and return them
        m_bpb = re.search(r"^val_bpb:\s+([\d.]+)", log, re.MULTILINE)
        m_vram = re.search(r"^peak_vram_mb:\s+([\d.]+)", log, re.MULTILINE)
        val_bpb = float(m_bpb.group(1)) if m_bpb else None
        memory_gb = float(m_vram.group(1)) / 1024 if m_vram else 0.0
        if ok and val_bpb is not None:
            return val_bpb, memory_gb

        # Training exited cleanly but didn't produce val_bpb — the code likely
        # removed or broke the output. Treat as a crash so the LLM can fix it,
        # but log the real situation clearly.
        if ok:
            print_log("[crash] Training exited cleanly but val_bpb was missing from output.")
            log += "\n[val_bpb missing from output — training completed but metric was not printed]"

        # Don't try to fix baselines, let a human deal with that.
        if is_baseline:
            print_log("[train] Baseline run failed. No point in retrying.")
            with open(ERRORS_FILE, "a") as f:
                f.write(f"\n=== baseline crash ===\n{log}\n")
            return None

        # If we crashed, attempt a fix and retry until we fix or give up or MAX_CRASH_FIXES.
        with open(ERRORS_FILE, "a") as f:
            f.write(f"\n=== attempt {attempt + 1} ({description}) ===\n{log}\n")
        if attempt == MAX_CRASH_FIXES:
            print_log("[crash] Max fix attempts reached. Giving up.")
        else:
            print_log("[crash] Asking LLM to diagnose...")
            crashed_train_py = Path("train.py").read_text()
            giveup_reason = diagnose_crash(idea, original_train_py, crashed_train_py, log)
            if giveup_reason is None:
                fixed_train_py = Path("train.py").read_text()
                if fixed_train_py == crashed_train_py:
                    print_log("[crash] LLM made no changes. Giving up.")
                else:
                    diff_log(crashed_train_py, fixed_train_py)
                    fix_desc = commit_message(f"Fix crash in: {description} attempt {attempt + 1}")
                    print_log(f"[crash] {fix_desc}")
                    subprocess.run(["git", "add", "train.py"], check=True)
                    subprocess.run(["git", "commit", "-m", fix_desc], check=True)
                    continue
            else:
                print_log(f"[crash] LLM gives up: {giveup_reason}")

        return None


def main() -> None:
    startup_checks()

    # Switch to a new (or old) branch to experiment on
    change_branch(BRANCH)

    # Read or create results file
    if not Path(RESULTS_FILE).exists():
        Path(RESULTS_FILE).write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
    kept = [float(l.split("\t")[1]) for l in Path(RESULTS_FILE).read_text().splitlines() if "\tkeep\t" in l]
    best = min(kept, default=float("inf"))

    # Run baseline before any experimentation
    if best == float("inf"):
        print_log("No results yet. Running baseline (unmodified train.py)...")
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        train_py = Path("train.py").read_text()
        result = run_experiment(train_py, "baseline", "baseline", is_baseline=True)
        if result is None:
            print_log("Baseline run failed. Exiting.")
            return
        best, memory_gb = result
        log_result(commit, best, memory_gb, "keep", "baseline")
        print_log(f"[keep] Baseline: {best:.6f}")
    else:
        print_log(f"Starting. Best val_bpb so far: {best:.6f}")

    # Autoresearch loop
    while True:
        # Grab current state
        baseline = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        train_py = Path("train.py").read_text()
        results = Path(RESULTS_FILE).read_text()

        # Propose an idea
        print_log("\n[propose] Thinking about what to try...")
        idea = propose_idea(train_py, results)
        print_log(f"[propose] {idea}")

        # Implement the idea
        print_log("[implement] Writing code...")
        implement_idea(train_py, idea)
        new_train_py = Path("train.py").read_text()
        if new_train_py == train_py:
            print_log("[implement] No changes made. Retrying.")
            continue
        diff_log(train_py, new_train_py)

        # Make a git commit
        impl_diff = "".join(difflib.unified_diff(
            train_py.splitlines(keepends=True),
            new_train_py.splitlines(keepends=True),
            fromfile="a/train.py", tofile="b/train.py",
        ))
        description = commit_message(idea, impl_diff)
        subprocess.run(["git", "add", "train.py"], check=True)
        subprocess.run(["git", "commit", "-m", description], check=True)
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()

        # Run the commit, get results or log crash
        result = run_experiment(train_py, description, idea)
        if result is None:
            log_result(commit, 0.0, 0.0, "crash", description)
            subprocess.run(["git", "reset", "--hard", baseline], check=True)
            continue

        # If crash fixes added commits on top, squash them back onto the original commit
        current_head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        if current_head != commit:
            subprocess.run(["git", "reset", "--soft", baseline], check=True)
            subprocess.run(["git", "commit", "-m", description], check=True)
            commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()

        # Keep or discard the commit based on results
        val_bpb, memory_gb = result
        if LLM_KEEP_DISCARD:
            print_log("[judge] Asking LLM whether to keep...")
            keep = should_keep(idea, train_py, Path("train.py").read_text(), val_bpb, memory_gb, best, results)
        else:
            keep = val_bpb < best
        if keep:
            best = min(best, val_bpb)
            print_log(f"[keep]    {val_bpb:.6f} (best: {best:.6f})")
            log_result(commit, val_bpb, memory_gb, "keep", description)
        else:
            print_log(f"[discard] {val_bpb:.6f} (best is {best:.6f})")
            subprocess.run(["git", "reset", "--hard", baseline], check=True)
            log_result(commit, val_bpb, memory_gb, "discard", description)


if __name__ == "__main__":
    main()
