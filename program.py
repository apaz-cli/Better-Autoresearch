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
from pathlib import Path

import anthropic

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
LLM_KEEP_DISCARD = False  # if True, ask the LLM to weigh complexity vs improvement instead of pure val_bpb

SYSTEM_PROMPT = """\
You may NOT add new package dependencies beyond what's in pyproject.toml.

Simplicity criterion: all else equal, prefer simpler code. A tiny gain that \
adds ugly complexity is not worth it. Removing code and getting equal or \
better results is a great outcome.

VRAM is a soft constraint. Some increase is acceptable for meaningful val_bpb \
gains, but it should not blow up dramatically.\
"""

###########
# Results #
###########

def print_log(*args):
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

def log_result(commit, val_bpb, memory_gb, status, description):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")

def best_kept_bpb():
    if not Path(RESULTS_FILE).exists():
        return float("inf")
    kept = [float(l.split("\t")[1]) for l in Path(RESULTS_FILE).read_text().splitlines() if "\tkeep\t" in l]
    return min(kept, default=float("inf"))


###########
# PROMPTS #
###########

client = anthropic.Anthropic()

def ask(messages, model=OPUS, strip_code=False, use_system=True, max_tokens=65536):
    kwargs = dict(model=model, max_tokens=max_tokens, messages=messages)
    if use_system:
        kwargs["system"] = SYSTEM_PROMPT
    text = client.messages.create(**kwargs).content[0].text
    if strip_code:
        text = re.sub(r"^```\w*\n?", "", text.strip())
        text = re.sub(r"\n?```\s*$", "", text).strip()
    return text

def propose_idea(train_py, results):
    return ask([{"role": "user", "content": f"""\
Experiment history (TSV):
{results}

Current train.py:
{train_py}

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
Describe your plan in a short paragraph — no code yet."""}], use_system=True)


def implement_idea(train_py: str, idea: str) -> str:
    return ask([{"role": "user", "content": f"""\
Implement this change to train.py:
{idea}

Current train.py:
{train_py}

Reply with only the complete modified train.py, no explanation. \
It is very important that you only respond with train.py, as it will be written directly to a file."""}], strip_code=True, use_system=True)


def diagnose_crash(original_train_py, crashed_train_py, log) -> tuple[str | None, None | str]:
    """Returns fixed train.py string and None, Otherwise returns None and the reason it gave up."""
    diff_lines = list(difflib.unified_diff(
        original_train_py.splitlines(keepends=True),
        crashed_train_py.splitlines(keepends=True),
        fromfile="a/train.py", tofile="b/train.py",
    ))
    diff_str = "".join(diff_lines) if diff_lines else "(no changes from original)"
    text = ask([{"role": "user", "content": f"""\
The training run crashed. Log:
{log}

Diff introduced by this experiment (original → crashed train.py):
{diff_str}

Full crashed train.py:
{crashed_train_py}

If this is a simple fixable bug, reply with only the complete fixed train.py.
If the idea is fundamentally broken (OOM with no easy fix, etc.), reply with: GIVE_UP: <reason>"""}], strip_code=True, use_system=True)
    if text.strip().upper().startswith("GIVE_UP"):
        reason = text[text.index(":")+1:].strip() if ":" in text else text
        return None, reason
    return text.strip(), None


def should_keep(idea, old_train_py, new_train_py, val_bpb, best):
    """Ask the LLM whether to keep or discard. Returns True to keep."""
    text = ask([{"role": "user", "content": f"""\
We just ran an ML experiment. Results:
  val_bpb:  {val_bpb:.6f}  (previous best: {best:.6f}, delta: {val_bpb - best:+.6f})

The change we made:
{idea}

Old train.py:
{old_train_py}

New train.py:
{new_train_py}

Should we keep this change? Weigh the val_bpb improvement against any added complexity and memory usage. \
A small gain that adds ugly or fragile code is not worth keeping. \
A simplification that breaks even on val_bpb is worth keeping. \
Think it through, then end your response with either KEEP or DISCARD."""}], use_system=True)
    return text.strip().upper().endswith("KEEP")


def print_diff(old: str, new: str, label: str = "train.py"):
    diff = list(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{label}", tofile=f"b/{label}",
    ))
    if not diff:
        print("\033[33m(no changes)\033[0m")
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


def commit_message(idea):
    return ask([{"role": "user", "content": f"One-line git commit message for this ML experiment (no quotes):\n{idea}"}], model=HAIKU, use_system=False)

def change_branch(branch):
    """Switch to branch if it exists, create it if not."""
    exists = subprocess.run(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"]).returncode == 0
    if exists:
        subprocess.run(["git", "checkout", branch], check=True)
        print_log(f"Switched to branch: {branch}")
    else:
        subprocess.run(["git", "checkout", "-b", branch], check=True)
        print_log(f"Created branch: {branch}")

##################
# STARTUP CHECKS #
##################

def startup_checks():
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
    name  = subprocess.run(["git", "config", "user.name"],  capture_output=True, text=True).stdout.strip()
    email = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True).stdout.strip()
    if not name or not email:
        print("Error: git user.name and user.email must be configured.")
        sys.exit(1)

    # Data preparation
    subprocess.run(["uv", "run", "prepare.py"], check=True)


#############
# MAIN LOOP #
#############

def run_training() -> tuple[str, bool]:
    """Run train.py. Returns (log: str, success: bool)."""
    try:
        r = subprocess.run(["uv", "run", "train.py"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=TRAIN_TIMEOUT)
        return r.stdout, r.returncode == 0
    except subprocess.TimeoutExpired as e:
        return (e.stdout or "") + "\n[TIMEOUT]", False

def run_experiment(baseline, original_train_py, description) -> tuple[float, float] | None:
    """Run training with crash-fix retries. Returns (val_bpb, memory_gb) or None on give-up."""
    for attempt in range(1 + MAX_CRASH_FIXES):
        print_log(f"[train] Running... (attempt {attempt + 1}/{1 + MAX_CRASH_FIXES})")
        log, ok = run_training()
        m_bpb  = re.search(r"^val_bpb:\s+([\d.]+)",     log, re.MULTILINE)
        m_vram = re.search(r"^peak_vram_mb:\s+([\d.]+)", log, re.MULTILINE)
        val_bpb   = float(m_bpb.group(1))          if m_bpb  else None
        memory_gb = float(m_vram.group(1)) / 1024  if m_vram else 0.0

        if ok and val_bpb is not None:
            return val_bpb, memory_gb

        with open(ERRORS_FILE, "a") as f:
            f.write(f"\n=== attempt {attempt + 1} ({description}) ===\n{log}\n")

        if attempt == MAX_CRASH_FIXES:
            print_log("[crash] Max fix attempts reached. Giving up.")
        else:
            print_log("[crash] Asking LLM to diagnose...")
            crashed_train_py = Path("train.py").read_text()
            fixed_train, giveup_reason = diagnose_crash(original_train_py, crashed_train_py, log)
            if fixed_train:
                print_diff(crashed_train_py, fixed_train)
                fix_desc = commit_message(f"Fix crash in: {description}")
                print_log(f"[crash] {fix_desc}")
                Path("train.py").write_text(fixed_train)
                subprocess.run(["git", "add", "train.py"], check=True)
                subprocess.run(["git", "commit", "-m", fix_desc], check=True)
                continue
            else:
                print_log(f"[crash] LLM gives up: {giveup_reason}")

        subprocess.run(["git", "reset", "--hard", baseline], check=True)
        return None

def main():
    startup_checks()

    # Switch to a new (or old) branch to experiment on
    change_branch(BRANCH)

    # Read or create results file
    if not Path(RESULTS_FILE).exists():
        Path(RESULTS_FILE).write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
    best = best_kept_bpb()

    # Run baseline before any experimentation
    if best == float("inf"):
        print_log("No results yet. Running baseline (unmodified train.py)...")
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        log, ok = run_training()
        m_bpb  = re.search(r"^val_bpb:\s+([\d.]+)",     log, re.MULTILINE)
        m_vram = re.search(r"^peak_vram_mb:\s+([\d.]+)", log, re.MULTILINE)
        if not ok or not m_bpb:
            with open(ERRORS_FILE, "a") as f:
                f.write(f"\n=== baseline ===\n{log}\n")
            print_log(f"Baseline run failed. See {ERRORS_FILE} for details.")
            return
        best      = float(m_bpb.group(1))
        memory_gb = float(m_vram.group(1)) / 1024 if m_vram else 0.0
        log_result(commit, best, memory_gb, "keep", "baseline")
        print_log(f"[keep] Baseline: {best:.6f}")
    else:
        print_log(f"Starting. Best val_bpb so far: {best:.6f}")

    # Autoresearch loop
    while True:
        # Grab current state
        baseline = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        train_py = Path("train.py").read_text()
        results  = Path(RESULTS_FILE).read_text()

        # Propose an idea
        print_log("\n[propose] Thinking about what to try...")
        idea = propose_idea(train_py, results)
        print_log(f"[propose] {idea}")

        # Implement the idea
        print_log("[implement] Writing code...")
        new_train_py = implement_idea(train_py, idea)
        if not new_train_py:
            print_log("[implement] No train.py returned. Retrying.")
            continue
        print_diff(train_py, new_train_py)

        # Write the file and make a git commit
        description = commit_message(idea)
        print_log(f"[implement] {description}")
        Path("train.py").write_text(new_train_py)
        subprocess.run(["git", "add", "train.py"], check=True)
        subprocess.run(["git", "commit", "-m", description], check=True)
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()

        # Run the commit, get results or log crash
        result = run_experiment(baseline, train_py, description)
        if result is None:
            log_result(commit, 0.0, 0.0, "crash", description)
            continue
        val_bpb, memory_gb = result

        # Keep or discard the commit based on results
        if LLM_KEEP_DISCARD:
            print_log("[judge] Asking LLM whether to keep...")
            keep = should_keep(idea, train_py, Path("train.py").read_text(), val_bpb, best)
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
