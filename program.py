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
from pathlib import Path

from config import BRANCH, ERRORS_FILE, EXPERIMENT_HOURS, LLM_KEEP_DISCARD, MAX_CRASH_FIXES, RESULTS_FILE, TRAIN_TIMEOUT
from log import diff_log, log_result, print_log
from prompts import commit_message, diagnose_crash, implement_idea, propose_idea, should_keep


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
    end_time = time.time() + EXPERIMENT_HOURS * 3600 if EXPERIMENT_HOURS is not None else float("inf")
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
        print_log(f"Starting. Best val_bpb so far: {best:.6f}. Time limit: {EXPERIMENT_HOURS:.1f}h")

    # Autoresearch loop
    while time.time() < end_time:
        # Grab current state
        baseline = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        train_py = Path("train.py").read_text()
        results = Path(RESULTS_FILE).read_text()

        # Propose an idea
        print_log("\n[propose] Thinking about what to try...")
        idea = propose_idea(train_py, results)

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
        if memory_gb > 80:
            print_log(f"[discard] {val_bpb:.6f} exceeds 80 GB VRAM limit ({memory_gb:.1f} GB)")
            subprocess.run(["git", "reset", "--hard", baseline], check=True)
            log_result(commit, val_bpb, memory_gb, "discard", description)
            continue
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
