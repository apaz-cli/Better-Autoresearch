import difflib
import re

from config import LOG_FILE, RESULTS_FILE


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


def _print_colored(msg: str, color: str) -> None:
    print(f"{color}{msg}\033[0m")
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def _format_messages(messages: list) -> str:
    parts = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block["type"] == "text":
                    parts.append(block["text"])
    return "\n\n".join(parts)


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
