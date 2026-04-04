OPUS = "claude-opus-4-6"
HAIKU = "claude-haiku-4-5-20251001"
API_STYLE = "anthropic"  # "anthropic" or "openai"
BASE_URL = None          # None = use provider default

BRANCH = "autoresearch/val_bpb-improvement"
LOG_FILE = "program.log"
ERRORS_FILE = "errors.log"
RESULTS_FILE = "results.tsv"

EXPERIMENT_HOURS = 2
LLM_KEEP_DISCARD = True  # if True, ask the LLM for keep/discard decisions; otherwise use raw val_bpb improvement

TRAIN_TIMEOUT = 660
MAX_CRASH_FIXES = 2
MAX_AGENT_TURNS = 20
MAX_TOKENS = 64000
QUICK_MAX_TOKENS = 512
BACKOFF_INITIAL = 1.0
BACKOFF_MAX = 300.0

