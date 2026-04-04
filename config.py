import os

########################
# propose_idea() Model #
########################
PROPOSE_IDEA_MODEL    = "claude-opus-4-6"
PROPOSE_IDEA_STYLE    = "anthropic"
PROPOSE_IDEA_BASE_URL = None
PROPOSE_IDEA_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")

##########################
# implement_idea() Model #
##########################
IMPLEMENT_IDEA_MODEL    = "claude-opus-4-6"
IMPLEMENT_IDEA_STYLE    = "anthropic"
IMPLEMENT_IDEA_BASE_URL = None
IMPLEMENT_IDEA_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")

##########################
# diagnose_crash() Model #
##########################
DIAGNOSE_CRASH_MODEL    = "claude-opus-4-6"
DIAGNOSE_CRASH_STYLE    = "anthropic"
DIAGNOSE_CRASH_BASE_URL = None
DIAGNOSE_CRASH_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")

#######################
# should_keep() Model #
#######################
SHOULD_KEEP_MODEL    = "claude-opus-4-6"
SHOULD_KEEP_STYLE    = "anthropic"
SHOULD_KEEP_BASE_URL = None
SHOULD_KEEP_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")

##########################
# commit_message() Model #
##########################
COMMIT_MESSAGE_MODEL    = "claude-haiku-4-5-20251001"
COMMIT_MESSAGE_STYLE    = "anthropic"
COMMIT_MESSAGE_BASE_URL = None
COMMIT_MESSAGE_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")

###########
# Logging #
###########
BRANCH = "autoresearch/val_bpb-improvement"
LOG_FILE = "program.log"
ERRORS_FILE = "errors.log"
RESULTS_FILE = "results.tsv"

#####################
# Experiment Config #
#####################
EXPERIMENT_HOURS = 2
LLM_KEEP_DISCARD = True  # if True, ask the LLM for keep/discard decisions; otherwise use raw val_bpb improvement


#################
# Misc Settings #
#################
TRAIN_TIMEOUT = 660
MAX_CRASH_FIXES = 2
MAX_AGENT_TURNS = 20
MAX_TOKENS = 64000
QUICK_MAX_TOKENS = 512
BACKOFF_INITIAL = 1.0
BACKOFF_MAX = 300.0
