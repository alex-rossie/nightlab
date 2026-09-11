#!/usr/bin/env python3
"""PreToolUse hook for Bash: no Claude session applies `needs-gpu`, approves the
`gpu` deployment, or force-pushes.

A human applies that label in the GitHub UI after reading the diff. This hook
blocks the obvious shell forms of adding it (`gh pr edit --add-label`,
`gh pr create --label`, `gh api` bodies), of approving a deployment, and of
force-pushing, from every session that loads this repo's settings: the nightly
routines and the reviewer's `/lab` session alike. Removing the label, listing
PRs by label, and merely mentioning it stay allowed.

This is a guard against a routine following stale instructions or a naive
prompt injection. It matches text, so a label passed through a shell variable,
a script file, or a non-Bash tool is not caught. It is not a security
boundary; see docs/runner-setup.md. Test it with
`uv run pytest -q -k gpu_label_hook`; a shell one-liner containing a blocked
form is itself blocked.
"""

import json
import re
import sys

LABEL = "needs-gpu"
# A flag value: optional spaces or '=', an optional quote, then label characters
# (letters, commas, dashes, quotes, spaces) up to the label, never crossing into
# another '--flag'.
VALUE = r"""[\s=]*["']?(?:(?!--)[\w,\-"' ])*?"""
ADD_LABEL = re.compile(r"--add-label" + VALUE + LABEL)
CREATE_LABEL = re.compile(r"(?:--label|(?<!\S)-l)" + VALUE + LABEL)
REMOVE_LABEL = re.compile(r"""--remove-label[\s=]*["']?""" + LABEL)
API_BODY = re.compile(r"labels(?:\[\]=|[^\]\n]{0,120})" + LABEL)
FORCE_PUSH = re.compile(r"git\s+push\b[^|;&\n]*(?:\s--force\b|\s-f\b|\s--force-with-lease|\s\+\S)")
DEPLOY = ("pending_deployments", "reviewdeployments")


def normalize(command: str) -> str:
    return re.sub(r"\\\n", " ", command).lower()


def blocked(command: str) -> str | None:
    cmd = normalize(command)
    if any(d in cmd for d in DEPLOY):
        return "approving a GitHub deployment is a human action"
    if FORCE_PUSH.search(cmd):
        return "force pushes are not for Claude sessions; push a new commit or open a PR"
    if LABEL not in cmd:
        return None
    stripped = REMOVE_LABEL.sub("", cmd)
    adds = (ADD_LABEL.search(stripped) or API_BODY.search(stripped)
            or ("create" in stripped and CREATE_LABEL.search(stripped)))
    if adds:
        return f"{LABEL} is applied by a human in the GitHub UI after reading the diff"
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    command = str(payload.get("tool_input", {}).get("command", ""))
    reason = blocked(command)
    if reason:
        print(f"blocked by .claude/hooks/guard_gpu_label.py: {reason} "
              "(docs/runner-setup.md)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
