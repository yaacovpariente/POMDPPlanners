#!/usr/bin/env sh
# Refuse a commit that credits an AI coding tool.
#
# A tool that commits for us tends to sign its work: a Co-authored-by trailer,
# a "Generated with" line, a session trailer, or its own name as author. None
# of that belongs in this repository's history. Three places are checked:
#   - the trailer block (Co-authored-by, Signed-off-by, Claude-Session, ...)
#   - a "Generated with", "Made by", "Built using" style line anywhere
#   - the author and committer identity
# Trailers come from git's own parser, so a subject such as
# "Docs: explain the codex-review agent" and prose that mentions a tool pass.
#
# Two ways to run it, sharing one rule:
#   scripts/check_no_ai_credit.sh <message-file>   # commit-msg hook: the
#       message git is about to use, plus the identity of the current commit
#   scripts/check_no_ai_credit.sh --range A..B      # CI: every commit in the
#       range, merges included, message and identity
set -eu

ai='claude|anthropic|codex|openai|chatgpt|gpt-?[0-9]|copilot|gemini|google-labs-jules|cursor(agent)?|devin|aider|qwen|windsurf|codeium|tabnine|codewhisperer|amazon[- ]q|sourcery|\[bot\]'

# Trailer keys that credit someone. A trailer with any other key counts only
# if the key itself names a tool (Claude-Session) or the value carries an
# email, so a lone "Note: the claude agent file is a symlink" last line,
# which git also parses as a trailer, passes.
credit_keys='^(co-authored-by|signed-off-by|reviewed-by|assisted-by|helped-by|suggested-by|authored-by|generated-by|reported-by|tested-by|acked-by):'

# <message on stdin>: the credit lines in it, if any.
credit_lines() {
    msg=$(cat)
    {
        printf '%s\n' "$msg" | git interpret-trailers --parse \
            | grep -iE "$credit_keys|^[^:]*($ai)[^:]*:|<[^>]*@[^>]*>" || true
        printf '%s\n' "$msg" | grep -E '(Generated|Made|Built|Created|Written) (with|by|using)' || true
    } | grep -iE "$ai" || true
}

status=0
report() { # <what> <lines>
    [ -z "$2" ] && return 0
    echo "$1" >&2
    echo "$2" | sed 's/^/    /' >&2
    status=1
}

case ${1-} in
--range)
    [ $# -eq 2 ] || { echo "usage: $0 --range A..B" >&2; exit 2; }
    # Resolve the range first: a failed substitution inside `for` would run
    # the loop zero times and report success.
    shas=$(git rev-list "$2") || exit 2
    for sha in $shas; do
        short=$(git rev-parse --short "$sha")
        report "$short: message credits an AI tool" \
            "$(git log -1 --format=%B "$sha" | credit_lines)"
        report "$short: author or committer is an AI tool" \
            "$(git log -1 --format='%an <%ae>%n%cn <%ce>' "$sha" | grep -iE "$ai" || true)"
    done
    ;;
"" | -*)
    echo "usage: $0 <message-file> | --range A..B" >&2
    exit 2
    ;;
*)
    # Drop comment lines and everything after the `git commit -v` scissors
    # line: a diff may legitimately contain these words (this file, for one).
    report "commit message credits an AI tool" \
        "$(sed -n '/^# -\{3,\} >8 -\{3,\}$/q; /^#/!p' "$1" | credit_lines)"
    report "author or committer is an AI tool" \
        "$({ git var GIT_AUTHOR_IDENT; git var GIT_COMMITTER_IDENT; } | grep -iE "$ai" || true)"
    ;;
esac

[ "$status" -eq 0 ] || {
    echo "" >&2
    echo "No Co-authored-by, Generated-with or session trailer for a tool, and no" >&2
    echo "tool as author or committer. Edit the message, or fix user.name/user.email." >&2
}
exit "$status"
