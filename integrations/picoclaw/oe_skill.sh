#!/bin/bash
# ocean eterna skill wrapper for picoclaw
# calls the OE REST API directly via curl — no python/mcporter dependency
#
# usage:
#   oe_skill.sh search "query here"
#   oe_skill.sh stats
#   oe_skill.sh health
#   oe_skill.sh add-file <filename> <content>
#   oe_skill.sh add-file-path /path/to/file
#   oe_skill.sh add-document /path/to/file
#   oe_skill.sh get-chunk <chunk_id> [context_window]
#   oe_skill.sh catalog [page] [page_size]
#   oe_skill.sh tell-me-more <turn_id>
#   oe_skill.sh reconstruct <chunk_id1> [chunk_id2] ...

set -euo pipefail

OE_BASE="${OE_BASE_URL:-http://localhost:9090}"

usage() {
    echo "usage: oe_skill.sh <command> [args...]"
    echo ""
    echo "commands:"
    echo "  search <query>                    search the knowledge base"
    echo "  stats                             server statistics"
    echo "  health                            server health check"
    echo "  add-file <name> <content>         ingest text content"
    echo "  add-file-path <path>              ingest file from disk"
    echo "  add-document <path>               ingest any document format"
    echo "  get-chunk <id> [context_window]   retrieve a chunk by ID"
    echo "  catalog [page] [page_size]        browse indexed chunks"
    echo "  tell-me-more <turn_id>            expand previous search"
    echo "  reconstruct <id1> [id2] ...       combine chunks"
    exit 1
}

[[ $# -lt 1 ]] && usage

CMD="$1"
shift

case "$CMD" in
    search)
        [[ $# -lt 1 ]] && { echo "error: search requires a query"; exit 1; }
        QUERY="$*"
        curl -s -X POST "${OE_BASE}/chat" \
            -H "Content-Type: application/json" \
            -d "{\"question\": $(echo "$QUERY" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))')}"
        ;;

    stats)
        curl -s "${OE_BASE}/stats"
        ;;

    health)
        curl -s "${OE_BASE}/health"
        ;;

    add-file)
        [[ $# -lt 2 ]] && { echo "error: add-file requires <filename> <content>"; exit 1; }
        FILENAME="$1"
        shift
        CONTENT="$*"
        curl -s -X POST "${OE_BASE}/add-file" \
            -H "Content-Type: application/json" \
            -d "{\"filename\": $(echo "$FILENAME" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))'), \"content\": $(echo "$CONTENT" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))')}"
        ;;

    add-file-path)
        [[ $# -lt 1 ]] && { echo "error: add-file-path requires a path"; exit 1; }
        curl -s -X POST "${OE_BASE}/add-file-path" \
            -H "Content-Type: application/json" \
            -d "{\"path\": $(echo "$1" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))')}"
        ;;

    add-document)
        [[ $# -lt 1 ]] && { echo "error: add-document requires a path"; exit 1; }
        # note: add-document uses server-side doc processing if available,
        # otherwise falls back to add-file-path for plain text
        curl -s -X POST "${OE_BASE}/add-file-path" \
            -H "Content-Type: application/json" \
            -d "{\"path\": $(echo "$1" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))')}"
        ;;

    get-chunk)
        [[ $# -lt 1 ]] && { echo "error: get-chunk requires a chunk ID"; exit 1; }
        CHUNK_ID="$1"
        CTX="${2:-1}"
        curl -s "${OE_BASE}/chunk/${CHUNK_ID}?context_window=${CTX}"
        ;;

    catalog)
        PAGE="${1:-1}"
        SIZE="${2:-20}"
        curl -s "${OE_BASE}/catalog?page=${PAGE}&page_size=${SIZE}"
        ;;

    tell-me-more)
        [[ $# -lt 1 ]] && { echo "error: tell-me-more requires a turn_id"; exit 1; }
        curl -s -X POST "${OE_BASE}/tell-me-more" \
            -H "Content-Type: application/json" \
            -d "{\"turn_id\": $(echo "$1" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))')}"
        ;;

    reconstruct)
        [[ $# -lt 1 ]] && { echo "error: reconstruct requires at least one chunk ID"; exit 1; }
        # build JSON array from args
        IDS="["
        FIRST=true
        for id in "$@"; do
            if [ "$FIRST" = true ]; then
                IDS="${IDS}\"${id}\""
                FIRST=false
            else
                IDS="${IDS},\"${id}\""
            fi
        done
        IDS="${IDS}]"
        curl -s -X POST "${OE_BASE}/reconstruct" \
            -H "Content-Type: application/json" \
            -d "{\"chunk_ids\": ${IDS}}"
        ;;

    *)
        echo "error: unknown command '$CMD'"
        usage
        ;;
esac

echo ""
