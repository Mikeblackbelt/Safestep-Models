#!/usr/bin/env bash
# ==============================================================================
# SafeStep Model Training Terminal Helper
# ==============================================================================
# Fast, easy terminal training command for SafeStep models (CART, JAW, PHISHING, or ALL).
# Auto-detects virtual environment, verifies dependencies, and supports both
# interactive and direct CLI workflows.
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color helpers
BOLD='\033[1m'
DIM='\033[2m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Determine Python executable (prioritize virtualenv if present)
if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PY_BIN="$VIRTUAL_ENV/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PY_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
    PY_BIN="$SCRIPT_DIR/venv/bin/python"
elif [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/Scripts/python.exe" ]; then
    PY_BIN="$VIRTUAL_ENV/Scripts/python.exe"
elif [ -x "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
    PY_BIN="$SCRIPT_DIR/.venv/Scripts/python.exe"
elif [ -x "$SCRIPT_DIR/venv/Scripts/python.exe" ]; then
    PY_BIN="$SCRIPT_DIR/venv/Scripts/python.exe"
elif command -v python3 &>/dev/null; then
    PY_BIN="python3"
elif command -v python &>/dev/null; then
    PY_BIN="python"
else
    echo -e "${RED}Error: Python not found on PATH or in .venv/bin/python.${NC}" >&2
    exit 1
fi

print_help() {
    echo -e "${BOLD}SafeStep Easy Terminal Trainer${NC}"
    echo -e "Usage: ${CYAN}./train.sh [COMMAND | MODEL] [OPTIONS...]${NC}\n"
    echo -e "${BOLD}Quick Model Commands:${NC}"
    echo -e "  ${GREEN}./train.sh cart${NC}                 Train CART (AI-text detector)"
    echo -e "  ${GREEN}./train.sh jaw${NC}                  Train JAW (AI-image detector)"
    echo -e "  ${GREEN}./train.sh phishing${NC}             Train PHISHING (phishing detector)"
    echo -e "  ${GREEN}./train.sh all${NC}                  Train all models concurrently"
    echo -e "  ${GREEN}./train.sh demo [cart|jaw|phishing|all]${NC}
                                  Run instant training on demo data"
    echo ""
    echo -e "${BOLD}Utility & Monitoring Commands:${NC}"
    echo -e "  ${CYAN}./train.sh prepare [model] [data-dir]${NC}
                                  Generate sample datasets (model: cart|jaw|phishing|all)"
    echo -e "  ${CYAN}./train.sh download [sources...]${NC}
                                  Download real phishing datasets (default: hf_emails hf_urls; 'all' for every source)"
    echo -e "  ${CYAN}./train.sh inspect [dir]${NC}        Inspect a folder and report image/text counts"
    echo -e "  ${CYAN}./train.sh cifake <dir>${NC}         Import downloaded CIFAKE folder into data/jaw"
    echo -e "  ${CYAN}./train.sh dashboard${NC}            Train with full-screen live dashboard & metrics"
    echo -e "  ${CYAN}./train.sh estimate [cart|jaw|phishing|all]${NC}
                                  Preview training time & hardware plan without training"
    echo -e "  ${CYAN}./train.sh history${NC}              Show history, charts, and metrics of past runs"
    echo -e "  ${CYAN}./train.sh interactive${NC}          Interactive step-by-step menu wizard"
    echo ""
    echo -e "${BOLD}Common Options:${NC}"
    echo -e "  --data-dir PATH           Custom dataset path (default: data/)"
    echo -e "  --device [cpu|cuda|mps]   Force compute device (default: auto-detected)"
    echo -e "  --workers N               Data loader worker threads"
    echo -e "  --batch-size N            Override batch size"
    echo -e "  --lr RATE                 Override learning rate"
    echo -e "  --help                    Show full training CLI help"
}

interactive_mode() {
    echo -e "${CYAN}${BOLD}=== SafeStep Interactive Training Wizard ===${NC}\n"
    echo -e "Select what you would like to do:"
    echo -e "  ${BOLD}1)${NC} Train CART (AI-generated text detector)"
    echo -e "  ${BOLD}2)${NC} Train JAW (AI-generated image detector)"
    echo -e "  ${BOLD}3)${NC} Train Both (CART & JAW) in parallel"
    echo -e "  ${BOLD}8)${NC} Train PHISHING (phishing email/URL detector)"
    echo -e "  ${BOLD}4)${NC} Quick Demo training (test pipeline on synthetic data)"
    echo -e "  ${BOLD}5)${NC} Prepare sample training datasets in data/"
    echo -e "  ${BOLD}6)${NC} Estimate training time only"
    echo -e "  ${BOLD}7)${NC} View past training history and metrics"
    echo -e "  ${BOLD}q)${NC} Quit"
    echo ""
    read -rp "Enter choice [1-8/q]: " choice

    case "$choice" in
        1)
            echo -e "\n${GREEN}Starting CART training...${NC}"
            exec "$PY_BIN" train.py --model cart
            ;;
        2)
            echo -e "\n${GREEN}Starting JAW training...${NC}"
            exec "$PY_BIN" train.py --model jaw
            ;;
        3)
            echo -e "\n${GREEN}Starting parallel training (both models)...${NC}"
            exec "$PY_BIN" train.py --model all --dashboard
            ;;
        4)
            read -rp "Which model for demo? (cart/jaw/phishing/all) [all]: " demo_model
            demo_model="${demo_model:-all}"
            echo -e "\n${GREEN}Running demo training for ${demo_model}...${NC}"
            exec "$PY_BIN" train.py --model "$demo_model" --demo
            ;;
        5)
            echo -e "\n${GREEN}Preparing sample datasets...${NC}"
            exec "$PY_BIN" train.py --prepare-data all
            ;;
        6)
            read -rp "Which model to estimate? (cart/jaw/phishing/all) [all]: " est_model
            est_model="${est_model:-all}"
            exec "$PY_BIN" train.py --model "$est_model" --estimate-only
            ;;
        7)
            exec "$PY_BIN" train.py --history
            ;;
        8)
            echo -e "
${GREEN}Starting PHISHING training...${NC}"
            exec "$PY_BIN" train.py --model phishing
            ;;
        q|Q)
            echo "Cancelled."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid selection.${NC}"
            exit 1
            ;;
    esac
}

# No arguments provided -> run interactive wizard if terminal is interactive, else show help
if [ $# -eq 0 ]; then
    if [ -t 0 ]; then
        interactive_mode
    else
        print_help
    fi
    exit 0
fi

CMD="$1"
shift

case "$CMD" in
    -h|--help|help)
        print_help
        echo ""
        echo -e "${DIM}--- Underlying Python CLI options: ---${NC}"
        "$PY_BIN" train.py --help
        exit 0
        ;;
    interactive|wizard|menu)
        interactive_mode
        ;;
    cart|jaw|phishing|all)
        exec "$PY_BIN" train.py --model "$CMD" "$@"
        ;;
    demo)
        TARGET_MODEL="all"
        if [ $# -gt 0 ] && [[ "$1" =~ ^(cart|jaw|phishing|all)$ ]]; then
            TARGET_MODEL="$1"
            shift
        fi
        exec "$PY_BIN" train.py --model "$TARGET_MODEL" --demo "$@"
        ;;
    prepare|prep-data|sample)
        TARGET="all"
        if [ $# -gt 0 ] && [[ "$1" =~ ^(cart|jaw|phishing|all)$ ]]; then
            TARGET="$1"
            shift
        fi
        # Bare path (not a flag) -> treat as --data-dir
        if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
            set -- --data-dir "$1" "${@:2}"
        fi
        exec "$PY_BIN" train.py --prepare-data "$TARGET" "$@"
        ;;
    download|dl)
        exec "$PY_BIN" train.py --download-data "$@"
        ;;
    inspect)
        DIR="${1:-data}"
        exec "$PY_BIN" -m util.manager --inspect "$DIR"
        ;;
    cifake)
        if [ -z "$1" ]; then
            echo -e "${RED}Usage: ./train.sh cifake <path_to_extracted_cifake>${NC}"
            exit 1
        fi
        CIFAKE_PATH="$1"
        shift
        exec "$PY_BIN" -m util.manager --cifake "$CIFAKE_PATH" "$@"
        ;;
    dashboard|dash)
        exec "$PY_BIN" train.py --model all --dashboard "$@"
        ;;
    estimate|est)
        TARGET_MODEL="all"
        if [ $# -gt 0 ] && [[ "$1" =~ ^(cart|jaw|phishing|all)$ ]]; then
            TARGET_MODEL="$1"
            shift
        fi
        exec "$PY_BIN" train.py --model "$TARGET_MODEL" --estimate-only "$@"
        ;;
    history|hist|logs)
        exec "$PY_BIN" train.py --history "$@"
        ;;
    *)
        # Passthrough any custom flags directly to train.py
        exec "$PY_BIN" train.py "$CMD" "$@"
        ;;
esac
