#!/usr/bin/env bash
# jepa-studio: unpack, set up, test and publish from an Ubuntu terminal (WSL2 or plain Ubuntu).
#
#   bash setup_ubuntu.sh [ZIP] [--cpu] [--browser-tests] [--skip-tests] [--no-push]
#                        [--ssh-host HOST] [--https] [--dest DIR] [--yes] [--doctor]
#
# (Saved outside the repo it is usually called ~/setup_jepa_studio.sh; same script.)
#
# Steps, all safe to re-run:
#   1  preflight: what machine this is, where the log goes
#   2  find the zip: the path you give, else the newest jepa-studio*.zip (including Chrome's
#      "jepa-studio-v0.2.0 (1).zip" re-downloads) in Windows Downloads, ~/Downloads, ~ and
#      this script's folder
#   3  system packages: ONE `sudo apt-get install` for whatever is missing among unzip git curl
#      ca-certificates python3-venv python3-pip build-essential (+ the GitHub CLI `gh`)
#   4  unpack to ~/jepa-studio (never into ~ itself). An existing git repo there is updated in
#      place: history kept, files deleted in the new version removed
#   5  Python: conda env "jepa-studio" (Python 3.12) if conda is installed, else ~/jepa-studio/.venv
#      (newest system Python 3.10-3.14, or Python 3.12 via uv). PyTorch: CUDA wheels when
#      `nvidia-smi -L` works, else the small CPU wheels. Then pip install -e ".[dev,video,onnx]"
#   6  tests: pytest -m "not slow" (browser tests only with --browser-tests). If something fails
#      you are asked "Publish anyway? [y/N]"
#   7  GitHub sign-in (gh, account Normansrule, `workflow` permission when pushing over HTTPS)
#   8  git identity (asked once, stored in this repo only) + commit
#   9  push: SSH through the github-normansrule host alias when it signs in as Normansrule,
#      else HTTPS through gh
#  10  GitHub Pages (source: GitHub Actions) + Actions on, Pages workflow started
#  11  optional: push tag vX.Y.Z so .github/workflows/desktop.yml builds the desktop installers
#
# Options:
#   ZIP               path to the zip (a Windows path like C:\Users\me\Downloads\x.zip works too)
#   --cpu             CPU-only PyTorch even if an NVIDIA GPU is present (much smaller download)
#   --browser-tests   install Playwright's Chromium and run the headless browser tests too
#   --skip-tests      don't run the tests
#   --no-push         stop after the tests (no git commit, nothing sent to GitHub)
#   --ssh-host HOST   push over SSH through this ~/.ssh/config host alias
#   --https           push over HTTPS through the GitHub CLI, even if SSH would work
#   --dest DIR        where the repo lives (default ~/jepa-studio)
#   --yes, -y         answer yes to "Publish anyway?" and "Build desktop installers now?"
#   --doctor          print a read-only report of this machine and exit; changes nothing
#   -h, --help        this text
#
# Everything is also written to ~/jepa-studio-setup-YYYYmmdd-HHMMSS.log. If a step fails, the
# script names the step and the command; paste the last 40 lines of that log to Claude.
#
# Test-only environment hooks (used by tests/test_setup_script.py, not for normal use):
#   JEPA_SETUP_SEARCH_DIRS  colon-separated folders (globs allowed) to look for the zip in
#   JEPA_SETUP_DRY_PIP=1    don't create an environment or install anything with pip; use the
#                           `python3` found on PATH and only log the pip / playwright commands

# Started with `sh`? Re-run under bash.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Sourced? `exit` would close the terminal, so refuse.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Run this with:  bash ${BASH_SOURCE[0]}   (not 'source' or '.')" >&2
  return 1 2>/dev/null || exit 1
fi

set -Eeuo pipefail

OWNER=Normansrule
REPO=jepa-studio
DEFAULT_SSH_HOST=github-normansrule
ENV_NAME=jepa-studio
PAGES_URL="https://normansrule.github.io/$REPO/"
DESCRIPTION="Pretrain, inspect and use LeJEPA models on your own unlabeled data, with a planning bot in a learned world model. Web demo + desktop app."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# options
ZIP="" CPU=0 BROWSER=0 SKIP_TESTS=0 PUSH=1 SSH_HOST_OPT="" FORCE_HTTPS=0 DEST="" YES=0 DOCTOR=0
DRY_PIP="${JEPA_SETUP_DRY_PIP:-0}"
# state
LOG="" TEE_PID="" WORK="" STEP_NO=0 STEP_TOTAL=6 STEP_NAME="start"
PY="" ACTIVATE="" TRANSPORT="" SSH_HOST="" REMOTE_URL="" TAG_PUSHED=""

if [ -t 1 ]; then B=$'\033[1m' CY=$'\033[1;36m' RD=$'\033[1;31m' YL=$'\033[1;33m' GR=$'\033[1;32m' N0=$'\033[0m'
else B="" CY="" RD="" YL="" GR="" N0=""; fi

# ------------------------------------------------------------------------------------ helpers
usage() { sed -n '2,/^# Test-only/{/^# Test-only/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; }
usage_die() { printf '%s\n\n' "$*" >&2; usage >&2; exit 2; }

step() {
  STEP_NO=$((STEP_NO + 1)); STEP_NAME="$*"
  printf '\n%s[%d/%d] %s%s\n' "$CY" "$STEP_NO" "$STEP_TOTAL" "$*" "$N0"
}
info() { printf '    %s\n' "$*"; }
ok()   { printf '    %sok%s %s\n' "$GR" "$N0" "$*"; }
warn() { printf '    %s! %s%s\n' "$YL" "$*" "$N0" >&2; }
die() {
  printf '\n%sSTOPPED at step %d (%s):%s %s\n' "$RD" "$STEP_NO" "$STEP_NAME" "$N0" "$1" >&2
  shift
  local l; for l in "$@"; do printf '    %s\n' "$l" >&2; done
  if [ -n "$LOG" ]; then
    printf '    full log: %s\n' "$LOG" >&2
    printf '    If this is unclear, paste the last 40 lines of the log to Claude:  tail -n 40 %q\n' "$LOG" >&2
  fi
  exit 1
}

on_err() {
  local rc=$1 line=$2 cmd=$3
  if [ "${BASHPID:-$$}" != "$$" ]; then           # inside $(...) or a subshell: the parent reports
    printf '    (inner command failed, line %s: %s)\n' "$line" "$cmd" >&2
    exit "$rc"
  fi
  trap - ERR
  printf '\n%sFAILED at step %d (%s)%s, line %s, exit code %s\n' "$RD" "$STEP_NO" "$STEP_NAME" "$N0" "$line" "$rc" >&2
  printf '    command: %s\n' "$cmd" >&2
  if [ -n "$LOG" ]; then
    printf '    full log: %s\n' "$LOG" >&2
    printf '    %sPaste the last 40 lines of the log to Claude:%s  tail -n 40 %q\n' "$B" "$N0" "$LOG" >&2
  fi
  exit "$rc"
}
arm_err_trap() { trap 'on_err $? $LINENO "$BASH_COMMAND"' ERR; }

on_exit() {                                        # the script's exit status is kept as is
  trap - ERR
  [ -z "$WORK" ] || rm -rf "$WORK"
  if [ -n "$TEE_PID" ]; then exec 1>&3 2>&4; wait "$TEE_PID" 2>/dev/null || true; fi
}
on_int() { printf '\nInterrupted (Ctrl-C) during step %d (%s). Re-run the same command to continue.\n' "$STEP_NO" "$STEP_NAME" >&2; exit 130; }

start_log() {
  LOG="$HOME/jepa-studio-setup-$(date +%Y%m%d-%H%M%S).log"
  : >"$LOG" || { LOG=""; warn "can't write a log file in $HOME"; return; }
  exec 3>&1 4>&2
  # terminal gets colours; the log file gets plain text
  exec > >(tee >(sed -u 's/\x1b\[[0-9;]*m//g' >>"$LOG")) 2>&1
  TEE_PID=$!
}

# yes/no question, default No. --yes answers yes; with no terminal the answer is No.
ask_yn() {
  local q=$1 a=""
  if [ "$YES" = 1 ]; then info "$q [y/N] y  (--yes)"; return 0; fi
  if [ ! -t 0 ]; then info "$q [y/N] n  (no terminal to ask; pass --yes to answer yes)"; return 1; fi
  read -r -p "    $q [y/N] " a || a=""
  case "$a" in [yY] | [yY][eE][sS]) return 0 ;; *) return 1 ;; esac
}
# free-text question with a default (the default is used when there is no terminal)
ask_text() {
  local q=$1 def=$2 a=""
  if [ -t 0 ]; then read -r -p "    $q [$def]: " a || a=""; fi
  printf '%s' "${a:-$def}"
}
# Commands that talk to the user themselves (gh sign-in). Attach them to the terminal if there is one.
run_interactive() {
  if [ -t 0 ] && { : >/dev/tty; } 2>/dev/null; then
    "$@" </dev/tty >/dev/tty 2>/dev/tty
  else
    "$@"
  fi
}
have() { command -v "$1" >/dev/null 2>&1; }
is_wsl() { [ -n "${WSL_DISTRO_NAME:-}" ] || grep -qi microsoft /proc/version 2>/dev/null; }
os_name() { if [ -r /etc/os-release ]; then (. /etc/os-release && printf '%s' "${PRETTY_NAME:-$NAME}"); else uname -sr; fi; }
pkg_ok() { local s; s=$(dpkg-query -W -f='${Status}' "$1" 2>/dev/null || true); [[ $s == *"install ok installed"* ]]; }
run_root() { if [ "$(id -u)" = 0 ]; then "$@"; else sudo "$@"; fi; }

py_in_range() { "$1" -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] <= (3, 14)))' >/dev/null 2>&1; }

find_conda() {
  local c
  for c in "${CONDA_EXE:-}" "$(command -v conda 2>/dev/null || true)" "$HOME/miniforge3/bin/conda" \
           "$HOME/mambaforge/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" /opt/conda/bin/conda; do
    if [ -n "$c" ] && [ -f "$c" ] && [ -x "$c" ]; then printf '%s' "$c"; return 0; fi
  done
  return 1
}

# newest system python in [3.10, 3.14]
pick_system_python() {
  local v p
  for v in 3.14 3.13 3.12 3.11 3.10; do
    p=$(command -v "python$v" 2>/dev/null || true)
    if [ -n "$p" ] && py_in_range "$p"; then printf '%s' "$p"; return 0; fi
  done
  p=$(command -v python3 2>/dev/null || true)
  if [ -n "$p" ] && py_in_range "$p"; then printf '%s' "$p"; return 0; fi
  return 1
}

search_dirs() {
  local spec pat d
  spec="${JEPA_SETUP_SEARCH_DIRS:-/mnt/*/Users/*/Downloads:$HOME/Downloads:$HOME:$SCRIPT_DIR}"
  local IFS=:
  for pat in $spec; do
    [ -n "$pat" ] || continue
    while IFS= read -r d; do [ -d "$d" ] && printf '%s\n' "$d"; done < <(compgen -G "$pat" || true)
  done
}
# all candidate zips, newest first, as "mtime<TAB>path"
list_zips() {
  local d
  while IFS= read -r d; do
    find "$d" -maxdepth 1 -type f -iname 'jepa-studio*.zip' -printf '%T@\t%p\n' 2>/dev/null || true
  done < <(search_dirs | awk '!seen[$0]++') | sort -t "$(printf '\t')" -k1,1 -rn
}

pip_install() {
  if [ "$DRY_PIP" = 1 ]; then info "[dry-pip] pip install $*"; return 0; fi
  "$PY" -m pip install "$@" || die "pip install $* failed (see the pip messages above)" \
    "'No matching distribution' or 'connection' errors: check the network (VPN/proxy) with  bash $0 --doctor" \
    "Out of disk space? CUDA PyTorch needs ~8 GB; --cpu needs ~2 GB."
}

# ------------------------------------------------------------------------------------ doctor
doctor() {
  local p v c u code conda_bin hn out scopes login
  printf '%sjepa-studio setup: preflight report (read-only)%s\n\n' "$B" "$N0"
  printf '%-22s %s\n' "OS" "$(os_name) ($(uname -m), kernel $(uname -r))"
  if is_wsl; then printf '%-22s yes (distro %s)\n' "WSL" "${WSL_DISTRO_NAME:-unknown}"; else printf '%-22s no\n' "WSL"; fi
  printf '%-22s %s, user %s, HOME %s\n' "shell" "bash $BASH_VERSION" "$(id -un)" "$HOME"
  if [ -e "$HOME/.git" ]; then printf '%-22s %s\n' "home is a git repo" "yes (the script never runs git there; the repo goes to its own folder)"; fi
  printf '%-22s %s\n' "repo folder" "$DEST$( [ -d "$DEST/.git" ] && printf ' (git repo, origin: %s)' "$(git -C "$DEST" remote get-url origin 2>/dev/null || echo none)" )"

  printf '\n%sPython%s\n' "$B" "$N0"
  for p in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
    c=$(command -v "$p" 2>/dev/null || true); [ -n "$c" ] || continue
    v=$("$c" --version 2>&1 || true)
    if py_in_range "$c"; then printf '  %-20s %s  %s\n' "$p" "$v" "$c"; else printf '  %-20s %s  %s (outside 3.10-3.14)\n' "$p" "$v" "$c"; fi
  done
  if p=$(pick_system_python); then
    if "$p" -c 'import ensurepip, venv' >/dev/null 2>&1; then v="venv works"; else v="venv module incomplete (python3-venv missing)"; fi
    printf '  %-20s %s (%s)\n' "would use (no conda)" "$p" "$v"
  else
    printf '  %-20s %s\n' "would use (no conda)" "none in 3.10-3.14: uv would install Python 3.12"
  fi
  if conda_bin=$(find_conda); then
    printf '  %-20s %s (%s), active env: %s, env "%s": %s\n' "conda" "$conda_bin" "$("$conda_bin" --version 2>/dev/null || echo '?')" \
      "${CONDA_DEFAULT_ENV:-none}" "$ENV_NAME" "$( [ -d "$("$conda_bin" info --base 2>/dev/null)/envs/$ENV_NAME" ] && echo exists || echo 'not yet')"
  else
    printf '  %-20s %s\n' "conda" "not found (a .venv will be used)"
  fi
  printf '  %-20s %s\n' "uv" "$(command -v uv 2>/dev/null || { [ -x "$HOME/.local/bin/uv" ] && echo "$HOME/.local/bin/uv"; } || echo 'not installed')"
  if have nvidia-smi && out=$(nvidia-smi -L 2>/dev/null) && [ -n "$out" ]; then printf '  %-20s %s\n' "NVIDIA GPU" "${out%%$'\n'*}"; else printf '  %-20s %s\n' "NVIDIA GPU" "none (CPU PyTorch)"; fi

  printf '\n%sTools and packages%s\n' "$B" "$N0"
  for p in unzip git curl ca-certificates python3-venv python3-pip build-essential; do
    if have dpkg-query; then if pkg_ok "$p"; then v=installed; else v=MISSING; fi
    elif have "$p"; then v=present; else v=missing; fi
    printf '  %-20s %s\n' "$p" "$v"
  done
  if have sudo; then if sudo -n true 2>/dev/null; then v="works without a password"; else v="will ask for your Linux password"; fi; else v="not installed"; fi
  [ "$(id -u)" != 0 ] || v="not needed (running as root)"
  printf '  %-20s %s\n' "sudo" "$v"
  printf '  %-20s %s\n' "node" "$(node --version 2>/dev/null || echo 'not installed (JavaScript parity tests are skipped)')"

  printf '\n%sgit and GitHub%s\n' "$B" "$N0"
  if have git; then
    printf '  %-20s %s\n' "git" "$(git --version)"
    printf '  %-20s %s / %s\n' "global identity" "$(git -C / config --global user.name 2>/dev/null || echo '(no name)')" "$(git -C / config --global user.email 2>/dev/null || echo '(no email)')"
  else
    printf '  %-20s %s\n' "git" "not installed"
  fi
  if have gh; then
    printf '  %-20s %s\n' "gh" "$( { gh --version 2>/dev/null || echo 'installed (version unknown)'; } | head -n 1)"
    if gh auth status -h github.com >/dev/null 2>&1; then
      login=$(gh api user --jq .login 2>/dev/null || echo '?')
      scopes=$(gh api -i user 2>/dev/null | tr -d '\r' | awk -F': ' 'tolower($1)=="x-oauth-scopes"{print $2}' || true)
      printf '  %-20s signed in as %s%s\n' "gh auth" "$login" "$( [ "$login" = "$OWNER" ] || printf ' (needs %s: gh auth switch -h github.com -u %s)' "$OWNER" "$OWNER")"
      printf '  %-20s %s\n' "token scopes" "${scopes:-unknown (fine-grained token or GH_TOKEN)}"
      gh auth status -h github.com 2>&1 | grep -v -i 'token:' | sed 's/^/      /' || true
    else
      printf '  %-20s %s\n' "gh auth" "not signed in (the script will start: gh auth login -h github.com -p https -s workflow -w)"
    fi
  else
    printf '  %-20s %s\n' "gh" "not installed (the script installs it)"
  fi
  if have ssh; then
    for p in "$DEFAULT_SSH_HOST" github.com; do
      hn=$(ssh -G "$p" 2>/dev/null | awk '$1=="hostname"{print $2}' || true)
      if [ "$p" != github.com ] && [ "$hn" != github.com ] && [ "$hn" != ssh.github.com ]; then
        printf '  %-20s %s\n' "ssh $p" "no such alias in ~/.ssh/config (HTTPS will be used)"; continue
      fi
      local kh; kh=$(mktemp); [ ! -f "$HOME/.ssh/known_hosts" ] || cp "$HOME/.ssh/known_hosts" "$kh"
      out=$(timeout 20 ssh -T -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$kh" "git@$p" 2>&1 || true)
      rm -f "$kh"
      out=$(printf '%s' "$out" | grep -E 'Hi |denied|refused|resolve|timed out' | sed -n 1p || true)
      printf '  %-20s %s\n' "ssh $p" "${out:-no answer}"
    done
    out=$(cd "$HOME/.ssh" 2>/dev/null && ls -- *.pub 2>/dev/null | tr '\n' ' ' || true)
    printf '  %-20s %s\n' "ssh keys" "${out:-none in ~/.ssh}"
  else
    printf '  %-20s %s\n' "ssh" "not installed (HTTPS will be used)"
  fi

  printf '\n%sDisk and network%s\n' "$B" "$N0"
  printf '  %-20s %s\n' "free in HOME" "$(df -h "$HOME" 2>/dev/null | awk 'NR==2{print $4}') (CPU PyTorch needs ~2 GB, CUDA ~8 GB)"
  for u in https://pypi.org/simple/ https://github.com https://download.pytorch.org/whl/cpu/; do
    if have curl; then code=$(curl -sI -o /dev/null -w '%{http_code}' --max-time 8 "$u" 2>/dev/null || true); else code="curl missing"; fi
    case "$code" in 2* | 3*) v="reachable (HTTP $code)" ;; 000 | "") v="NOT reachable (DNS/proxy/VPN?)" ;; *) v="HTTP $code" ;; esac
    printf '  %-34s %s\n' "$u" "$v"
  done

  printf '\n%sZip%s\n' "$B" "$N0"
  if [ -n "$ZIP" ]; then printf '  %s\n' "$ZIP (given)"
  else
    local z; z=$(list_zips | sed -n 1p | cut -f2-)
    printf '  %s\n' "${z:-no jepa-studio*.zip found in: $(search_dirs | tr '\n' ' ')}"
  fi
  printf '\nNothing was changed.\n'
}

# ------------------------------------------------------------------------------------ steps
step_preflight() {
  step "Preflight"
  info "machine: $(os_name)$(is_wsl && printf ', WSL2 distro %s' "${WSL_DISTRO_NAME:-?}"), user $(id -un)"
  info "repo folder: $DEST"
  info "log: ${LOG:-(none)}"
  local opts=""
  [ "$CPU" = 0 ] || opts+=" --cpu"; [ "$BROWSER" = 0 ] || opts+=" --browser-tests"; [ "$SKIP_TESTS" = 0 ] || opts+=" --skip-tests"
  [ "$PUSH" = 1 ] || opts+=" --no-push"; [ -z "$SSH_HOST_OPT" ] || opts+=" --ssh-host $SSH_HOST_OPT"; [ "$FORCE_HTTPS" = 0 ] || opts+=" --https"
  [ "$YES" = 0 ] || opts+=" --yes"; [ "$DRY_PIP" != 1 ] || opts+=" (JEPA_SETUP_DRY_PIP=1: test mode, no installs)"
  info "options:${opts:- none}"
  case "$DEST" in /mnt/*) warn "$DEST is on the Windows drive: much slower and file permissions are odd. ~/jepa-studio is better." ;; esac
  if [ -e "$HOME/.git" ]; then info "note: your home folder is a git repo; this script only runs git inside $DEST"; fi
}

step_find_zip() {
  step "Find the zip"
  if [ -n "$ZIP" ]; then
    if [[ $ZIP =~ ^[A-Za-z]:[\\/] ]] && have wslpath; then ZIP=$(wslpath -u "$ZIP"); fi
    [ -f "$ZIP" ] || die "no file at $ZIP" "Check the path; quote it if it has spaces, e.g. \"~/Downloads/jepa-studio-v0.2.0 (1).zip\""
  else
    local all; all=$(list_zips)
    if [ -z "$all" ]; then
      die "no jepa-studio*.zip found" "Looked in: $(search_dirs | tr '\n' ' ')" \
        "Download it first, or pass the path:  bash $0 /path/to/jepa-studio.zip"
    fi
    ZIP=$(printf '%s\n' "$all" | sed -n 1p | cut -f2-)
    if [ "$(printf '%s\n' "$all" | wc -l)" -gt 1 ]; then
      info "found $(printf '%s\n' "$all" | wc -l) zips; using the newest. Others:"
      printf '%s\n' "$all" | sed -n '2,4p' | cut -f2- | sed 's/^/        /'
    fi
  fi
  ZIP=$(realpath -- "$ZIP")
  ok "using $ZIP ($(du -h -- "$ZIP" | cut -f1), modified $(date -r "$ZIP" '+%Y-%m-%d %H:%M'))"
}

add_gh_apt_repo() {
  info "adding the GitHub CLI apt repository (cli.github.com)"
  local k; k=$(mktemp)
  curl -fsSL --max-time 60 https://cli.github.com/packages/githubcli-archive-keyring.gpg -o "$k" || { rm -f "$k"; return 1; }
  run_root mkdir -p -m 755 /etc/apt/keyrings /etc/apt/sources.list.d
  run_root install -m 644 "$k" /etc/apt/keyrings/githubcli-archive-keyring.gpg
  rm -f "$k"
  printf 'deb [arch=%s signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main\n' \
    "$(dpkg --print-architecture)" | run_root tee /etc/apt/sources.list.d/github-cli.list >/dev/null
}

apt_update() {
  run_root apt-get update -qq || die "apt-get update failed (see the lines above)" \
    "WSL: 'Release file ... is not valid yet' means the clock drifted: run  sudo hwclock -s  (or 'wsl --shutdown' in PowerShell) and retry." \
    "'Temporary failure resolving' means no DNS/network: check VPN, then retry."
}
apt_install() { run_root env DEBIAN_FRONTEND=noninteractive apt-get install -y -q "$@" || die "apt-get install $* failed (see above)"; }

step_packages() {
  step "System packages"
  local want_gh=0 need=() p
  if [ "$PUSH" = 1 ] && ! have gh; then want_gh=1; fi
  if ! have dpkg-query || ! have apt-get; then
    for p in unzip git curl; do have "$p" || need+=("$p"); done
    [ "$want_gh" = 0 ] || need+=(gh)
    [ ${#need[@]} -eq 0 ] || die "this isn't a Debian/Ubuntu system and these are missing: ${need[*]}" "Install them with your package manager and re-run."
    ok "not apt-based, but the needed tools are present"; return
  fi
  for p in unzip git curl ca-certificates python3-venv python3-pip build-essential; do pkg_ok "$p" || need+=("$p"); done
  if [ ${#need[@]} -eq 0 ] && [ "$want_gh" = 0 ]; then ok "all present, apt not needed"; return; fi

  info "missing: ${need[*]}${need[*]:+ }$( [ "$want_gh" = 1 ] && echo gh)"
  if [ "$(id -u)" != 0 ]; then
    have sudo || die "need to install: ${need[*]} gh, but sudo isn't available" "Log in as a user who can use sudo, or install those packages as root, then re-run."
    if ! sudo -n true 2>/dev/null; then
      [ -t 0 ] || die "sudo needs your password, but there is no terminal to type it in" "Run the script from a terminal, or first run:  sudo true"
      info "sudo will ask for your ${B}Ubuntu (Linux) password${N0}, not Windows/GitHub. Nothing shows while you type; that's normal."
    fi
  fi
  local gh_repo=0
  if [ "$want_gh" = 1 ] && have curl; then
    if add_gh_apt_repo; then gh_repo=1; else warn "couldn't add the GitHub CLI repository; trying Ubuntu's own gh package"; fi
  fi
  apt_update
  if [ "$want_gh" = 1 ] && have curl; then need+=(gh); fi
  [ ${#need[@]} -eq 0 ] || apt_install "${need[@]}"
  if [ "$want_gh" = 1 ] && ! have gh; then           # curl was missing before this step
    if add_gh_apt_repo; then gh_repo=1; apt_update; else warn "couldn't add the GitHub CLI repository; trying Ubuntu's own gh package"; fi
    apt_install gh
  fi
  ok "installed: ${need[*]}$( [ "$gh_repo" = 1 ] && echo ' (gh from cli.github.com)')"
}

step_unpack() {
  step "Unpack into $DEST"
  local t; t=$(unzip -tq "$ZIP" 2>&1) || die "$ZIP is not a complete zip" "$t" "Download it again (wait until the browser says it finished)."
  ok "zip verified, sha256 $(sha256sum -- "$ZIP" | cut -c1-16)…"
  WORK=$(mktemp -d)
  unzip -q "$ZIP" -d "$WORK/x"
  local src="" c
  for c in "$WORK/x" "$WORK/x"/* "$WORK/x"/*/*; do
    if [ -f "$c/pyproject.toml" ] && [ -d "$c/jepa_studio" ]; then src=$c; break; fi
  done
  [ -n "$src" ] || die "the zip doesn't contain jepa-studio (no pyproject.toml + jepa_studio/ inside)" "Is $ZIP the right file?"
  rm -rf "$src/.git"

  mkdir -p "$DEST"
  export GIT_CEILING_DIRECTORIES; GIT_CEILING_DIRECTORIES=$(dirname "$DEST")   # never find ~/.git from in here
  local dirs=""
  if [ -d "$DEST/.git" ]; then
    local top; top=$(git -C "$DEST" rev-parse --show-toplevel)
    [ "$(realpath "$top")" = "$DEST" ] || die "git says the repo top is $top, not $DEST; refusing to touch it"
    info "existing git repo: updating in place (history kept; files removed in the new version are deleted)"
    dirs=$(git -C "$DEST" ls-files | sed -n 's#/[^/]*$##p' | sort -ru)
    (cd "$DEST" && git ls-files -z | xargs -0 -r rm -f --)
  fi
  cp -a "$src"/. "$DEST"/
  if [ -n "$dirs" ]; then                           # tidy folders the new version no longer has
    (cd "$DEST" && while IFS= read -r c; do [ ! -d "$c" ] || rmdir -p --ignore-fail-on-non-empty "$c" 2>/dev/null || true; done <<<"$dirs")
  fi
  cd "$DEST"
  [ -f pyproject.toml ] && [ -d jepa_studio ] || die "$DEST does not look like jepa-studio after unpacking"
  ok "unpacked version $(project_version) to $DEST"
}

project_version() { sed -n "s/^__version__ *= *[\"']\([^\"']*\)[\"'].*/\1/p" "$DEST/jepa_studio/__init__.py"; }

setup_conda_env() {
  local conda_bin=$1 base rc=0 envs
  base=$("$conda_bin" info --base 2>/dev/null) || die "conda at $conda_bin doesn't answer 'conda info --base'"
  [ -f "$base/etc/profile.d/conda.sh" ] || die "can't find $base/etc/profile.d/conda.sh"
  info "conda found at $base: using env \"$ENV_NAME\""
  # conda's shell functions reference unset variables and expect failures to be non-fatal
  trap - ERR; set +eu
  # shellcheck disable=SC1091
  source "$base/etc/profile.d/conda.sh"
  for _ in 1 2 3 4 5 6; do [ "${CONDA_SHLVL:-0}" -gt 0 ] || break; conda deactivate >/dev/null 2>&1 || break; done
  envs=$(conda env list 2>/dev/null | awk '!/^#/{print $1}')
  if [[ $'\n'"$envs"$'\n' != *$'\n'"$ENV_NAME"$'\n'* ]]; then
    echo "    creating conda env $ENV_NAME (Python 3.12, conda-forge)"
    conda create -y -q -n "$ENV_NAME" -c conda-forge --override-channels python=3.12 pip; rc=$?
  fi
  if [ "$rc" = 0 ]; then conda activate "$ENV_NAME"; rc=$?; fi
  set -eu; arm_err_trap
  [ "$rc" = 0 ] || die "conda couldn't create/activate env $ENV_NAME (exit $rc, see above)" \
    "Try by hand:  conda create -n $ENV_NAME -c conda-forge python=3.12 pip" "Or run with conda not on PATH to use a .venv instead."
  [ "${CONDA_DEFAULT_ENV:-}" = "$ENV_NAME" ] && [ -x "${CONDA_PREFIX:-}/bin/python" ] || die "conda env $ENV_NAME didn't activate"
  PY="$CONDA_PREFIX/bin/python"
  ACTIVATE="conda activate $ENV_NAME"
}

setup_venv() {
  local venv="$DEST/.venv" sys_py uv
  if [ -x "$venv/bin/python" ] && py_in_range "$venv/bin/python" && "$venv/bin/python" -m pip --version >/dev/null 2>&1; then
    info "reusing $venv"
  else
    rm -rf "$venv"
    if sys_py=$(pick_system_python) && "$sys_py" -m venv "$venv"; then
      info "created $venv with $sys_py"
    else
      rm -rf "$venv"
      info "no usable system Python 3.10-3.14 with venv: getting Python 3.12 through uv"
      uv=$(command -v uv 2>/dev/null || true)
      if [ -z "$uv" ] && [ -x "$HOME/.local/bin/uv" ]; then uv="$HOME/.local/bin/uv"; fi
      if [ -z "$uv" ]; then
        info "installing uv (https://astral.sh/uv) into ~/.local/bin"
        curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
        uv="$HOME/.local/bin/uv"
      fi
      "$uv" venv --seed --python 3.12 "$venv"
    fi
  fi
  PY="$venv/bin/python"
  ACTIVATE="source $venv/bin/activate"
}

step_python() {
  step "Python environment + PyTorch"
  local conda_bin torch_index="https://download.pytorch.org/whl/cpu"
  if [ "$DRY_PIP" = 1 ]; then
    PY=$(command -v python3); ACTIVATE="(test mode: no environment)"
    info "[dry-pip] no environment created; using $PY"
  elif conda_bin=$(find_conda); then
    setup_conda_env "$conda_bin"
  else
    setup_venv
  fi
  ok "python: $PY ($("$PY" --version 2>&1))"

  if [ "$CPU" = 0 ] && have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
    torch_index=""
    info "NVIDIA GPU found ($(nvidia-smi -L 2>/dev/null | sed -n 1p)): CUDA PyTorch from PyPI, a few GB, can take 5-15 min"
  else
    info "CPU PyTorch (~200 MB)$( [ "$CPU" = 1 ] && echo ', --cpu given')"
  fi
  pip_install -q --upgrade pip
  if [ -n "$torch_index" ]; then pip_install torch torchvision --index-url "$torch_index"; else pip_install torch torchvision; fi
  info "installing jepa-studio (editable) with extras dev, video, onnx"
  pip_install -e ".[dev,video,onnx]"
  if [ "$DRY_PIP" != 1 ]; then
    "$PY" -c 'import torch; d = "cuda" if torch.cuda.is_available() else "cpu"; print(f"    ok torch {torch.__version__}, device {d}")'
  fi
}

step_tests() {
  step "Tests"
  if [ "$SKIP_TESTS" = 1 ]; then info "skipped (--skip-tests)"; return 0; fi
  local args=(-m pytest -q -m "not slow") failed=0 out="$WORK/pytest.out"
  if [ "$BROWSER" = 1 ]; then
    info "installing Playwright's Chromium for the browser tests"
    if [ "$DRY_PIP" = 1 ]; then info "[dry-pip] python -m playwright install --with-deps chromium"
    elif ! "$PY" -m playwright install --with-deps chromium; then
      warn "installing Chromium's system libraries failed (unsupported distro?); trying the browser alone"
      "$PY" -m playwright install chromium || warn "Chromium install failed: the browser tests will be skipped"
    fi
  else
    args+=(--ignore=tests/test_site_playwright.py --ignore=tests/test_desktop_tier.py)
    info "browser tests not run (add --browser-tests to include them)"
  fi
  have node || info "Node.js not found: the JavaScript parity tests skip themselves (sudo apt-get install -y nodejs)"
  info "running: python ${args[*]}"
  if "$PY" "${args[@]}" 2>&1 | tee "$out"; then :; else failed=1; fi
  if "$PY" scripts/build_docs_index.py --check; then :; else failed=1; warn "docs index check failed"; fi
  local cli; cli="$(dirname "$PY")/jepa-studio"
  if [ -x "$cli" ]; then if "$cli" --help >/dev/null; then ok "jepa-studio command works"; else failed=1; warn "'jepa-studio --help' failed"; fi; fi

  if [ "$failed" = 0 ]; then ok "tests passed"; return 0; fi
  printf '\n    %sSome checks failed:%s\n' "$RD" "$N0"
  grep -E '^(FAILED|ERROR) ' "$out" 2>/dev/null | sed -n '1,15p' | sed 's/^/      /' || true
  tail -n 1 "$out" 2>/dev/null | sed 's/^/      /' || true
  if [ "$PUSH" = 0 ]; then
    die "tests failed (--no-push, so nothing would have been published anyway)" "The environment is ready: cd $DEST && $ACTIVATE" "Paste the failures above to Claude."
  fi
  if ask_yn "Publish anyway?"; then warn "publishing although checks failed"; return 0; fi
  die "not publishing because checks failed. Nothing was committed or pushed." \
    "Paste the failures above (or: tail -n 40 ${LOG:-the log}) to Claude, or re-run with --yes to publish anyway."
}

gh_scopes() { gh api -i user 2>/dev/null | tr -d '\r' | awk -F': ' 'tolower($1)=="x-oauth-scopes"{print $2}' || true; }

step_github_signin() {
  step "GitHub sign-in ($OWNER)"
  have gh || die "the GitHub CLI (gh) is not installed" "Install: https://cli.github.com, then re-run."
  if [ -n "${GH_TOKEN:-}${GITHUB_TOKEN:-}" ]; then warn "GH_TOKEN/GITHUB_TOKEN is set in this terminal; gh uses it instead of your login. Run 'unset GH_TOKEN GITHUB_TOKEN' if sign-in looks wrong."; fi
  if ! gh auth status -h github.com >/dev/null 2>&1; then
    info "gh is not signed in. Starting GitHub's browser sign-in (asks for the 'workflow' permission too)."
    info "Sign in as ${B}$OWNER${N0}. If no browser opens, go to https://github.com/login/device and type the code shown."
    run_interactive gh auth login -h github.com -p https -s workflow -w || die "gh sign-in didn't finish" "Run it by hand:  gh auth login -h github.com -p https -s workflow -w   then re-run this script."
  fi
  local login; login=$(gh api user --jq .login 2>/dev/null || true)
  [ -n "$login" ] || die "gh is signed in but can't reach the GitHub API" "Check:  gh auth status   and your network, then re-run."
  if [ "$login" != "$OWNER" ]; then
    local accounts; accounts=$(gh auth status -h github.com 2>&1 || true)
    if [[ $accounts == *"account $OWNER "* || $accounts == *"account $OWNER"$'\n'* ]]; then
      die "gh is using account $login, but the repo belongs to $OWNER" "Switch:  gh auth switch -h github.com -u $OWNER   then re-run this script."
    fi
    die "gh is signed in as $login, but the repo belongs to $OWNER" \
      "Add $OWNER (in a private browser window, or after signing out of $login on github.com):" \
      "    gh auth login -h github.com -p https -s workflow -w" \
      "then:  gh auth switch -h github.com -u $OWNER   and re-run this script."
  fi
  ok "gh signed in as $login"
}

ssh_alias_ok() {
  local h=$1 hn
  [ "$h" != github.com ] || return 0
  hn=$(ssh -G "$h" 2>/dev/null | awk '$1=="hostname"{print $2}' || true)
  [ "$hn" = github.com ] || [ "$hn" = ssh.github.com ]
}
ssh_signs_in_as_owner() {
  local out
  out=$(timeout 25 ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "git@$1" 2>&1 || true)
  info "ssh git@$1: $(printf '%s' "$out" | grep -E 'Hi |denied|resolve|refused|timed out|verification' | sed -n 1p || true)"
  [[ $out == *"Hi $OWNER!"* ]]
}

step_choose_transport() {
  local h hosts=()
  if [ "$FORCE_HTTPS" = 0 ] && have ssh; then
    if [ -n "$SSH_HOST_OPT" ]; then hosts=("$SSH_HOST_OPT"); else hosts=("$DEFAULT_SSH_HOST" github.com); fi
    for h in "${hosts[@]}"; do
      if ! ssh_alias_ok "$h"; then info "ssh: no '$h' host alias in ~/.ssh/config"; continue; fi
      if ssh_signs_in_as_owner "$h"; then TRANSPORT=ssh; SSH_HOST=$h; break; fi
    done
    if [ -z "$TRANSPORT" ] && [ -n "$SSH_HOST_OPT" ]; then
      die "--ssh-host $SSH_HOST_OPT doesn't sign in to GitHub as $OWNER" "Check ~/.ssh/config and the key added to github.com/$OWNER, or use --https."
    fi
  fi
  if [ "$TRANSPORT" = ssh ]; then
    REMOTE_URL="git@$SSH_HOST:$OWNER/$REPO.git"
    ok "push over SSH via $SSH_HOST"
    return
  fi
  TRANSPORT=https
  REMOTE_URL="https://github.com/$OWNER/$REPO.git"
  info "push over HTTPS through gh$( [ "$FORCE_HTTPS" = 1 ] && echo ' (--https)')"
  local scopes; scopes=$(gh_scopes)
  if [ -z "$scopes" ]; then
    warn "couldn't read the token's scopes (fine-grained token?). Pushing .github/workflows needs the 'workflow' permission."
  elif [[ ",${scopes//[[:space:]]/}," != *,workflow,* ]]; then
    info "the gh token can't push workflow files yet (scopes: $scopes). Asking GitHub for the 'workflow' scope;"
    info "a browser/device-code page opens: sign in as $OWNER and approve."
    run_interactive gh auth refresh -h github.com -s workflow || die "couldn't add the workflow scope" "Run by hand:  gh auth refresh -h github.com -s workflow   then re-run."
  fi
  gh auth setup-git -h github.com || die "gh auth setup-git failed"
  ok "git uses gh's credentials for https://github.com"
}

gh_default_email() {
  local e
  e=$(gh api user/emails --jq 'map(select(.primary))[0].email // empty' 2>/dev/null || true)
  [ -n "$e" ] || e=$(gh api user --jq '.email // empty' 2>/dev/null || true)
  [ -n "$e" ] || e=$(gh api user --jq '"\(.id)+\(.login)@users.noreply.github.com"' 2>/dev/null || true)
  printf '%s' "$e"
}

step_identity_commit() {
  step "Git identity + commit"
  [ -d .git ] || { git init -q -b main; info "new git repo in $DEST"; }
  [ "$(realpath "$(git rev-parse --show-toplevel)")" = "$DEST" ] || die "git top level is not $DEST; refusing to commit"
  local name email
  name=$(git config user.name || true); email=$(git config user.email || true)
  if [ -z "$name" ]; then
    name=$(ask_text "Your name for commits" "Aleksander Norman")
    git config --local user.name "$name"; info "set user.name for this repo only"
  fi
  if [ -z "$email" ]; then
    email=$(ask_text "Your email for commits" "$(gh_default_email)")
    [ -n "$email" ] || die "no email for commits" "Set one:  git -C $DEST config user.email you@example.com"
    git config --local user.email "$email"; info "set user.email for this repo only"
  fi
  ok "commits by $name <$email>"
  git add -A
  if git diff --cached --quiet; then
    info "nothing new to commit"
  else
    git commit -q -m "jepa-studio $(project_version): LeJEPA studio (web + desktop)"
    ok "committed $(git rev-parse --short HEAD)"
  fi
  git rev-parse -q --verify HEAD >/dev/null || die "the repo has no commit to push"
}

push_hints() {
  local f=$1
  if grep -q GH007 "$f"; then echo "GitHub blocks pushes that expose a private email. Use your noreply address:"
    echo "    git -C $DEST config user.email '$(gh api user --jq '"\(.id)+\(.login)@users.noreply.github.com"' 2>/dev/null || echo ID+LOGIN@users.noreply.github.com)'"
    echo "    git -C $DEST commit --amend --reset-author --no-edit   then re-run"; fi
  if grep -q -i 'workflow' "$f" && grep -q -i 'refusing\|scope' "$f"; then echo "The token lacks the 'workflow' scope: gh auth refresh -h github.com -s workflow"; fi
  if grep -q -i 'permission denied (publickey)' "$f"; then echo "SSH key not accepted: re-run with --https"; fi
  if grep -q -i 'not found\|denied to' "$f"; then echo "Wrong GitHub account for this repo? Check: gh auth status ; ssh -T git@$DEFAULT_SSH_HOST"; fi
  return 0
}

step_push() {
  step "Push to github.com/$OWNER/$REPO"
  if gh repo view "$OWNER/$REPO" >/dev/null 2>&1; then
    info "repo exists"
  else
    info "creating github.com/$OWNER/$REPO (public)"
    gh repo create "$OWNER/$REPO" --public --description "$DESCRIPTION" --homepage "$PAGES_URL" >/dev/null
  fi
  if git remote get-url origin >/dev/null 2>&1; then git remote set-url origin "$REMOTE_URL"; else git remote add origin "$REMOTE_URL"; fi
  info "origin = $REMOTE_URL"
  if git ls-remote --exit-code --heads origin main >/dev/null 2>&1; then
    git fetch -q origin main
    if ! git merge-base --is-ancestor origin/main HEAD; then
      info "GitHub has commits this copy doesn't: merging them in, keeping this version's files"
      git merge -q -s ours --allow-unrelated-histories -m "Merge GitHub history (keep local tree)" origin/main
    fi
  fi
  local out="$WORK/push.out"
  if git push -u origin HEAD:main 2>&1 | tee "$out"; then
    ok "pushed main"
  else
    mapfile -t hints < <(push_hints "$out")
    die "git push failed" "${hints[@]}"
  fi
}

step_pages() {
  step "GitHub Pages + Actions"
  if gh api -X POST "repos/$OWNER/$REPO/pages" -f build_type=workflow >/dev/null 2>&1; then ok "Pages enabled (source: GitHub Actions)"
  elif gh api -X PUT "repos/$OWNER/$REPO/pages" -f build_type=workflow >/dev/null 2>&1; then ok "Pages source set to GitHub Actions"
  else warn "couldn't set Pages; do it once by hand: repo Settings -> Pages -> Source: GitHub Actions"; fi
  if gh api -X PUT "repos/$OWNER/$REPO/actions/permissions" -F enabled=true -f allowed_actions=all >/dev/null 2>&1; then ok "Actions enabled"
  else warn "couldn't confirm Actions are enabled (Settings -> Actions -> General)"; fi
  for _ in 1 2 3 4 5 6; do                         # a brand-new repo needs a moment to register workflows
    if gh workflow run pages.yml -R "$OWNER/$REPO" --ref main >/dev/null 2>&1; then ok "Pages workflow started"; return 0; fi
    sleep 5
  done
  warn "couldn't start the Pages workflow; start it from the Actions tab (Pages -> Run workflow)"
}

step_tag() {
  step "Desktop installers (optional)"
  local v tag; v=$(project_version); tag="v$v"
  [ -n "$v" ] || { warn "couldn't read the version from jepa_studio/__init__.py"; return 0; }
  if git ls-remote --exit-code --tags origin "refs/tags/$tag" >/dev/null 2>&1; then
    info "tag $tag is already on GitHub; installers were built for it (bump __version__ for a new release)"; return 0
  fi
  if ask_yn "Build desktop installers now (push tag $tag)?"; then
    git rev-parse -q --verify "refs/tags/$tag" >/dev/null || git tag -a "$tag" -m "jepa-studio $tag"
    git push origin "$tag"
    TAG_PUSHED=$tag
    ok "pushed $tag: the desktop workflow builds Windows/macOS/Linux installers (~20-30 min)"
  else
    info "skipped. Later:  cd $DEST && git tag -a $tag -m 'jepa-studio $tag' && git push origin $tag"
  fi
}

summary() {
  printf '\n%sAll set.%s\n' "$GR" "$N0"
  if [ "$PUSH" = 1 ]; then
    cat <<EOF
  Repo:      https://github.com/$OWNER/$REPO
  Website:   $PAGES_URL   (live when the "Pages" workflow finishes, ~1-2 min)
  Web app:   ${PAGES_URL}app.html
  Actions:   https://github.com/$OWNER/$REPO/actions
             gh run list -R $OWNER/$REPO          # recent runs
             gh run watch -R $OWNER/$REPO         # follow one live (pick it from the list)
EOF
    [ -z "$TAG_PUSHED" ] || echo "  Installers: https://github.com/$OWNER/$REPO/releases/tag/$TAG_PUSHED (when the desktop workflow is done)"
  fi
  cat <<EOF
  Locally:   cd $DEST && $ACTIVATE
             jepa-studio doctor
             jepa-studio train --epochs 2 --eval            # a short real PyTorch run
             python -m http.server -d site 8000              # web app at http://localhost:8000/app.html
  Log:       ${LOG:-none}
EOF
}

# ------------------------------------------------------------------------------------ main
main() {
  local a
  while [ $# -gt 0 ]; do
    a=$1
    case "$a" in
      --cpu) CPU=1 ;;
      --browser-tests) BROWSER=1 ;;
      --skip-tests) SKIP_TESTS=1 ;;
      --no-push) PUSH=0 ;;
      --https) FORCE_HTTPS=1 ;;
      --yes | -y) YES=1 ;;
      --doctor) DOCTOR=1 ;;
      --ssh-host) [ $# -ge 2 ] || usage_die "--ssh-host needs a host alias"; SSH_HOST_OPT=$2; shift ;;
      --ssh-host=*) SSH_HOST_OPT=${a#*=} ;;
      --dest) [ $# -ge 2 ] || usage_die "--dest needs a folder"; DEST=$2; shift ;;
      --dest=*) DEST=${a#*=} ;;
      -h | --help) usage; exit 0 ;;
      -*) usage_die "unknown option: $a" ;;
      *) [ -z "$ZIP" ] || usage_die "give one zip, got '$ZIP' and '$a' (quote paths with spaces)"; ZIP=$a ;;
    esac
    shift
  done
  [ -n "$SSH_HOST_OPT" ] && [ "$FORCE_HTTPS" = 1 ] && usage_die "--https and --ssh-host contradict each other"
  [ -n "${HOME:-}" ] && [ -d "$HOME" ] || { echo "HOME is not set to a folder" >&2; exit 1; }

  DEST=$(realpath -m -- "${DEST:-$HOME/$REPO}")
  local home_r; home_r=$(realpath -m -- "$HOME")
  if [ "$DEST" = "$home_r" ] || [ "$DEST" = / ] || [[ $home_r == "$DEST"/* ]]; then
    echo "Refusing to use $DEST as the repo folder: it is your home folder (or contains it)." >&2
    echo "Use the default (~/jepa-studio) or --dest ~/some/new/folder." >&2
    exit 1
  fi

  arm_err_trap
  trap on_exit EXIT
  trap on_int INT

  if [ "$DOCTOR" = 1 ]; then doctor; exit 0; fi

  [ "$PUSH" = 0 ] || STEP_TOTAL=11
  start_log
  printf '%sjepa-studio setup%s  %s\n' "$B" "$N0" "$(date '+%Y-%m-%d %H:%M:%S')"
  step_preflight
  step_find_zip
  step_packages
  step_unpack
  step_python
  step_tests
  if [ "$PUSH" = 0 ]; then
    info "--no-push: stopping before git/GitHub"
    summary; return 0
  fi
  step_github_signin
  step_choose_transport
  step_identity_commit
  step_push
  step_pages
  step_tag
  summary
}

main "$@"
exit $?
