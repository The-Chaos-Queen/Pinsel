#!/bin/bash
# One-time setup of a porter's build environment on ML-WS.
#   bash agent_setup.sh NAME BRANCH
# Clones the branch from the bare repo into ~/pinsel-agents/NAME and creates
# a venv there with numpy/scipy/numba/pillow/pytest/maturin and pinsel
# installed editable. Safe to re-run (it pulls instead of cloning).
set -euo pipefail
NAME=${1:?name}; BRANCH=${2:?branch}
DIR=~/pinsel-agents/$NAME
mkdir -p ~/pinsel-agents
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch -q origin && git -C "$DIR" checkout -q "$BRANCH" && git -C "$DIR" pull -q
else
  git clone -q -b "$BRANCH" ~/git/pinsel.git "$DIR"
fi
cd "$DIR"
[ -x venv/bin/python ] || python3 -m venv venv
venv/bin/pip install -q --upgrade pip >/dev/null
venv/bin/pip install -q numpy scipy numba pillow pytest "maturin>=1.9,<2"
venv/bin/pip install -q -e .
echo "ready: $DIR (branch $BRANCH, venv $DIR/venv)"
