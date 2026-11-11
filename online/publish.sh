#!/bin/sh
# Rebuild the online demo and publish it to the gh-pages branch, which GitHub Pages serves.
#     sh online/publish.sh
set -e
cd "$(dirname "$0")/.."
REPO=$(git remote get-url origin | sed -e 's/\.git$//')
.venv/bin/python online/build.py --repo "$REPO"
TMP=$(mktemp -d)
git worktree add -q "$TMP" gh-pages 2>/dev/null || git worktree add -q --orphan -b gh-pages "$TMP"
find "$TMP" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -R online/site/. "$TMP"/
git -C "$TMP" add -A
git -C "$TMP" commit -q -m "Online demo built from $(git rev-parse --short HEAD)" || echo "online demo unchanged"
git -C "$TMP" push -q origin gh-pages
git worktree remove --force "$TMP"
echo "published; GitHub Pages updates within a minute or two"
