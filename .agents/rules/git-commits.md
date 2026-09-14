# Git Commit Signing Rule

Whenever Antigravity generates or executes a Git commit, always end the commit message with two newlines and the co-author signature:

```text
Co-authored-by: Antigravity Agent <antigravity-bot@google.internal>
```

Example:
```bash
git commit -m "$(cat <<'EOF'
feat(analytics): add anomaly detection model

Implemented rMSSD outlier detection baseline.

Co-authored-by: Antigravity Agent <antigravity-bot@google.internal>
EOF
)"
```
