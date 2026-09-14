# Antigravity Agent Instructions & Guidelines

## Git Commit Rules

Whenever the agent (Antigravity) crafts a Git commit message, always append two newlines and the co-author attribution trailer at the end of the commit message:

```text
<commit summary>

<optional body description>

Co-authored-by: Antigravity Agent <antigravity-bot@google.internal>
```

### Commit Formatting Pattern
1. Use conventional commit formatting (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, etc.).
2. Two newlines (`\n\n`) before the co-author signature.
3. Signature line:
   `Co-authored-by: Antigravity Agent <antigravity-bot@google.internal>`

This distinguishes commits made by or in collaboration with the Antigravity assistant while keeping the user's primary GitHub identity as the commit author.
