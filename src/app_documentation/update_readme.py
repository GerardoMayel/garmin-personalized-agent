#!/usr/bin/env python3
"""Project Structure Mapper and README Generator.

This script scans the repository structure, generates a formatted tree representation
(without descriptions by default), and creates an updated 'newreadme.md' in the
app_documentation folder for review before manually updating the root README.md.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Default directories and files to exclude from the mapped tree
DEFAULT_EXCLUDED_DIRS: Set[str] = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    ".idea",
    ".vscode",
    ".cache",
    "qdrant_storage",
    "chroma_db",
    ".chroma",
    "wandb",
    "htmlcov",
    "build",
    "dist",
    ".eggs",
}

DEFAULT_EXCLUDED_FILES: Set[str] = {
    ".DS_Store",
    "*.pyc",
    "*.pyo",
    "*.swp",
    "*.swo",
    "*~",
    ".coverage",
}

# Subdirectories under which certain files should be hidden (e.g. .gitkeep in empty tracking dirs)
HIDE_GITKEEP: bool = True

# Standard top-level ordering to maintain clean logical grouping in README
TOP_LEVEL_ORDER: List[str] = [
    ".github",
    "deploy",
    "configs",
    "data",
    "notebooks",
    "src",
    "tests",
    ".dvc",
    ".dvcignore",
    ".env.example",
    ".gitignore",
    "LICENSE",
    ".pre-commit-config.yaml",
    "pyproject.toml",
    "uv.lock",
    "README.md",
]


def find_repo_root(start_path: Optional[Path] = None) -> Path:
    """Locate the root of the repository by finding .git, pyproject.toml, or src parent."""
    curr = (start_path or Path(__file__)).resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    # Fallback to grandparent of this script (assuming src/app_documentation/script.py)
    return Path(__file__).resolve().parent.parent.parent


def is_ignored(
    path: Path,
    repo_root: Path,
    exclude_dirs: Set[str],
    exclude_files: Set[str],
    include_gitkeep: bool = False,
) -> bool:
    """Check if a file or folder should be ignored during mapping."""
    name = path.name

    # Check symlinks (skip broken symlinks)
    if path.is_symlink():
        try:
            if not path.resolve().exists():
                return True
        except Exception:
            return True

    # Check excluded directory names
    if path.is_dir():
        if name in exclude_dirs:
            return True
        # Exclude DVC internal storage
        rel = path.relative_to(repo_root)
        if str(rel).startswith(".dvc/cache") or str(rel).startswith(".dvc/tmp"):
            return True
        return False

    # Check excluded files
    if name in exclude_files or name == ".DS_Store":
        return True
    for pat in exclude_files:
        if pat.startswith("*") and name.endswith(pat[1:]):
            return True

    # Hide placeholder .gitkeep unless explicitly asked
    if not include_gitkeep and name == ".gitkeep":
        return True

    return False


def get_sort_key(item: Path, top_level: bool = False) -> Tuple[int, int, str]:
    """Sort items: prioritized top-level order, directories first, then alphabetical."""
    name = item.name
    is_dir = 0 if item.is_dir() else 1

    if top_level and name in TOP_LEVEL_ORDER:
        priority = TOP_LEVEL_ORDER.index(name)
    else:
        priority = 1000

    return (priority, is_dir, name.lower())


def parse_existing_descriptions(readme_content: str) -> Dict[str, str]:
    """Parse comments from the existing README tree (e.g. `file.py # comment`)."""
    descriptions: Dict[str, str] = {}
    pattern = re.compile(r"^[│\s├└─]+([A-Za-z0-9_\-\./📊📈🖼️🤖]+(?:\.[a-zA-Z0-9]+)?)\s+#\s*(.+)$")

    for line in readme_content.splitlines():
        match = pattern.match(line)
        if match:
            item_name = match.group(1).strip().rstrip("/")
            comment = match.group(2).strip()
            descriptions[item_name] = comment

    return descriptions


def build_tree_nodes(
    dir_path: Path,
    repo_root: Path,
    exclude_dirs: Set[str],
    exclude_files: Set[str],
    include_gitkeep: bool,
    top_level: bool = False,
) -> List[Path]:
    """Retrieve filtered and sorted child paths for a directory."""
    try:
        children = [
            child
            for child in dir_path.iterdir()
            if not is_ignored(child, repo_root, exclude_dirs, exclude_files, include_gitkeep)
        ]
    except (PermissionError, FileNotFoundError):
        return []

    children.sort(key=lambda p: get_sort_key(p, top_level=top_level))
    return children


def generate_tree_lines(
    dir_path: Path,
    repo_root: Path,
    prefix: str = "",
    is_last: bool = True,
    is_root: bool = True,
    project_name: str = "garmin-personal-insight-agent/",
    exclude_dirs: Set[str] = DEFAULT_EXCLUDED_DIRS,
    exclude_files: Set[str] = DEFAULT_EXCLUDED_FILES,
    include_gitkeep: bool = False,
    descriptions: Optional[Dict[str, str]] = None,
    with_spacers: bool = True,
) -> List[str]:
    """Recursively generate tree lines with unicode branches."""
    lines: List[str] = []

    if is_root:
        root_label = project_name if project_name.endswith("/") else f"{project_name}/"
        lines.append(root_label)
        children = build_tree_nodes(
            dir_path,
            repo_root,
            exclude_dirs,
            exclude_files,
            include_gitkeep,
            top_level=True,
        )
        for idx, child in enumerate(children):
            child_is_last = idx == len(children) - 1
            lines.extend(
                generate_tree_lines(
                    child,
                    repo_root,
                    prefix="",
                    is_last=child_is_last,
                    is_root=False,
                    project_name=project_name,
                    exclude_dirs=exclude_dirs,
                    exclude_files=exclude_files,
                    include_gitkeep=include_gitkeep,
                    descriptions=descriptions,
                    with_spacers=with_spacers,
                )
            )
            # Add spacer line between major top-level items if requested and not last
            if with_spacers and child.is_dir() and not child_is_last:
                lines.append("│")
        return lines

    # Formatting current item
    connector = "└── " if is_last else "├── "
    name = dir_path.name + ("/" if dir_path.is_dir() else "")
    line = f"{prefix}{connector}{name}"

    # Append description if requested and available
    if descriptions:
        desc_key = dir_path.name
        if desc_key in descriptions:
            padding = max(1, 35 - len(line))
            line = f"{line}{' ' * padding}# {descriptions[desc_key]}"

    lines.append(line)

    # Recurse into subdirectories
    if dir_path.is_dir():
        new_prefix = prefix + ("    " if is_last else "│   ")
        sub_children = build_tree_nodes(
            dir_path,
            repo_root,
            exclude_dirs,
            exclude_files,
            include_gitkeep,
            top_level=False,
        )
        for idx, sub_child in enumerate(sub_children):
            sub_is_last = idx == len(sub_children) - 1
            lines.extend(
                generate_tree_lines(
                    sub_child,
                    repo_root,
                    prefix=new_prefix,
                    is_last=sub_is_last,
                    is_root=False,
                    project_name=project_name,
                    exclude_dirs=exclude_dirs,
                    exclude_files=exclude_files,
                    include_gitkeep=include_gitkeep,
                    descriptions=descriptions,
                    with_spacers=False,  # spacers only at top level
                )
            )

    return lines


def update_readme_content(original_readme: str, new_tree_str: str) -> str:
    """Replace or insert the project structure tree inside the README content."""
    header_pattern = re.compile(
        r"^##\s+(?:Estructura del Proyecto|Project Structure)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    match = header_pattern.search(original_readme)
    if not match:
        return f"{original_readme.rstrip()}\n\n## Estructura del Proyecto\n\n{new_tree_str}\n"

    header_end = match.end()

    # Look for the next top-level or second-level heading (e.g. "\n## ") after this header
    next_heading_pattern = re.compile(r"\n(?=##\s+[^\n]+)", re.MULTILINE)
    next_match = next_heading_pattern.search(original_readme, pos=header_end)

    before_header = original_readme[:header_end].rstrip()
    after_section = (
        "\n\n" + original_readme[next_match.start() :].lstrip("\n")
        if next_match
        else ""
    )

    return f"{before_header}\n\n{new_tree_str}\n{after_section}".rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Map project path hierarchy and generate updated newreadme.md."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root path (defaults to auto-detected root)",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="garmin-personal-insight-agent/",
        help="Name of the root folder shown in the tree",
    )
    parser.add_argument(
        "--output-readme",
        type=Path,
        default=None,
        help="Output path for newreadme.md (defaults to src/app_documentation/newreadme.md)",
    )
    parser.add_argument(
        "--output-tree",
        type=Path,
        default=None,
        help="Output path for plain text tree (defaults to src/app_documentation/project_structure.txt)",
    )
    parser.add_argument(
        "--apply-to-root",
        action="store_true",
        help="Also overwrite the root README.md directly (default: false, leaves it for manual review)",
    )
    parser.add_argument(
        "--with-descriptions",
        action="store_true",
        help="Retain comments and descriptions from existing README in the tree",
    )
    parser.add_argument(
        "--no-spacers",
        action="store_true",
        help="Omit spacer lines between top-level directories",
    )
    parser.add_argument(
        "--include-gitkeep",
        action="store_true",
        help="Include .gitkeep placeholder files in tree",
    )

    args = parser.parse_args()

    repo_root = find_repo_root(args.root)
    doc_dir = repo_root / "src" / "app_documentation"
    doc_dir.mkdir(parents=True, exist_ok=True)

    output_readme_path = args.output_readme or (doc_dir / "newreadme.md")
    output_tree_path = args.output_tree or (doc_dir / "project_structure.txt")
    root_readme_path = repo_root / "README.md"

    print(f"[*] Repository root: {repo_root}")
    print(f"[*] Scanning paths and building tree hierarchy...")

    descriptions = None
    if args.with_descriptions and root_readme_path.exists():
        original_text = root_readme_path.read_text(encoding="utf-8")
        descriptions = parse_existing_descriptions(original_text)
        print(f"[*] Extracted {len(descriptions)} description annotations from existing README.")

    # Generate tree lines (without descriptions by default, matching request)
    tree_lines = generate_tree_lines(
        dir_path=repo_root,
        repo_root=repo_root,
        project_name=args.project_name,
        include_gitkeep=args.include_gitkeep,
        descriptions=descriptions,
        with_spacers=not args.no_spacers,
    )
    tree_str = "\n".join(tree_lines)

    # Save isolated tree text file
    output_tree_path.write_text(tree_str + "\n", encoding="utf-8")
    print(f"[+] Saved project structure tree to: {output_tree_path}")

    # Read existing README.md if present
    if root_readme_path.exists():
        original_readme = root_readme_path.read_text(encoding="utf-8")
        updated_readme = update_readme_content(original_readme, tree_str)
    else:
        updated_readme = f"# {args.project_name}\n\n## Estructura del Proyecto\n\n{tree_str}\n"

    # Write to newreadme.md
    output_readme_path.write_text(updated_readme, encoding="utf-8")
    print(f"[+] Successfully generated new README: {output_readme_path}")

    if args.apply_to_root and root_readme_path.exists():
        root_readme_path.write_text(updated_readme, encoding="utf-8")
        print(f"[+] Overwrote root README: {root_readme_path}")
    else:
        print(f"[*] Root README.md was NOT overwritten (ready for your manual update).")

    print("\n--- Project Tree Preview ---")
    # Show first 40 lines of tree preview
    preview_lines = tree_lines[:40]
    print("\n".join(preview_lines))
    if len(tree_lines) > 40:
        print(f"... ({len(tree_lines) - 40} more lines in {output_tree_path.name})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
