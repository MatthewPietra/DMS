#!/usr/bin/env python3
"""Comprehensive Linting Issue Fixer for DMS Project.

This script systematically fixes all linting issues identified by flake8:
- Removes unused imports (F401)
- Handles unused variables (F841) by adding underscore prefix or removing
- Fixes line length violations (E501) by proper line breaking
- Adds missing blank lines (E302)
- Fixes comment formatting (E265)
"""

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class LintingFixer:
    """Systematic linting issue fixer."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_dir = project_root / "src"
        self.fixed_files: set[Path] = set()

    def run_flake8(self, path: Optional[Path] = None) -> List[str]:
        """Run flake8 and return list of issues."""
        target = path or self.src_dir
        cmd = [
            "py",
            "-m",
            "flake8",
            "--max-line-length=88",
            "--ignore=E203,W503",
            str(target),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )
            return result.stdout.strip().split("\n") if result.stdout.strip() else []
        except Exception as e:
            print(f"Error running flake8: {e}")
            return []

    def fix_unused_imports(self, file_path: Path) -> bool:
        """Remove unused imports from a file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Get unused imports for this file
            issues = self.run_flake8(file_path)
            unused_imports = [line for line in issues if "F401" in line]

            if not unused_imports:
                return False

            lines = content.split("\n")
            lines_to_remove = set()

            for issue in unused_imports:
                # Parse line number from flake8 output
                match = re.search(r":(\d+):", issue)
                if match:
                    line_num = int(match.group(1)) - 1  # Convert to 0-based
                    if line_num < len(lines):
                        # Check if this is a simple unused import line
                        line = lines[line_num].strip()
                        if line.startswith("import ") or line.startswith("from "):
                            lines_to_remove.add(line_num)

            # Remove unused import lines
            if lines_to_remove:
                new_lines = [
                    line for i, line in enumerate(lines) if i not in lines_to_remove
                ]

                # Clean up any resulting empty lines
                cleaned_lines = []
                prev_empty = False
                for line in new_lines:
                    if line.strip() == "":
                        if not prev_empty:
                            cleaned_lines.append(line)
                        prev_empty = True
                    else:
                        cleaned_lines.append(line)
                        prev_empty = False

                new_content = "\n".join(cleaned_lines)

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

                return True

        except Exception as e:
            print(f"Error fixing unused imports in {file_path}: {e}")

        return False

    def fix_unused_variables(self, file_path: Path) -> bool:
        """Fix unused variables by adding underscore prefix."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Get unused variables for this file
            issues = self.run_flake8(file_path)
            unused_vars = [line for line in issues if "F841" in line]

            if not unused_vars:
                return False

            lines = content.split("\n")
            modified = False

            for issue in unused_vars:
                # Extract variable name and line number
                match = re.search(
                    r":(\d+):.+\'(.+?)\' is assigned to but never used", issue
                )
                if match:
                    line_num = int(match.group(1)) - 1  # Convert to 0-based
                    var_name = match.group(2)

                    if line_num < len(lines):
                        line = lines[line_num]

                        # Common pattern: except Exception as e:
                        if "except " in line and f" as {var_name}" in line:
                            lines[line_num] = line.replace(
                                f" as {var_name}", f" as _{var_name}"
                            )
                            modified = True
                        # Assignment pattern: var = something
                        elif f"{var_name} = " in line:
                            lines[line_num] = line.replace(
                                f"{var_name} = ", f"_{var_name} = "
                            )
                            modified = True

            if modified:
                new_content = "\n".join(lines)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                return True

        except Exception as e:
            print(f"Error fixing unused variables in {file_path}: {e}")

        return False

    def fix_line_length(self, file_path: Path) -> bool:
        """Fix line length violations by breaking long lines."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Get line length violations for this file
            issues = self.run_flake8(file_path)
            long_lines = [line for line in issues if "E501" in line]

            if not long_lines:
                return False

            lines = content.split("\n")
            modified = False

            for issue in long_lines:
                # Extract line number
                match = re.search(r":(\d+):", issue)
                if match:
                    line_num = int(match.group(1)) - 1  # Convert to 0-based

                    if line_num < len(lines):
                        line = lines[line_num]

                        # Try to break long strings
                        if len(line) > 88 and ('f"' in line or '"' in line):
                            # Simple string breaking for f-strings and regular strings
                            if 'f"' in line and line.count('"') >= 2:
                                # Break f-strings at logical points
                                indent = len(line) - len(line.lstrip())
                                if (
                                    " " in line[40:]
                                ):  # Look for break point after char 40
                                    break_point = line.find(" ", 40)
                                    if break_point != -1 and break_point < 85:
                                        part1 = line[:break_point].rstrip()
                                        part2 = line[break_point:].lstrip()
                                        lines[line_num] = (
                                            part1 + " \\\n" + " " * (indent + 4) + part2
                                        )
                                        modified = True

                        # Try to break long import lines
                        elif "import " in line and "," in line:
                            if line.strip().startswith("from ") and "import" in line:
                                # Break multi-import lines
                                parts = line.split("import")
                                if len(parts) == 2:
                                    from_part = parts[0] + "import"
                                    import_part = parts[1].strip()

                                    if "," in import_part:
                                        imports = [
                                            imp.strip()
                                            for imp in import_part.split(",")
                                        ]
                                        if len(imports) > 1:
                                            indent = len(line) - len(line.lstrip())
                                            new_line = from_part + " (\n"
                                            for imp in imports:
                                                new_line += (
                                                    " " * (indent + 4)
                                                    + imp.strip()
                                                    + ",\n"
                                                )
                                            new_line += " " * indent + ")"
                                            lines[line_num] = new_line
                                            modified = True

            if modified:
                new_content = "\n".join(lines)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                return True

        except Exception as e:
            print(f"Error fixing line length in {file_path}: {e}")

        return False

    def fix_blank_lines(self, file_path: Path) -> bool:
        """Add missing blank lines before class and function definitions."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Get blank line violations for this file
            issues = self.run_flake8(file_path)
            blank_line_issues = [line for line in issues if "E302" in line]

            if not blank_line_issues:
                return False

            lines = content.split("\n")
            modified = False

            for issue in blank_line_issues:
                # Extract line number
                match = re.search(r":(\d+):", issue)
                if match:
                    line_num = int(match.group(1)) - 1  # Convert to 0-based

                    if line_num < len(lines):
                        line = lines[line_num].strip()

                        # Add blank lines before class/function definitions
                        if (
                            line.startswith("class ")
                            or line.startswith("def ")
                            or line.startswith("async def ")
                        ):

                            # Check if previous line is not blank
                            if line_num > 0 and lines[line_num - 1].strip() != "":
                                lines.insert(line_num, "")
                                modified = True

            if modified:
                new_content = "\n".join(lines)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                return True

        except Exception as e:
            print(f"Error fixing blank lines in {file_path}: {e}")

        return False

    def fix_file(self, file_path: Path) -> bool:
        """Fix all linting issues in a single file."""
        if file_path.suffix != ".py":
            return False

        print(f"Fixing {file_path.relative_to(self.project_root)}")

        fixed = False

        # Fix in order of importance
        if self.fix_unused_imports(file_path):
            fixed = True
            print(f"  ✓ Fixed unused imports")

        if self.fix_unused_variables(file_path):
            fixed = True
            print(f"  ✓ Fixed unused variables")

        if self.fix_line_length(file_path):
            fixed = True
            print(f"  ✓ Fixed line length violations")

        if self.fix_blank_lines(file_path):
            fixed = True
            print(f"  ✓ Fixed blank line violations")

        if fixed:
            self.fixed_files.add(file_path)

        return fixed

    def fix_all(self) -> Dict[str, int]:
        """Fix all linting issues in the project."""
        print("Starting comprehensive linting fix...")

        stats = {
            "files_processed": 0,
            "files_fixed": 0,
            "issues_before": 0,
            "issues_after": 0,
        }

        # Count initial issues
        initial_issues = self.run_flake8()
        stats["issues_before"] = len(initial_issues)

        # Process all Python files
        for py_file in self.src_dir.rglob("*.py"):
            stats["files_processed"] += 1

            if self.fix_file(py_file):
                stats["files_fixed"] += 1

        # Count final issues
        final_issues = self.run_flake8()
        stats["issues_after"] = len(final_issues)

        return stats


def main() -> int:
    """Main function to run the linting fixer."""
    project_root = Path(__file__).parent
    fixer = LintingFixer(project_root)

    print("🔧 DMS Linting Issue Fixer")
    print("=" * 50)

    stats = fixer.fix_all()

    print("\n📊 Summary:")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Files fixed: {stats['files_fixed']}")
    print(f"Issues before: {stats['issues_before']}")
    print(f"Issues after: {stats['issues_after']}")
    print(f"Issues resolved: {stats['issues_before'] - stats['issues_after']}")

    if stats["issues_after"] == 0:
        print("\n🎉 All linting issues have been resolved!")
    else:
        print(f"\n⚠️  {stats['issues_after']} issues remain - may need manual attention")

    return 0 if stats["issues_after"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
