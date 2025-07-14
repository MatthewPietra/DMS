#!/usr/bin/env python3
"""
Script to automatically fix common linting issues.
"""

import os
import re
import sys
from pathlib import Path


def fix_file(file_path):
    """Fix common linting issues in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix trailing whitespace
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
    
    # Fix blank lines with whitespace
    content = re.sub(r'^\s+$', '', content, flags=re.MULTILINE)
    
    # Fix line length issues (basic approach)
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        if len(line) > 79 and not line.startswith('#'):
            # Try to break long lines at logical points
            if 'import' in line and len(line) > 79:
                # Handle long import lines
                if 'from' in line and 'import' in line:
                    parts = line.split('import')
                    if len(parts) == 2:
                        from_part = parts[0].strip()
                        import_part = parts[1].strip()
                        if len(from_part) + len(import_part) > 70:
                            line = f"{from_part}\nimport {import_part}"
            elif '(' in line and ')' in line and len(line) > 79:
                # Try to break function calls
                if line.count('(') == line.count(')'):
                    # Simple case: single function call
                    match = re.match(r'(\s*)(\w+)\s*\((.*)\)', line)
                    if match:
                        indent, func_name, args = match.groups()
                        if len(args) > 50:
                            # Break into multiple lines
                            args_parts = args.split(',')
                            if len(args_parts) > 1:
                                new_args = ',\n'.join([f"{indent}    {arg.strip()}" for arg in args_parts])
                                line = f"{indent}{func_name}(\n{new_args}\n{indent})"
        
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    # Remove unused imports (basic approach)
    lines = content.split('\n')
    import_lines = []
    other_lines = []
    in_import_block = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')):
            in_import_block = True
            import_lines.append(line)
        elif in_import_block and stripped == '':
            import_lines.append(line)
        elif in_import_block and not stripped.startswith(('import ', 'from ')):
            in_import_block = False
            other_lines.append(line)
        else:
            other_lines.append(line)
    
    # Filter out obviously unused imports
    filtered_imports = []
    for import_line in import_lines:
        if import_line.strip():
            # Keep the import for now - we'll need more sophisticated analysis
            filtered_imports.append(import_line)
    
    # Reconstruct content
    if filtered_imports:
        content = '\n'.join(filtered_imports) + '\n\n' + '\n'.join(other_lines)
    else:
        content = '\n'.join(other_lines)
    
    # Fix f-string issues
    content = re.sub(r'f"([^"]*)"', r'"\1"', content)  # Remove f-strings without placeholders
    
    # Fix lambda assignments
    content = re.sub(r'(\w+)\s*=\s*lambda\s*([^:]+):\s*([^,\n]+)', 
                    r'def \1(\2):\n    return \3', content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def main():
    """Main function to fix linting issues."""
    src_dir = Path('src')
    tests_dir = Path('tests')
    
    fixed_files = []
    
    # Process source files
    for py_file in src_dir.rglob('*.py'):
        if fix_file(py_file):
            fixed_files.append(str(py_file))
    
    # Process test files
    for py_file in tests_dir.rglob('*.py'):
        if fix_file(py_file):
            fixed_files.append(str(py_file))
    
    if fixed_files:
        print(f"Fixed {len(fixed_files)} files:")
        for file in fixed_files:
            print(f"  - {file}")
    else:
        print("No files needed fixing.")


if __name__ == '__main__':
    main() 