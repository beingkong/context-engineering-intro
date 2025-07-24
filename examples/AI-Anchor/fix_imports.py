#!/usr/bin/env python3
"""Script to fix relative imports to absolute imports in the src directory."""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix relative imports in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace relative imports with absolute imports
    # Pattern: from ..something import -> from src.something import
    # Pattern: from ...something import -> from src.something import (for deeper levels)
    
    content = re.sub(r'from \.\.\.([^.][^\s]+) import', r'from src.\1 import', content)
    content = re.sub(r'from \.\.([^.][^\s]+) import', r'from src.\1 import', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed imports in {file_path}")

def main():
    """Fix imports in all Python files in src directory."""
    src_dir = Path("src")
    
    for py_file in src_dir.rglob("*.py"):
        if py_file.name != "__init__.py":  # Skip __init__.py files
            try:
                fix_imports_in_file(py_file)
            except Exception as e:
                print(f"Error fixing {py_file}: {e}")

if __name__ == "__main__":
    main()