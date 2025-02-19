import sys
import os
from pathlib import Path

def debug_paths():
    print("\nPython Path:")
    for p in sys.path:
        print(f"  {p}")
    
    print("\nCurrent Directory:")
    print(f"  {os.getcwd()}")
    
    print("\nProject Structure:")
    project_root = Path(__file__).parent.parent
    for p in project_root.rglob('__init__.py'):
        print(f"  {p.relative_to(project_root)}")

if __name__ == "__main__":
    debug_paths() 