import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('c:/All/semester 8/CSE 400 - thesis/v2/thesis_v2_implementation_sir_proposed.ipynb', encoding='utf-8') as f:
    d = json.load(f)
for i, cell in enumerate(d.get('cells', [])):
    source = "".join(cell.get('source', []))
    print(f"--- Cell {i} ({cell.get('cell_type')}) ---")
    print(source[:200] + ("..." if len(source) > 200 else ""))
    print()
