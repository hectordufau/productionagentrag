import json
from pathlib import Path

Path("datasets/sample").mkdir(parents=True, exist_ok=True)
Path("datasets/sample/knowledge.md").write_text(
    "# Sample\nQdrant stores vectors; PostgreSQL stores metadata.\n", encoding="utf-8"
)

print(
    json.dumps(
        {"status": "placeholder", "reason": "M1 benchmark harness will be implemented in M3"}
    )
)
