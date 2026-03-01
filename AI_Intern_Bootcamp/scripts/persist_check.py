import os
import sys
from pathlib import Path


def main() -> int:
    action = (sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()
    runtime_dir = os.getenv("RUNTIME_DIR") or "/data"
    p = Path(runtime_dir) / "persist.txt"

    if action == "write":
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("persist_test", encoding="utf-8")
        print(str(p))
        return 0

    if action == "read":
        if not p.exists():
            print("missing")
            return 2
        print(p.read_text(encoding="utf-8", errors="replace"))
        return 0

    print("usage: persist_check.py write|read")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
