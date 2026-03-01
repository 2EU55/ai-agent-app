import os
import runpy
import sys


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(root, "AI_Intern_Bootcamp", "rag_gen_eval.py"),
        os.path.join(root, "ai_intern_bootcamp", "rag_gen_eval.py"),
    ]
    for p in candidates:
        if os.path.exists(p):
            sys.path.insert(0, os.path.dirname(p))
            runpy.run_path(p, run_name="__main__")
            return
    raise FileNotFoundError("找不到 rag_gen_eval.py（请确认 AI_Intern_Bootcamp 或 ai_intern_bootcamp 目录存在）")


if __name__ == "__main__":
    main()
