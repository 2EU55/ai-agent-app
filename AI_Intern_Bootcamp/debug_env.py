import json
import os
import urllib.request

from dotenv import load_dotenv


def main() -> int:
    load_dotenv(dotenv_path="/app/.env", override=True)
    key = os.getenv("SILICONFLOW_API_KEY") or ""
    base_url = os.getenv("BASE_URL") or ""
    print("KEY_PREFIX=", key[:6])
    print("BASE_URL=", base_url)
    if not key or not base_url:
        return 2

    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            first = (((data or {}).get("data") or [])[:1] or [{}])[0].get("id")
            print("MODELS_OK=", bool(first))
            print("FIRST_MODEL_ID=", first)
            return 0
    except Exception as e:
        print("MODELS_ERROR=", str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
