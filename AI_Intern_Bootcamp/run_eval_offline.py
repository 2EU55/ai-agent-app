import csv
from pathlib import Path


def fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def pass_(msg: str) -> None:
    print(f"[PASS] {msg}")


def read_sales(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "date": r.get("日期", ""),
                    "product": r.get("产品", ""),
                    "sales": int(float(r.get("销售额", "0") or 0)),
                    "cost": int(float(r.get("成本", "0") or 0)),
                    "profit": int(float(r.get("利润", "0") or 0)),
                }
            )
    return rows


def main() -> int:
    root = Path(__file__).resolve().parent
    sales_path = root / "sales_data.csv"
    policy_path = root / "company_policy.txt"

    if not sales_path.exists():
        return fail("缺少 sales_data.csv")
    if not policy_path.exists():
        return fail("缺少 company_policy.txt")

    rows = read_sales(sales_path)
    if not rows:
        return fail("sales_data.csv 为空")
    pass_("加载 sales_data.csv")

    total_sales = sum(r["sales"] for r in rows)
    if total_sales != 8600:
        return fail(f"销售额总和异常：{total_sales} != 8600")
    pass_("销售额总和=8600")

    max_sales_row = max(rows, key=lambda r: r["sales"])
    if max_sales_row["date"] != "2024-01-06" or max_sales_row["sales"] != 3000:
        return fail(f"销售额最高异常：{max_sales_row}")
    pass_("销售额最高=2024-01-06 / 3000")

    max_profit_row = max(rows, key=lambda r: r["profit"])
    if max_profit_row["date"] != "2024-01-06" or max_profit_row["profit"] != 2500:
        return fail(f"利润最高异常：{max_profit_row}")
    pass_("利润最高=2024-01-06 / 2500")

    sales_sorted = sorted(r["sales"] for r in rows)
    median_sales = sales_sorted[len(sales_sorted) // 2]
    if median_sales != 1000:
        return fail(f"销售额中位数异常：{median_sales} != 1000")
    pass_("销售额中位数=1000")

    by_product: dict[str, int] = {}
    for r in rows:
        by_product[r["product"]] = by_product.get(r["product"], 0) + r["sales"]
    if by_product.get("AI课程") != 4500 or by_product.get("咨询服务") != 3000 or by_product.get("Python书籍") != 1100:
        return fail(f"按产品汇总异常：{by_product}")
    pass_("按产品汇总销售额=AI课程4500/咨询服务3000/Python书籍1100")

    policy = policy_path.read_text(encoding="utf-8")
    required_snippets = ["600 元", "400 元", "每天补助 100 元", "5 天年假", "7 天年假", "10 天年假", "每月 10 日", "20% - 30%"]
    missing = [s for s in required_snippets if s not in policy]
    if missing:
        return fail(f"政策文本缺少关键片段：{missing}")
    pass_("政策文本包含关键条款")

    print("\nTotal: 6 | Passed: 6 | Failed: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

