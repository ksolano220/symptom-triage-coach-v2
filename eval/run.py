import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference import triage, triage_text_only  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"


def run_case(case):
    image_path = EVAL_DIR / case["image"]
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    mm = triage(image_bytes, case["symptom"])
    txt = triage_text_only(case["symptom"])
    return {"id": case["id"], "case": case, "image_text": mm, "text_only": txt}


def causes_str(result):
    return "; ".join(c["name"] for c in result["possible_causes"][:3])


def print_summary(rows):
    print()
    print("| id | path | top 3 causes | systems |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r['id']} | image+text | {causes_str(r['image_text'])} | {', '.join(r['image_text']['systems'])} |")
        print(f"| {r['id']} | text-only  | {causes_str(r['text_only'])} | {', '.join(r['text_only']['systems'])} |")
    print()


def main():
    cases = json.loads((EVAL_DIR / "cases.json").read_text())
    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    for case in cases:
        print(f"running {case['id']}...")
        rows.append(run_case(case))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"{stamp}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")

    print_summary(rows)


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set")
        sys.exit(1)
    main()
