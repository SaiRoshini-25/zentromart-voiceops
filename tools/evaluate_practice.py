"""Run the complete supplied question and multi-turn conversation evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import evaluate_practice

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", default="data", help="Directory containing the CSV and KB data")
parser.add_argument(
    "--practice",
    default="Capstone_Project_Dataset_FTTT_2026/practice",
    help="Directory containing questions.csv and conversations.csv",
)
parser.add_argument("--output", default="outputs/evaluation")
args = parser.parse_args()

print(json.dumps(
    evaluate_practice(args.data, args.practice, args.output),
    indent=2,
    ensure_ascii=False,
))
