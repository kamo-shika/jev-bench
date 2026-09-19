"""JNLI をハーネスの入力 JSONL に変換する。

測定用は valid から 1,000 件、温度の検証用は train から 500 件。別の split なので重複しない。
出力は data/ に置く（JGLUE は CC BY-SA 4.0。継承条件があるので変換物はコミットしない）。

  python tools/jnli_to_jsonl.py [--jglue data/JGLUE] [--out data] [--seed 20260919]
"""

import argparse
import json
import os
import random

INSTRUCTIONS = "次の前提文と仮説文の関係として最も適切なものを選んでください。"

CRITERIA = {
    "entailment": "含意。前提文が正しいとき、仮説文も必ず正しい",
    "contradiction": "矛盾。前提文が正しいとき、仮説文は正しくない",
    "neutral": "どちらでもない。前提文からは仮説文が正しいかどうか決まらない",
}


def to_item(row, split):
    # sentence_pair_id は split ごとに振り直されていて train と valid で重複するので、
    # split を前に付けて区別できるようにする
    return {
        "id": "jnli-%s-%s" % (split, row["sentence_pair_id"]),
        "state": "前提文: %s\n仮説文: %s" % (row["sentence1"], row["sentence2"]),
        "questions": [
            {
                "id": "q1",
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": dict(CRITERIA),
            }
        ],
        "gold": [row["label"]],
    }


def sample(path, n, seed):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return random.Random(seed).sample(rows, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jglue", default="data/JGLUE")
    parser.add_argument("--out", default="data")
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    src = os.path.join(args.jglue, "datasets", "jnli-v1.3")
    os.makedirs(args.out, exist_ok=True)
    for split, count, name in (("valid", 1000, "jnli-test"), ("train", 500, "jnli-dev")):
        rows = sample(os.path.join(src, "%s-v1.3.json" % split), count, args.seed)
        path = os.path.join(args.out, name + ".jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(to_item(row, split), ensure_ascii=False) + "\n")
        ids = {json.loads(line)["id"] for line in open(path, encoding="utf-8")}
        assert len(ids) == count, "%s: ID が重複している" % path
        print("wrote", path, count, "items")


if __name__ == "__main__":
    main()
