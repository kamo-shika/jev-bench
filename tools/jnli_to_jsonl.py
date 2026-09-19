"""JNLI をハーネスの入力 JSONL に変換する。

測定用は valid から 1,000 件、温度の検証用は train から 500 件。別の split なので重複しない。
出力は data/ に置く（JGLUE は CC BY-SA 4.0。継承条件があるので変換物はコミットしない）。

  python tools/jnli_to_jsonl.py [--jglue data/JGLUE] [--out data] [--seed 20260919]
                                [--prompt v1|v2|v3]

文面は下の PROMPTS に 3 通り置いてある。出力のファイル名に文面の名前が入る。
"""

import argparse
import json
import os
import random

# v1: 含意・矛盾・どちらでもないを言葉どおりに説明した文面
INSTRUCTIONS_V1 = "次の前提文と仮説文の関係として最も適切なものを選んでください。"
CRITERIA_V1 = {
    "entailment": "含意。前提文が正しいとき、仮説文も必ず正しい",
    "contradiction": "矛盾。前提文が正しいとき、仮説文は正しくない",
    "neutral": "どちらでもない。前提文からは仮説文が正しいかどうか決まらない",
}

# v2: JNLI の正解を付けた作業者への指示（data/JGLUE/task_guidelines.md の
# 「JNLI: step 1」）に合わせた文面。前提文の光景を思い浮かべ、その同じ光景で
# 仮説文が成立し得るかを選ばせる
INSTRUCTIONS_V2 = (
    "前提文の光景を思い浮かべてください。"
    "思い浮かべたその光景において、仮説文の内容が成立し得るかを選んでください。"
)
CRITERIA_V2 = {
    "entailment": "その光景では、仮説文は確実に正しい",
    "contradiction": "その光景では、仮説文は確実に正しくない",
    "neutral": "その光景で、仮説文は正しいかもしれない",
}

# 文面の名前 → (指示文, 選択肢の説明, 例を付けるか)
PROMPTS = {
    "v1": (INSTRUCTIONS_V1, CRITERIA_V1, False),
    "v2": (INSTRUCTIONS_V2, CRITERIA_V2, False),
    "v3": (INSTRUCTIONS_V2, CRITERIA_V2, True),
}

# 例を出す順。答えの記号が A / B / C の順に並ばないようにする。
# 選択肢の並びは CRITERIA の順なので、入れ替えなしなら答えは B / C / A になる
EXAMPLE_ORDER = ["contradiction", "neutral", "entailment"]


def state_of(row):
    return "前提文: %s\n仮説文: %s" % (row["sentence1"], row["sentence2"])


def to_item(row, split, instructions, criteria, examples):
    # sentence_pair_id は split ごとに振り直されていて train と valid で重複するので、
    # split を前に付けて区別できるようにする
    question = {
        "id": "q1",
        "type": "choice",
        "instructions": instructions,
        "criteria": dict(criteria),
    }
    if examples:
        question["examples"] = examples
    return {
        "id": "jnli-%s-%s" % (split, row["sentence_pair_id"]),
        "state": state_of(row),
        "questions": [question],
        "gold": [row["label"]],
    }


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pick_examples(train_rows, used_ids, seed):
    """train から各クラス 1 件ずつ、検証用に使った行を避けて選ぶ。

    種を固定した並べ替えの先頭から拾うので、同じ種なら毎回同じ 3 件になる。
    """
    pool = [r for r in train_rows if r["sentence_pair_id"] not in used_ids]
    random.Random(seed + 1).shuffle(pool)
    chosen = {}
    for row in pool:
        chosen.setdefault(row["label"], row)
        if len(chosen) == len(EXAMPLE_ORDER):
            break
    assert len(chosen) == len(EXAMPLE_ORDER), "例に使えるクラスが足りない %r" % sorted(chosen)
    return [{"state": state_of(chosen[label]), "gold": label} for label in EXAMPLE_ORDER]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jglue", default="data/JGLUE")
    parser.add_argument("--out", default="data")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="v1")
    args = parser.parse_args()

    instructions, criteria, with_examples = PROMPTS[args.prompt]
    src = os.path.join(args.jglue, "datasets", "jnli-v1.3")
    os.makedirs(args.out, exist_ok=True)

    train_rows = load(os.path.join(src, "train-v1.3.json"))
    valid_rows = load(os.path.join(src, "valid-v1.3.json"))
    dev_rows = random.Random(args.seed).sample(train_rows, 500)
    test_rows = random.Random(args.seed).sample(valid_rows, 1000)
    examples = (
        pick_examples(train_rows, {r["sentence_pair_id"] for r in dev_rows}, args.seed)
        if with_examples
        else None
    )

    for split, rows, name in (
        ("valid", test_rows, "jnli-test"),
        ("train", dev_rows, "jnli-dev"),
    ):
        path = os.path.join(args.out, "%s-%s.jsonl" % (name, args.prompt))
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                item = to_item(row, split, instructions, criteria, examples)
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        ids = {json.loads(line)["id"] for line in open(path, encoding="utf-8")}
        assert len(ids) == len(rows), "%s: ID が重複している" % path
        print("wrote", path, len(rows), "items")


if __name__ == "__main__":
    main()
