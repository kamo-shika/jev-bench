"""JSONL を読んでバックエンドに投げ、生の確率を results/ に落とす。

指標は raw から後処理で出す（report サブコマンド）ので、順序入れ替えの平均や
温度の調整を試すたびにモデルを動かし直す必要はない。

入力 JSONL の 1 行:
  {"id": "...", "state": "...",
   "questions": [{"id": "q1", "type": "choice", "instructions": "...",
                  "criteria": {"a": "...", "b": "..."}}],
   "gold": ["a"]}
gold は questions と同じ並びの正解ラベル。score は段階の番号の文字列、noul は "true"/"false"。

使い方:
  python -m harness.run ask  data/xxx.jsonl results/xxx.raw.jsonl [--repeats 5] [--orders 3]
  python -m harness.run report results/xxx.raw.jsonl [--calibrate results/dev.raw.jsonl]
"""

import argparse
import itertools
import json
import os
import sys
import time

from harness import backends, metrics


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def shift_for(question, order):
    """order 番目の順序での巡回シフト量。0 は元の並び。

    選択肢が K 個なら、元の並びを除いて K-1 通り。random.shuffle と違い
    order が 0 でない限り必ず並びが変わるので、順序変化率が低めに出ない。
    noul は criteria の並びから答えが決まらないので入れ替えない。
    """
    if order == 0 or question["type"] == "noul":
        return 0
    k = len(question["criteria"])
    return 0 if k < 2 else (order - 1) % (k - 1) + 1


def permute(question, order):
    """選択肢の順序を入れ替えた質問を作る。criteria の並び順だけを変える。

    score は段階の番号が並びと一緒に動くので、答えの確率は
    unpermute_score で元の番号に戻す。
    """
    shift = shift_for(question, order)
    if shift == 0:
        return question
    keys = list(question["criteria"])
    keys = keys[shift:] + keys[:shift]
    if question["type"] == "score":
        return {**question, "criteria": keys}
    return {**question, "criteria": {k: question["criteria"][k] for k in keys}}


def unpermute_score(probs, n_levels, shift):
    """入れ替えて聞いた score の確率を、元の段階の番号に戻す。

    入れ替えたあとの j 番目は、元の並びでは (j + shift) 番目。
    """
    return {str((int(k) + shift) % n_levels): v for k, v in probs.items()}


def ask(args, backend):
    out = []
    for item in load_jsonl(args.input):
        record = {"id": item["id"], "gold": item["gold"], "runs": [], "latency": []}
        for order in range(args.orders):
            questions = [permute(q, order) for q in item["questions"]]
            for _ in range(args.repeats):
                started = time.monotonic()
                answers = backend(item["state"], questions)
                record["latency"].append(time.monotonic() - started)
                probs = []
                for q in item["questions"]:
                    p = backends.to_probs(answers[q["id"]])
                    if q["type"] == "score":
                        p = unpermute_score(p, len(q["criteria"]), shift_for(q, order))
                    probs.append(p)
                record["runs"].append({"order": order, "probs": probs})
        out.append(record)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in out:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("wrote", args.output, len(out), "items")


def _flatten(records, order=None, temperature=1.0):
    """(確率, 正解) の列にならす。使うのはどの順序でも 1 回目の実行だけ。

    order を指定するとその順序の 1 回目、order=None は各順序の 1 回目の平均。
    繰り返しのぶれをここに混ぜないので、「順序入れ替えの平均」の欄には
    順序の効果だけが出る。ぶれは元の順序の N 回から別に出す。
    """
    items = []
    for record in records:
        firsts = {}
        for r in record["runs"]:
            firsts.setdefault(r["order"], r)
        runs = list(firsts.values()) if order is None else [firsts[o] for o in firsts if o == order]
        if not runs:
            continue
        for i, gold in enumerate(record["gold"]):
            probs = metrics.average_probs([r["probs"][i] for r in runs])
            items.append((metrics.apply_temperature(probs, temperature), gold))
    return items


def report(args):
    records = load_jsonl(args.input)
    temperature = 1.0
    if args.calibrate:
        temperature = metrics.fit_temperature(_flatten(load_jsonl(args.calibrate), order=0))
        print("温度 T = %.3f（%s で推定）" % (temperature, args.calibrate))
        if temperature < 0.06 or temperature > 9.9:
            print("  注意: 探索範囲の端に張り付いている。収束していない可能性がある")

    for label, order in (("順序入れ替えの平均なし", 0), ("順序入れ替えの平均あり", None)):
        for t_label, t in (("温度 1.0", 1.0), ("温度 %.3f" % temperature, temperature)):
            if t == 1.0 and t_label != "温度 1.0":
                continue
            items = _flatten(records, order=order, temperature=t)
            preds = [metrics.argmax(p) for p, _ in items]
            golds = [g for _, g in items]
            value, table = metrics.ece(
                [(max(p.values()), metrics.argmax(p) == g) for p, g in items]
            )
            print("\n== %s / %s ==" % (label, t_label))
            print("正解率 %.4f  ECE %.4f  件数 %d" % (metrics.accuracy(preds, golds), value, len(items)))
            for lo, hi, n, conf, acc in table:
                print("  [%.1f, %.1f) n=%-4d 平均信頼度 %.3f 正解率 %.3f" % (lo, hi, n, conf, acc))

    # ぶれと順序による答えの変化、応答時間
    spreads = [
        metrics.repeat_spread([r["probs"][i] for r in rec["runs"] if r["order"] == 0])
        for rec in records
        for i in range(len(rec["gold"]))
        if any(r["order"] == 0 for r in rec["runs"])
    ]
    if spreads:
        print(
            "\nぶれ: 答えが変わった割合 %.4f  確率の最大差 %.4f"
            % (
                sum(c for c, _ in spreads) / len(spreads),
                max(s for _, s in spreads),
            )
        )

    orders = sorted({r["order"] for rec in records for r in rec["runs"]})
    for order in orders[1:]:
        base = [metrics.argmax(p) for p, _ in _flatten(records, order=0)]
        other = [metrics.argmax(p) for p, _ in _flatten(records, order=order)]
        print("順序 %d での答えの変化率 %.4f" % (order, metrics.order_change_rate(base, other)))

    latencies = list(itertools.chain.from_iterable(r["latency"] for r in records))
    if latencies:
        p50, p95 = metrics.percentiles(latencies)
        print("応答時間 p50 %.3fs  p95 %.3fs" % (p50, p95))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask")
    a.add_argument("input")
    a.add_argument("output")
    a.add_argument("--repeats", type=int, default=5)
    a.add_argument("--orders", type=int, default=1, help="順序の入れ替えを何通り試すか（1 は元の順序のみ）")
    a.add_argument("--backend", choices=["typesafe", "fake"], default="fake")
    a.add_argument("--url", default=backends.TYPESAFE_URL)
    a.add_argument("--model", default="jev-latest")

    r = sub.add_parser("report")
    r.add_argument("input")
    r.add_argument("--calibrate", help="温度の推定に使う検証用データの raw ファイル")

    args = parser.parse_args(argv)
    if args.cmd == "report":
        return report(args)

    if args.backend == "fake":
        return ask(args, backends.ask_fake)
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit("TYPESAFE_API_KEY を環境変数に設定してください")
    return ask(
        args,
        lambda s, q: backends.ask_typesafe(s, q, api_key=key, url=args.url, model=args.model),
    )


if __name__ == "__main__":
    main()
