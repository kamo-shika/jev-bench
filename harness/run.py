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
    """order 番目の順序での巡回シフト量。0 は元の並びで、入れ替えないことを表す。

    入れ替えるのは choice だけ。score の段階は低→高の順序尺度なので巡回させると
    意味が変わるし、noul は criteria の並びから答えが決まらない。
    選択肢が K 個なら元の並びを含めて K 通りしかないので、order が K 以上なら
    同じ並びを二度投げないよう 0 を返す。
    """
    if order == 0 or question["type"] != "choice":
        return 0
    return order if order < len(question["criteria"]) else 0


def permute(question, order):
    """選択肢の順序を入れ替えた質問を作る。criteria の並び順だけを変える。"""
    shift = shift_for(question, order)
    if shift == 0:
        return question
    keys = list(question["criteria"])
    keys = keys[shift:] + keys[:shift]
    return {**question, "criteria": {k: question["criteria"][k] for k in keys}}


def ask(args, backend):
    out = []
    for item in load_jsonl(args.input):
        record = {
            "id": item["id"],
            "qids": [q["id"] for q in item["questions"]],
            "gold": item["gold"],
            "runs": [],
            "latency": [],
        }
        for order in range(args.orders):
            # この順序で実際に並びが変わる質問だけを記録する。1 件もなければ
            # 元の並びを投げ直すだけなので、その順序ごと飛ばす。
            moved = [q["id"] for q in item["questions"] if order == 0 or shift_for(q, order)]
            if not moved:
                continue
            questions = [permute(q, order) for q in item["questions"]]
            for _ in range(args.repeats):
                started = time.monotonic()
                answers = backend(item["state"], questions)
                record["latency"].append(time.monotonic() - started)
                probs = {qid: backends.to_probs(answers[qid]) for qid in moved}
                record["runs"].append({"order": order, "probs": probs})
        out.append(record)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in out:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("wrote", args.output, len(out), "items")


def _flatten(records, order=None, temperature=1.0):
    """((項目 ID, 質問 ID), 確率, 正解) の列にならす。使うのはどの順序でも 1 回目の実行だけ。

    order を指定するとその順序の 1 回目、order=None は各順序の 1 回目の平均。
    その質問で実際に投げていない順序（score・noul・選択肢が足りない場合）は
    平均に入れないので、同じ並びを二重に数えることはない。
    繰り返しのぶれをここに混ぜないので、「順序入れ替えの平均」の欄には
    順序の効果だけが出る。ぶれは元の順序の N 回から別に出す。
    """
    items = []
    for record in records:
        firsts = {}
        for r in record["runs"]:
            firsts.setdefault(r["order"], r)
        runs = list(firsts.values()) if order is None else [r for o, r in firsts.items() if o == order]
        for qid, gold in zip(record["qids"], record["gold"]):
            probs = [r["probs"][qid] for r in runs if qid in r["probs"]]
            if not probs:
                continue
            avg = metrics.average_probs(probs)
            items.append(((record["id"], qid), metrics.apply_temperature(avg, temperature), gold))
    return items


def _fit(calibrate_records, order, path, label):
    t = metrics.fit_temperature([(p, g) for _, p, g in _flatten(calibrate_records, order=order)])
    print("温度 T = %.3f（%s の「%s」で推定）" % (t, path, label))
    if t < 0.06 or t > 9.9:
        print("  注意: 探索範囲の端に張り付いている。収束していない可能性がある")
    return t


def report(args):
    if args.calibrate and os.path.realpath(args.calibrate) == os.path.realpath(args.input):
        sys.exit("--calibrate には測定用と別のファイルを渡してください（同じデータで温度を当てると過小評価になる）")

    records = load_jsonl(args.input)
    calibrate_records = load_jsonl(args.calibrate) if args.calibrate else []

    for label, order in (("順序入れ替えの平均なし", 0), ("順序入れ替えの平均あり", None)):
        # 温度はその行の確率で当てる。平均なしで当てた T を平均ありの行に使い回さない
        temps = [("温度 1.0", 1.0)]
        if args.calibrate:
            t = _fit(calibrate_records, order, args.calibrate, label)
            temps.append(("温度 %.3f" % t, t))
        for t_label, t in temps:
            items = _flatten(records, order=order, temperature=t)
            preds = [metrics.argmax(p) for _, p, _ in items]
            golds = [g for _, _, g in items]
            value, table = metrics.ece(
                [(max(p.values()), metrics.argmax(p) == g) for _, p, g in items]
            )
            print("\n== %s / %s ==" % (label, t_label))
            print("正解率 %.4f  ECE %.4f  件数 %d" % (metrics.accuracy(preds, golds), value, len(items)))
            for lo, hi, n, conf, acc in table:
                print("  [%.1f, %.1f) n=%-4d 平均信頼度 %.3f 正解率 %.3f" % (lo, hi, n, conf, acc))

    # ぶれと順序による答えの変化、応答時間
    spreads = []
    for rec in records:
        for qid in rec["qids"]:
            runs = [r["probs"][qid] for r in rec["runs"] if r["order"] == 0 and qid in r["probs"]]
            if runs:
                spreads.append(metrics.repeat_spread(runs))
    if spreads:
        print(
            "\nぶれ: 答えが変わった割合 %.4f  確率の最大差 %.4f"
            % (
                sum(c for c, _ in spreads) / len(spreads),
                max(s for _, s in spreads),
            )
        )

    # 分母は実際に入れ替えた質問だけ。(項目 ID, 質問 ID) で引き当て、片方に無い組は数えない
    orders = sorted({r["order"] for rec in records for r in rec["runs"]})
    base = {key: metrics.argmax(p) for key, p, _ in _flatten(records, order=0)}
    for order in orders[1:]:
        pairs = [
            (base[key], metrics.argmax(p))
            for key, p, _ in _flatten(records, order=order)
            if key in base
        ]
        if not pairs:
            continue
        rate = metrics.order_change_rate([a for a, _ in pairs], [b for _, b in pairs])
        print("順序 %d での答えの変化率 %.4f（入れ替えた質問 %d 件）" % (order, rate, len(pairs)))

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
    a.add_argument(
        "--orders",
        type=int,
        default=1,
        help="順序の入れ替えを何通り試すか（1 は元の順序のみ）。choice だけが対象で、"
        "選択肢が K 個の質問は元の並びを含めて K 通りまで",
    )
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
