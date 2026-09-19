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


def _orders_of(records):
    return sorted({r["order"] for rec in records for r in rec["runs"]})


def ask(args, backend):
    out = []
    for item in load_jsonl(args.input):
        qids = [q["id"] for q in item["questions"]]
        assert len(item["gold"]) == len(qids), (
            "%s: gold %d 件に対して questions %d 件。並びが一致していない"
            % (item["id"], len(item["gold"]), len(qids))
        )
        assert len(set(qids)) == len(qids), "%s: 質問 ID が重複している %r" % (item["id"], qids)
        record = {
            "id": item["id"],
            "qids": qids,
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
    """((項目 ID, 質問 ID), 確率, 正解) の列にならす。

    まず順序ごとに繰り返し分の確率を平均し、そのうえで order=None なら順序どうしを平均する
    （順序ごとに繰り返し回数が違っても、順序の重みは等しくなる）。
    1 回目だけを使うと繰り返しのぶれが順序の効果に見えてしまうので、先に平均する。
    その質問で実際に投げていない順序（score・noul・選択肢が足りない場合）は
    平均に入れないので、同じ並びを二重に数えることはない。
    """
    items = []
    for record in records:
        groups = {}
        for r in record["runs"]:
            if order is None or r["order"] == order:
                groups.setdefault(r["order"], []).append(r)
        for qid, gold in zip(record["qids"], record["gold"]):
            per_order = [
                metrics.average_probs([r["probs"][qid] for r in runs if qid in r["probs"]])
                for runs in groups.values()
                if any(qid in r["probs"] for r in runs)
            ]
            if not per_order:
                continue
            avg = metrics.average_probs(per_order)
            items.append(((record["id"], qid), metrics.apply_temperature(avg, temperature), gold))
    return items


def _is_choice(probs):
    """raw には質問の型が入っていないので、ラベルの形から choice かどうかを見る。

    ponytail: score は段階の番号、noul は true/false と決まっているので、それ以外を choice
    とみなす。型が要る指標が増えたら ask の出力に type を書くようにしてこの推定を消す。
    """
    labels = set(probs)
    return labels != {"true", "false"} and not all(k.isdigit() for k in labels)


def _choice_counts(records):
    """choice の質問だけを数える。戻り値は (順序 → 混同行列, ラベル → 位置ごとの回数)。

    混同行列は 正解 → (答え → 件数)。答えなし（同点）は None のまま数える。
    位置は記号 A, B, C… の何番目にその選択肢が置かれていたか。
    同じ (項目, 質問, 順序) は繰り返しの 1 回目だけを数え、分母を質問数にそろえる。
    """
    gold_of = {
        (rec["id"], qid): g for rec in records for qid, g in zip(rec["qids"], rec["gold"])
    }
    confusion, positions, seen = {}, {}, set()
    for rec in records:
        for r in rec["runs"]:
            for qid, probs in r["probs"].items():
                key = (rec["id"], qid, r["order"])
                if key in seen or not _is_choice(probs):
                    continue
                seen.add(key)
                pred = metrics.argmax(probs)
                rows = confusion.setdefault(r["order"], {}).setdefault(gold_of[key[:2]], {})
                rows[pred] = rows.get(pred, 0) + 1
                if pred is not None:
                    at = positions.setdefault(pred, {})
                    at[list(probs).index(pred)] = at.get(list(probs).index(pred), 0) + 1
    return confusion, positions


def _print_choice_counts(records):
    confusion, positions = _choice_counts(records)
    if not confusion:
        return
    labels = sorted({g for rows in confusion.values() for g in rows})
    preds = sorted(
        {p for rows in confusion.values() for row in rows.values() for p in row},
        key=lambda p: (p is None, p),
    )
    width = max(len(s) for s in labels + [p or "答えなし" for p in preds])
    for order in sorted(confusion):
        print("\n混同行列（順序 %d、choice のみ）  正解 \\ 答え" % order)
        print("  %-*s %s" % (width, "", " ".join("%*s" % (width, p or "答えなし") for p in preds)))
        for gold in labels:
            row = confusion[order].get(gold, {})
            print(
                "  %-*s %s  (正解 %d 件)"
                % (
                    width,
                    gold,
                    " ".join("%*d" % (width, row.get(p, 0)) for p in preds),
                    sum(row.values()),
                )
            )

    slots = sorted({i for at in positions.values() for i in at})
    print("\n選択肢 × 記号の位置ごとに選ばれた回数（全順序、choice のみ）")
    print("  %-*s %s" % (width, "", " ".join("%*s" % (width, backends.LETTERS[i]) for i in slots)))
    for label in sorted(positions):
        print(
            "  %-*s %s"
            % (width, label, " ".join("%*d" % (width, positions[label].get(i, 0)) for i in slots))
        )


def _fit(calibrate_records, order, path, label):
    t = metrics.fit_temperature([(p, g) for _, p, g in _flatten(calibrate_records, order=order)])
    print("温度 T = %.3f（%s の「%s」で推定）" % (t, path, label))
    if t < metrics.T_LO * 1.2 or t > metrics.T_HI * 0.99:
        print("  注意: 探索範囲の端に張り付いている。収束していない可能性がある")
    return t


def report(args):
    if args.calibrate and os.path.realpath(args.calibrate) == os.path.realpath(args.input):
        sys.exit("--calibrate には測定用と別のファイルを渡してください（同じデータで温度を当てると過小評価になる）")

    records = load_jsonl(args.input)
    calibrate_records = load_jsonl(args.calibrate) if args.calibrate else []

    # 順序の集合がずれていると、「平均あり」の T が黙って平均なしの値になる
    if args.calibrate and _orders_of(calibrate_records) != _orders_of(records):
        sys.exit(
            "--calibrate と測定用で順序の集合が違います（検証用 %r / 測定用 %r）。"
            "同じ --orders で作り直してください"
            % (_orders_of(calibrate_records), _orders_of(records))
        )

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
            scored = [(max(p.values()), metrics.argmax(p) == g) for _, p, g in items]
            value, table = metrics.ece(scored)
            print("\n== %s / %s ==" % (label, t_label))
            print("正解率 %.4f  ECE %.4f  件数 %d" % (metrics.accuracy(preds, golds), value, len(items)))
            # ECE は「信頼度の値が当たっているか」しか見ない。信頼度の高い答えほど
            # 当たっているか（識別力）は AUROC と上位 / 下位の差で見る
            area = metrics.auroc(scored)
            k, low, high, diff = metrics.confidence_extremes(scored)
            print(
                "  AUROC %s  下位 20%% 正解率 %.3f  上位 20%% 正解率 %.3f  差 %+.3f（各 %d 件）"
                % ("なし（正解か不正解しかない）" if area is None else "%.3f" % area, low, high, diff, k)
            )
            for lo, hi, n, conf, acc in table:
                print("  [%.1f, %.1f) n=%-4d 平均信頼度 %.3f 正解率 %.3f" % (lo, hi, n, conf, acc))

    # 温度は答えを変えないので、混同行列と位置の表は 4 通りで同じ。順序ごとに 1 度だけ出す
    _print_choice_counts(records)

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
    orders = _orders_of(records)
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

    # latency と runs は同じ回で追記されるので並びが対応する。
    # 2 回目以降の繰り返しは同じプロンプトなのでプレフィックスキャッシュに当たって
    # 桁違いに速い。項目ごとに最初の 1 回（＝その項目で初めて投げるプロンプト）だけを使う
    latencies = [
        t
        for rec in records
        for t, r in list(zip(rec["latency"], rec["runs"]))[:1]
        if r["order"] == 0
    ]
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
    a.add_argument("--backend", choices=["typesafe", "fake", "llamacpp"], default="fake")
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
    if args.backend == "llamacpp":
        url = args.url if args.url != backends.TYPESAFE_URL else backends.LLAMACPP_URL
        return ask(args, lambda s, q: backends.ask_llamacpp(s, q, url=url))
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit("TYPESAFE_API_KEY を環境変数に設定してください")
    return ask(
        args,
        lambda s, q: backends.ask_typesafe(s, q, api_key=key, url=args.url, model=args.model),
    )


if __name__ == "__main__":
    main()
