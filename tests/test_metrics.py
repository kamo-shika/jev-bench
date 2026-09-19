"""指標の計算が手計算の値と一致することを確かめる。

pytest でも `python tests/test_metrics.py` でも動く。
"""

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import backends, metrics, run  # noqa: E402


def test_accuracy():
    assert metrics.accuracy(["a", "b", "a", "c"], ["a", "b", "b", "c"]) == 0.75


def test_ece_手計算():
    # 信頼度 0.9(正) 0.6(誤) 0.4(正) 0.2(誤) → ビンは 9, 6, 4, 2 に 1 件ずつ入る。
    # 各ビンの |正解率 - 平均信頼度| は 0.1, 0.6, 0.6, 0.2。
    # ECE = (0.1 + 0.6 + 0.6 + 0.2) / 4 = 0.375
    value, table = metrics.ece([(0.9, True), (0.6, False), (0.4, True), (0.2, False)])
    assert abs(value - 0.375) < 1e-12
    assert len(table) == 4
    assert table[0] == (0.2, 0.3, 1, 0.2, 0.0)  # 下限, 上限, 件数, 平均信頼度, 正解率
    assert table[-1] == (0.9, 1.0, 1, 0.9, 1.0)


def test_ece_信頼度1は最後のビン():
    value, table = metrics.ece([(1.0, True)])
    assert value == 0.0
    assert table == [(0.9, 1.0, 1, 1.0, 1.0)]


def test_wilson_手計算():
    # 0/10: 中心 (0 + 1.9208/20) / (1 + 0.38416) = 0.06938、半幅 1.96*sqrt(0 + 0.0096)/1.38416
    lo, hi = metrics.wilson(0, 10)
    assert lo == 0.0  # 下限は 0 で止める
    assert abs(hi - 0.27753) < 1e-4
    lo, hi = metrics.wilson(5, 10)
    assert abs(lo - 0.23659) < 1e-4
    assert abs(hi - 0.76341) < 1e-4
    assert metrics.wilson(0, 0) == (0.0, 1.0)


def test_brier_手計算():
    # 1 件目: (0.7-1)^2 + 0.2^2 + 0.1^2 = 0.09 + 0.04 + 0.01 = 0.14
    # 2 件目: 0.2^2 + (0.3-1)^2 + 0.5^2 = 0.04 + 0.49 + 0.25 = 0.78
    # 平均 0.46。正解の 1 件だけを見る流儀（1/2 倍）なら 0.23 になる
    items = [
        ({"a": 0.7, "b": 0.2, "c": 0.1}, "a"),
        ({"a": 0.2, "b": 0.3, "c": 0.5}, "b"),
    ]
    assert abs(metrics.brier(items) - 0.46) < 1e-12
    # 正解のラベルが辞書に無ければ、その正解の確率 0 として数える
    assert abs(metrics.brier([({"a": 1.0}, "b")]) - 2.0) < 1e-12


def test_auroc_手計算():
    # 正解のほうが信頼度が高く完全に分かれる → 1.0、逆 → 0.0
    assert metrics.auroc([(0.9, True), (0.8, True), (0.4, False), (0.3, False)]) == 1.0
    assert metrics.auroc([(0.9, False), (0.8, False), (0.4, True), (0.3, True)]) == 0.0
    # 全部同じ信頼度なら見分けられない（同点はすべて 0.5 として数える）
    assert metrics.auroc([(0.7, True), (0.7, True), (0.7, False), (0.7, False)]) == 0.5
    # 混ざった例: 正解 0.9, 0.5 / 不正解 0.7, 0.5。組は (0.9,0.7)=1 (0.9,0.5)=1
    # (0.5,0.7)=0 (0.5,0.5)=0.5 → 2.5 / 4 = 0.625
    assert abs(metrics.auroc([(0.9, True), (0.5, True), (0.7, False), (0.5, False)]) - 0.625) < 1e-12
    # 片方しか無ければ定義できない
    assert metrics.auroc([(0.9, True), (0.5, True)]) is None


def test_confidence_extremes_手計算():
    # 10 件を信頼度の順に並べ、下位 2 件（0.1, 0.2）と上位 2 件（0.9, 1.0）を見る
    items = [(0.1, True), (0.2, False), (0.3, True), (0.4, True), (0.5, False),
             (0.6, True), (0.7, True), (0.8, True), (0.9, False), (1.0, False)]
    k, low, high, diff = metrics.confidence_extremes(items)
    assert k == 2
    assert low == 0.5 and high == 0.0 and diff == -0.5


def test_percentiles_最近接順位():
    # 4 件なので p50 は 2 番目、p95 は 4 番目
    assert metrics.percentiles([0.4, 0.1, 0.3, 0.2]) == (0.2, 0.4)


def test_repeat_spread():
    runs = [
        {"a": 0.70, "b": 0.30},
        {"a": 0.60, "b": 0.40},
        {"a": 0.45, "b": 0.55},
    ]
    changed, spread = metrics.repeat_spread(runs)
    assert abs(changed - 1 / 3) < 1e-12  # 3 回中 1 回だけ b になった
    assert abs(spread - 0.25) < 1e-12  # a も b も 最大 - 最小 = 0.25


def test_order_change_rate():
    assert metrics.order_change_rate(["a", "b", "c", "d"], ["a", "b", "x", "d"]) == 0.25


def test_average_probs_欠けたラベルは0扱い():
    got = metrics.average_probs([{"a": 0.8, "b": 0.2}, {"a": 0.4}])
    assert abs(got["a"] - 0.6) < 1e-12
    assert abs(got["b"] - 0.1) < 1e-12


def test_apply_temperature():
    probs = {"a": 0.8, "b": 0.2}
    same = metrics.apply_temperature(probs, 1.0)
    assert abs(same["a"] - 0.8) < 1e-12
    # 温度を上げるとならされ、下げるととがる
    assert metrics.apply_temperature(probs, 5.0)["a"] < 0.8
    assert metrics.apply_temperature(probs, 0.5)["a"] > 0.8


def test_fit_temperature_自信過剰なら1より大きい温度():
    # 0.99 の確信で半分外している → ならす方向（T > 1）に寄るはず
    items = [({"a": 0.99, "b": 0.01}, "a" if i % 2 == 0 else "b") for i in range(10)]
    assert metrics.fit_temperature(items) > 1.0
    # 逆に全部当たっているならとがらせる方向（T < 1）
    items = [({"a": 0.6, "b": 0.4}, "a") for _ in range(10)]
    assert metrics.fit_temperature(items) < 1.0


def test_to_probs_noulは2値に開く():
    assert metrics.argmax(backends.to_probs({"type": "noul", "noul": 0.92})) == "true"
    assert abs(backends.to_probs({"type": "noul", "noul": 0.92})["false"] - 0.08) < 1e-12


def test_偽バックエンドは決定的で確率の和が1():
    questions = [
        {"id": "q1", "type": "choice", "instructions": "", "criteria": {"a": "", "b": "", "c": ""}},
        {"id": "q2", "type": "score", "instructions": "", "criteria": ["低", "中", "高"]},
        {"id": "q3", "type": "noul", "instructions": "", "criteria": {"true": "", "false": ""}},
    ]
    first = backends.ask_fake("問い合わせ本文", questions)
    assert first == backends.ask_fake("問い合わせ本文", questions)
    for q in questions:
        probs = backends.to_probs(first[q["id"]])
        assert abs(sum(probs.values()) - 1.0) < 1e-12
    assert set(backends.to_probs(first["q2"])) == {"0", "1", "2"}


def test_scoreとnoulは入れ替えない():
    # 段階（低/中/高）を巡回させると順序尺度が壊れるので、常に元の並びだけを投げる
    s = {"id": "s", "type": "score", "instructions": "", "criteria": ["低", "中", "高"]}
    n = {"id": "n", "type": "noul", "instructions": "", "criteria": {"true": "", "false": ""}}
    for order in range(5):
        assert run.shift_for(s, order) == 0
        assert run.shift_for(n, order) == 0
        assert run.permute(s, order) is s
        assert run.permute(n, order) is n


def test_choice_の入れ替えは並びを変えて選択肢集合を保つ():
    q = {"id": "c", "type": "choice", "instructions": "", "criteria": {"a": "1", "b": "2", "c": "3"}}
    for order in (1, 2):
        moved = run.permute(q, order)
        assert list(moved["criteria"]) != list(q["criteria"])  # 必ず並びが変わる
        assert moved["criteria"] == q["criteria"]  # 中身は同じ（dict の比較は並びを見ない）
    assert run.permute(q, 0) is q
    # 選択肢が 3 個なら並びは 3 通りしかない。それ以上の order は元の並びに戻す（＝投げない）
    assert run.shift_for(q, 3) == 0
    assert run.shift_for(q, 4) == 0


def test_flatten_は順序ごとに繰り返しを平均してから順序を平均する():
    record = {
        "id": "x",
        "qids": ["q1"],
        "gold": ["a"],
        "runs": [
            {"order": 0, "probs": {"q1": {"a": 0.8, "b": 0.2}}},
            {"order": 0, "probs": {"q1": {"a": 0.0, "b": 1.0}}},
            {"order": 1, "probs": {"q1": {"a": 0.4, "b": 0.6}}},
            {"order": 1, "probs": {"q1": {"a": 0.0, "b": 1.0}}},
        ],
    }
    (_, なし, _), = run._flatten([record], order=0)
    assert abs(なし["a"] - 0.4) < 1e-12  # order 0 の 0.8 と 0.0 の平均
    (_, あり, _), = run._flatten([record], order=None)
    assert abs(あり["a"] - 0.3) < 1e-12  # 順序ごとの平均 0.4 と 0.2 の平均


def test_choice_counts_は繰り返しを平均してから数え_choice以外を除く():
    records = [
        {
            "id": "x",
            "qids": ["q1"],
            "gold": ["a"],
            "runs": [
                {"order": 0, "probs": {"q1": {"a": 0.9, "b": 0.1}}},
                {"order": 0, "probs": {"q1": {"a": 0.3, "b": 0.7}}},  # 平均すると a 0.6 で a
                {"order": 1, "probs": {"q1": {"b": 0.8, "a": 0.2}}},  # 入れ替えで b が A の位置
            ],
            "latency": [0.1],
        },
        {  # score は段階の番号なので数えない
            "id": "y",
            "qids": ["s1"],
            "gold": ["0"],
            "runs": [{"order": 0, "probs": {"s1": {"0": 0.8, "1": 0.2}}}],
            "latency": [0.1],
        },
    ]
    confusion, positions = run._choice_counts(records)
    assert confusion == {0: {"a": {"a": 1}}, 1: {"a": {"b": 1}}}
    assert positions == {"a": {0: 1}, "b": {0: 1}}  # どちらも記号 A の位置で選ばれた


def _write_raw(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _report_lines(records, calibrate=None):
    """report を走らせて標準出力の行を返す。"""
    with tempfile.TemporaryDirectory() as d:
        args = argparse.Namespace(input=os.path.join(d, "測定.jsonl"), calibrate=None)
        _write_raw(args.input, records)
        if calibrate is not None:
            args.calibrate = os.path.join(d, "検証.jsonl")
            _write_raw(args.calibrate, calibrate)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run.report(args)
        return buf.getvalue().splitlines()


def _two_order_records(n=10, p0=0.97, p1=0.60):
    """order 0 と order 1 で確信の強さが違う raw。10 件中 2 件だけ正解が b。"""
    return [
        {
            "id": "x%d" % i,
            "qids": ["q1"],
            "gold": ["a" if i % 5 else "b"],
            "runs": [
                {"order": 0, "probs": {"q1": {"a": p0, "b": 1 - p0}}},
                {"order": 1, "probs": {"q1": {"a": p1, "b": 1 - p1}}},
            ],
            "latency": [0.1],
        }
        for i in range(n)
    ]


def test_温度は行ごとに当てる():
    records = _two_order_records()
    lines = _report_lines(records, calibrate=records and _two_order_records())
    出た = [float(line.split("=")[1].split("（")[0]) for line in lines if line.startswith("温度 T =")]
    assert len(出た) == 2  # 「平均なし」と「平均あり」で 1 つずつ

    期待 = [
        metrics.fit_temperature([(p, g) for _, p, g in run._flatten(_two_order_records(), order=o)])
        for o in (0, None)
    ]
    assert abs(出た[0] - 期待[0]) < 5e-4  # 平均なしの行は order 0 の確率で当てた T
    assert abs(出た[1] - 期待[1]) < 5e-4  # 平均ありの行は平均した確率で当てた T
    assert abs(出た[0] - 出た[1]) > 0.01  # 確信の強さが違うので同じ T にはならない


def test_選択肢数を超える順序は重複して平均しない():
    # 選択肢 2 個に --orders 3。元 0.9 / 入れ替え 0.3 の 2 通りだけ投げて平均 0.6
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "入力.jsonl")
        with open(src, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": "i1",
                "state": "本文",
                "questions": [{"id": "q1", "type": "choice", "instructions": "",
                               "criteria": {"a": "", "b": ""}}],
                "gold": ["a"],
            }) + "\n")
        out = os.path.join(d, "生.jsonl")

        def backend(state, questions):
            先頭 = list(questions[0]["criteria"])[0]
            probs = {"a": 0.9, "b": 0.1} if 先頭 == "a" else {"a": 0.3, "b": 0.7}
            return {"q1": {"type": "choice", "probabilities": probs}}

        args = argparse.Namespace(input=src, output=out, orders=3, repeats=1)
        with contextlib.redirect_stdout(io.StringIO()):
            run.ask(args, backend)

        record, = run.load_jsonl(out)
        assert sorted({r["order"] for r in record["runs"]}) == [0, 1]  # order 2 は投げない
        (_, 平均, _), = run._flatten([record], order=None)
        assert abs(平均["a"] - 0.6) < 1e-12


def test_順序変化率の分母は入れ替えた質問だけ():
    records = [
        {  # score は order 1 に出てこないので分母に入らない
            "id": "A",
            "qids": ["c1", "s1"],
            "gold": ["a", "0"],
            "runs": [
                {"order": 0, "probs": {"c1": {"a": 0.9, "b": 0.1}, "s1": {"0": 0.8, "1": 0.2}}},
                {"order": 1, "probs": {"c1": {"a": 0.2, "b": 0.8}}},
            ],
            "latency": [0.1],
        },
        {  # 選択肢 1 個なので入れ替えようがなく、order 0 しかない
            "id": "B",
            "qids": ["c2"],
            "gold": ["a"],
            "runs": [{"order": 0, "probs": {"c2": {"a": 1.0}}}],
            "latency": [0.1],
        },
    ]
    行, = [line for line in _report_lines(records) if line.startswith("順序 1")]
    assert "変化率 1.0000" in 行  # 入れ替えた 1 件が変わったので 1/1
    assert "入れ替えた質問 1 件" in 行


def test_順序変化率に繰り返しのぶれが混ざらない():
    # どちらの順序も 3 回の平均は a 0.5 / b 0.5。真の順序効果は 0 なので変化率も 0。
    record = {
        "id": "x",
        "qids": ["q1"],
        "gold": ["a"],
        "runs": [
            {"order": 0, "probs": {"q1": {"a": p, "b": 1 - p}}} for p in (0.55, 0.45, 0.50)
        ] + [
            {"order": 1, "probs": {"q1": {"a": p, "b": 1 - p}}} for p in (0.45, 0.55, 0.50)
        ],
        "latency": [0.1],
    }
    for order in (0, 1):
        (_, 平均, _), = run._flatten([record], order=order)
        assert metrics.argmax(平均) is None  # 同点なので両順序とも答えなし

    行, = [line for line in _report_lines([record]) if line.startswith("順序 1")]
    assert "変化率 0.0000" in 行


def test_同点は不正解として数える():
    # gold は名前順で先の a。同点を a と読めば正解になってしまうが、答えなしなので不正解
    records = [{
        "id": "x",
        "qids": ["q1"],
        "gold": ["a"],
        "runs": [{"order": 0, "probs": {"q1": {"a": 0.5, "b": 0.5}}}],
        "latency": [0.1],
    }]
    行 = [line for line in _report_lines(records) if line.startswith("正解率")]
    assert 行 and all("正解率 0.0000" in line for line in 行)


def test_calibrate_と測定用で順序の集合が違うと止まる():
    測定 = _two_order_records()  # 順序 0 と 1
    検証 = [{**r, "runs": [r["runs"][0]]} for r in _two_order_records()]  # 順序 0 だけ
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _report_lines(測定, calibrate=検証)
    except SystemExit as e:
        assert "順序の集合" in str(e)
        assert "[0]" in str(e) and "[0, 1]" in str(e)  # 両方の順序を出す
    else:
        raise AssertionError("順序の集合が違っても止まらなかった")


def _ask_item(item):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "入力.jsonl")
        with open(src, "w", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        args = argparse.Namespace(input=src, output=os.path.join(d, "生.jsonl"), orders=1, repeats=1)
        with contextlib.redirect_stdout(io.StringIO()):
            run.ask(args, backends.ask_fake)


def test_ask_は壊れた入力で止まる():
    q = {"id": "q1", "type": "choice", "instructions": "", "criteria": {"a": "", "b": ""}}
    _ask_item({"id": "ok", "state": "本文", "questions": [q], "gold": ["a"]})  # これは通る

    for 壊れた in (
        {"id": "gold不足", "state": "本文", "questions": [q, {**q, "id": "q2"}], "gold": ["a"]},
        {"id": "ID重複", "state": "本文", "questions": [q, q], "gold": ["a", "a"]},
    ):
        try:
            _ask_item(壊れた)
        except AssertionError as e:
            assert 壊れた["id"] in str(e)
        else:
            raise AssertionError("%s で止まらなかった" % 壊れた["id"])


def test_calibrate_に測定用と同じパスを渡すと止まる():
    records = _two_order_records()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "同じ.jsonl")
        _write_raw(path, records)
        args = argparse.Namespace(input=path, calibrate=os.path.join(d, ".", "同じ.jsonl"))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                run.report(args)
        except SystemExit as e:
            assert "別のファイル" in str(e)
        else:
            raise AssertionError("同じファイルを渡しても止まらなかった")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("すべて通過")
