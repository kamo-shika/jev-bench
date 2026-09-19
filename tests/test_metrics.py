"""指標の計算が手計算の値と一致することを確かめる。

pytest でも `python tests/test_metrics.py` でも動く。
"""

import os
import sys

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


def test_score_の逆順は元の段階の番号に戻る():
    q = {"id": "s", "type": "score", "instructions": "", "criteria": ["低", "中", "高"]}
    assert run.permute(q, "x", 0)["criteria"] == ["低", "中", "高"]
    assert run.permute(q, "x", 1)["criteria"] == ["高", "中", "低"]
    assert run.permute(q, "x", 2)["criteria"] == ["低", "中", "高"]  # 偶数は反転しない
    # 逆順で聞いたときの「0 番目＝高」は、元の並びでは 2 番目
    assert run.unpermute_score({"0": 0.7, "1": 0.2, "2": 0.1}, 3) == {
        "2": 0.7,
        "1": 0.2,
        "0": 0.1,
    }


def test_choice_の入れ替えは同じ選択肢集合を保つ():
    q = {"id": "c", "type": "choice", "instructions": "", "criteria": {"a": "1", "b": "2", "c": "3"}}
    moved = run.permute(q, "x", 1)
    assert moved["criteria"] == {"a": "1", "b": "2", "c": "3"}  # 中身は同じ
    assert run.permute(q, "x", 0) is q


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("すべて通過")
