"""llama.cpp の応答を確率に直す部分を、サーバーを立てずに確かめる。

pytest でも `python tests/test_backends.py` でも動く。
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import backends  # noqa: E402

CHOICE = {
    "id": "q1",
    "type": "choice",
    "instructions": "選んでください。",
    "criteria": {"あか": "赤い", "あお": "青い", "みどり": "緑の"},
}
SCORE = {
    "id": "q2",
    "type": "score",
    "instructions": "測ってください。",
    "criteria": ["まったく似ていない", "少し似ている", "とても似ている"],
}
NOUL = {
    "id": "q3",
    "type": "noul",
    "instructions": "はいかいいえで答えてください。",
    "criteria": {"true": "はい", "false": "いいえ"},
}


def _response(logprobs):
    """{記号: logprob} から /completion の応答の形を作る。"""
    return {
        "completion_probabilities": [
            {
                "top_logprobs": [
                    {"id": 0, "token": t, "bytes": [], "logprob": v}
                    for t, v in logprobs.items()
                ]
            }
        ]
    }


def test_option_names_and_texts():
    assert backends.option_names(CHOICE) == ["あか", "あお", "みどり"]
    assert backends.option_texts(CHOICE) == ["赤い", "青い", "緑の"]
    assert backends.option_names(SCORE) == ["0", "1", "2"]
    assert backends.option_texts(SCORE) == SCORE["criteria"]
    assert backends.option_names(NOUL) == ["true", "false"]
    assert backends.option_texts(NOUL) == ["はい", "いいえ"]


def test_option_names_follow_permuted_order():
    # run.permute は criteria の並びを回す。記号はその並びに追従しないといけない
    rotated = {**CHOICE, "criteria": {"あお": "青い", "みどり": "緑の", "あか": "赤い"}}
    assert backends.option_names(rotated) == ["あお", "みどり", "あか"]


def test_probs_from_completion_softmax():
    # 候補 3 つの logprob が等しければ 1/3 ずつ。候補外のトークンは無視する
    res = _response({"A": -1.0, "B": -1.0, "C": -1.0, "D": 0.0, "1": 0.0})
    probs = backends.probs_from_completion(res, backends.option_names(CHOICE))
    assert set(probs) == {"あか", "あお", "みどり"}
    for v in probs.values():
        assert abs(v - 1 / 3) < 1e-12


def test_probs_from_completion_values():
    # logprob の差 log(3) → 確率の比 3 : 1 : 1（候補内で正規化して 0.6 / 0.2 / 0.2）
    res = _response({"A": math.log(3), "B": 0.0, "C": 0.0})
    probs = backends.probs_from_completion(res, backends.option_names(CHOICE))
    assert abs(probs["あか"] - 0.6) < 1e-12
    assert abs(probs["あお"] - 0.2) < 1e-12
    assert abs(sum(probs.values()) - 1.0) < 1e-12


def test_probs_from_completion_missing_label_raises():
    # 候補の記号が上位に無ければ確率 0 で埋めずに例外
    res = _response({"A": 0.0, "B": -1.0, "X": -2.0})
    try:
        backends.probs_from_completion(res, backends.option_names(CHOICE))
    except KeyError as e:
        assert "C" in str(e)
    else:
        raise AssertionError("C が無いのに例外が出なかった")


def test_to_answer_round_trips_through_to_probs():
    probs = backends.probs_from_completion(
        _response({"A": math.log(3), "B": 0.0, "C": 0.0}), backends.option_names(CHOICE)
    )
    assert abs(backends.to_probs(backends.to_answer(CHOICE, probs))["あか"] - 0.6) < 1e-12

    score = backends.probs_from_completion(
        _response({"A": 0.0, "B": 0.0, "C": math.log(2)}), backends.option_names(SCORE)
    )
    assert abs(backends.to_probs(backends.to_answer(SCORE, score))["2"] - 0.5) < 1e-12

    noul = backends.probs_from_completion(
        _response({"A": math.log(4), "B": 0.0}), backends.option_names(NOUL)
    )
    got = backends.to_probs(backends.to_answer(NOUL, noul))
    assert abs(got["true"] - 0.8) < 1e-12 and abs(got["false"] - 0.2) < 1e-12


def test_prompt_has_no_stale_placeholder():
    text = backends.PROMPT.format(
        instructions=CHOICE["instructions"],
        state="前提文: あ\n仮説文: い",
        options="\n".join(
            "%s. %s" % (backends.LETTERS[i], t)
            for i, t in enumerate(backends.option_texts(CHOICE))
        ),
    )
    assert "{" not in text and "}" not in text
    assert "A. 赤い" in text and "C. 緑の" in text
    # 思考を空にする印は、チャットテンプレートの末尾に足す前提
    assert backends.THINK == "<think>\n\n</think>\n\n"

def test_build_prompt_例の記号は今の選択肢の並びから引く():
    q = dict(CHOICE, examples=[{"state": "例1", "gold": "みどり"}, {"state": "例2", "gold": "あか"}])
    text = backends.build_prompt("本題", q)
    # 並びは あか / あお / みどり なので A / B / C
    assert "例1\n答え: C" in text
    assert "例2\n答え: A" in text
    # 本題は例より後ろ。例の答えの続きを書かせないため
    assert text.index("本題") > text.index("例2")

    # 選択肢を 1 つ回すと記号も一緒に動く（あお / みどり / あか → A / B / C）
    rotated = dict(q, criteria={k: CHOICE["criteria"][k] for k in ["あお", "みどり", "あか"]})
    rotated_text = backends.build_prompt("本題", rotated)
    assert "例1\n答え: B" in rotated_text
    assert "例2\n答え: C" in rotated_text


def test_build_prompt_例が無ければ従来の並び():
    text = backends.build_prompt("本題", CHOICE)
    assert text.index("本題") < text.index("A. 赤い")
    assert "答え:" not in text


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("すべて通過")
