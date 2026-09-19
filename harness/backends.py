"""バックエンド。向き先を変えたいときは run.py に渡す関数を差し替えるだけ。

関数の形はどれも ask(state, questions) -> {質問 ID: 答え}。
答えは本家 Jev の `/v1/systemone` の応答と同じ形にそろえる。
"""

import json
import urllib.request
import zlib

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"


def to_probs(answer):
    """本家の答えを「ラベル → 確率」の辞書に正規化する。

    - noul: `noul` は「true である確率」なので 2 値に開く
    - choice: `probabilities` をそのまま使う
    - score: `probabilities` のキーは段階の番号の文字列。番号をラベルとして扱う
      （score も段階のラベルを当てる問題として測る。JSTS のような連続値の正解は、
      データを JSONL にするときに最も近い段階へ丸めておく）
    """
    kind = answer["type"]
    if kind == "noul":
        p = float(answer["noul"])
        return {"true": p, "false": 1.0 - p}
    return {str(k): float(v) for k, v in answer["probabilities"].items()}


def ask_typesafe(state, questions, *, api_key, url=TYPESAFE_URL, model="jev-latest"):
    """本家互換の HTTP に投げる。ローカルに立てた互換サーバーなら url だけ変える。"""
    body = {
        "state": state,
        "model": model,
        # 本家の questions は配列ではなく質問 ID をキーにしたマップ
        "questions": {q["id"]: {k: v for k, v in q.items() if k != "id"} for q in questions},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as res:
        return json.load(res)["answers"]


def ask_fake(state, questions):
    """テスト用の偽バックエンド。state と選択肢名から決まる固定の確率を返す。

    ネットワークを使わないので、指標の計算だけを確かめたいときはこれで足りる。
    """
    answers = {}
    for q in questions:
        if q["type"] == "noul":
            answers[q["id"]] = {"type": "noul", "noul": _weight(state, "true")}
            continue
        labels = (
            [str(i) for i in range(len(q["criteria"]))]
            if q["type"] == "score"
            else list(q["criteria"])
        )
        raw = {k: _weight(state, k) for k in labels}
        z = sum(raw.values())
        probs = {k: v / z for k, v in raw.items()}
        answers[q["id"]] = {
            "type": q["type"],
            "probabilities": probs,
            "confidence": max(probs.values()),
        }
        if q["type"] == "choice":
            answers[q["id"]]["choice"] = max(sorted(probs), key=lambda k: probs[k])
        else:
            answers[q["id"]]["score"] = sum(int(k) * v for k, v in probs.items())
    return answers


def _weight(state, label):
    # 0 にならない範囲でばらつく、決定的な重み。
    # 組み込みの hash() はプロセスごとに変わるので crc32 を使う
    return (zlib.crc32((state + "\0" + label).encode()) % 97 + 1) / 98
