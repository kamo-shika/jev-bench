"""バックエンド。向き先を変えたいときは run.py に渡す関数を差し替えるだけ。

関数の形はどれも ask(state, questions) -> {質問 ID: 答え}。
答えは本家 Jev の `/v1/systemone` の応答と同じ形にそろえる。
"""

import json
import math
import urllib.request
import zlib

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
LLAMACPP_URL = "http://localhost:8099"

# 選択肢に振る記号。1 記号 = 1 トークンであることが前提（Qwen3 では A〜Z がそれぞれ 1 トークン）
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 文面を変えると結果が変わるので、定数は 1 か所だけに置く。
# 末尾の THINK は Qwen3 の思考を空にして、次の 1 トークンを必ず答えにするためのもの
PROMPT = "{instructions}\n\n{state}\n\n{options}\n\n記号 1 文字だけで答えてください。"
# 例を添えるときだけの並び。例の答えは記号なので、選択肢の一覧より後ろに置く。
# 答えを当てる本題は最後に置き、例の答えの続きを書かせないようにする
PROMPT_EXAMPLES = (
    "{instructions}\n\n{options}\n\n{examples}\n\n{state}\n\n記号 1 文字だけで答えてください。"
)
THINK = "<think>\n\n</think>\n\n"


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


def option_names(question):
    """選択肢の名前を、記号 A, B, C… を振る並びで返す。

    choice は criteria の並び（run.permute が入れ替えるのはここ）、
    score は段階の番号、noul は「はい / いいえ」の 2 択として true / false。
    """
    if question["type"] == "noul":
        return ["true", "false"]
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return list(question["criteria"])


def option_texts(question):
    """選択肢の説明を option_names と同じ並びで返す。"""
    criteria = question["criteria"]
    if question["type"] == "score":
        return list(criteria)
    return [criteria[name] for name in option_names(question)]


def probs_from_completion(response, names):
    """/completion の応答を「選択肢名 → 確率」に直す。純粋関数。

    上位トークンに候補の記号が 1 つでも無ければ KeyError。確率 0 で埋めると
    「候補が上位 20 に入らないほど自信がない」のか「本当に 0」のか区別できなくなる。
    """
    top = response["completion_probabilities"][0]["top_logprobs"]
    by_token = {t["token"]: t["logprob"] for t in top}
    logprobs = [by_token[LETTERS[i]] for i in range(len(names))]
    # 候補の中だけで softmax（候補外のトークンに逃げた分は捨てる）
    high = max(logprobs)
    exps = [math.exp(lp - high) for lp in logprobs]
    total = sum(exps)
    return {name: e / total for name, e in zip(names, exps)}


def build_prompt(state, question):
    """モデルに投げる本文を組み立てる。純粋関数。

    question に examples（{"state": ..., "gold": 選択肢名} の列）があれば、
    その答えの記号は**いまの選択肢の並び**から引く。run.permute が並びを入れ替えても
    例の記号が選択肢の一覧とずれない。
    """
    names = option_names(question)
    options = "\n".join("%s. %s" % (LETTERS[i], t) for i, t in enumerate(option_texts(question)))
    examples = question.get("examples")
    if not examples:
        return PROMPT.format(instructions=question["instructions"], state=state, options=options)
    shown = "\n\n".join(
        "例\n%s\n答え: %s" % (e["state"], LETTERS[names.index(e["gold"])]) for e in examples
    )
    return PROMPT_EXAMPLES.format(
        instructions=question["instructions"], state=state, options=options, examples=shown
    )


def to_answer(question, probs):
    """「選択肢名 → 確率」を本家の答えの形にする。to_probs が読むキーだけ入れる。"""
    if question["type"] == "noul":
        return {"type": "noul", "noul": probs["true"]}
    return {"type": question["type"], "probabilities": probs}


def ask_llamacpp(state, questions, *, url=LLAMACPP_URL):
    """llama-server に 1 質問 = /completion 1 回で投げ、記号の確率を読む。"""
    answers = {}
    for question in questions:
        names = option_names(question)
        text = build_prompt(state, question)
        prompt = _post(url + "/apply-template", {"messages": [{"role": "user", "content": text}]})
        prompt = prompt["prompt"] + THINK
        for n_probs in (20, 100):
            body = {
                "prompt": prompt,
                "n_predict": 1,
                "n_probs": n_probs,
                "temperature": 0,
                "cache_prompt": True,
            }
            try:
                probs = probs_from_completion(_post(url + "/completion", body), names)
                break
            except KeyError as e:
                missing = e
        else:
            raise RuntimeError(
                "上位 %d トークンに候補の記号 %s が無い（質問 %s）" % (n_probs, missing, question["id"])
            )
        answers[question["id"]] = to_answer(question, probs)
    return answers


def _post(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as res:
        return json.load(res)


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
