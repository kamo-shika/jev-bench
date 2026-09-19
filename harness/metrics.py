"""指標の計算。入力はすべて「ラベル → 確率」の辞書で、バックエンドには依存しない。

本家 Jev もローカルのモデルも、答えはこの形に正規化してから渡す（backends.to_probs）。
生の確率を results/ に残しておけば、順序入れ替えの平均も温度の調整も
モデルを動かし直さずにここで後処理できる。
"""

import math

EPS = 1e-12


def argmax(probs):
    """最も確率の高いラベル。同点のときは答えなし（None）。

    名前順で先のラベルを選ぶと、確率が割れているだけの質問が
    ラベルの付け方次第で当たったり外れたりする。正解率では None を不正解として数え、
    順序変化率では None 同士を「変化なし」として数える。
    """
    if not probs:
        return None
    top = max(probs.values())
    best = [k for k, v in probs.items() if v == top]
    return best[0] if len(best) == 1 else None


def accuracy(preds, golds):
    if not preds:
        return 0.0
    return sum(p == g for p, g in zip(preds, golds)) / len(preds)


def ece(items, bins=10):
    """ECE と、信頼度ごとの正解率の表を返す。

    items は (信頼度, 正解かどうか) の列。信頼度 1.0 は最後のビンに入れる。
    戻り値は (ECE, [(下限, 上限, 件数, 平均信頼度, 正解率), ...])。空のビンは表に出さない。
    """
    buckets = [[] for _ in range(bins)]
    for conf, correct in items:
        buckets[min(int(conf * bins), bins - 1)].append((conf, correct))

    total = len(items)
    value = 0.0
    table = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(ok for _, ok in bucket) / len(bucket)
        value += len(bucket) / total * abs(acc - avg_conf)
        table.append((i / bins, (i + 1) / bins, len(bucket), avg_conf, acc))
    return value, table


def percentiles(values, ps=(50, 95)):
    """最近接順位法。応答時間の p50 / p95 に使う。"""
    ordered = sorted(values)
    return tuple(ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)] for p in ps)


def repeat_spread(runs):
    """同じ入力を N 回投げたときのぶれ。

    runs は同一入力に対する確率の辞書の列。
    戻り値は (答えが変わった割合, 同じラベルの確率の最大差)。
    答えが変わった割合は、最も多かった答え以外の割合。
    """
    answers = [argmax(p) for p in runs]
    majority = max(set(answers), key=answers.count)
    changed = sum(a != majority for a in answers) / len(answers)

    labels = set().union(*(p.keys() for p in runs))
    spread = max(
        (max(p.get(k, 0.0) for p in runs) - min(p.get(k, 0.0) for p in runs))
        for k in labels
    )
    return changed, spread


def order_change_rate(base_answers, permuted_answers):
    """選択肢の順序を入れ替えたときに答えが変わった割合。"""
    return sum(a != b for a, b in zip(base_answers, permuted_answers)) / len(base_answers)


def average_probs(prob_dicts):
    """順序入れ替えの平均。出てこなかったラベルは 0 として平均する。"""
    labels = set().union(*(p.keys() for p in prob_dicts))
    n = len(prob_dicts)
    return {k: sum(p.get(k, 0.0) for p in prob_dicts) / n for k in labels}


def apply_temperature(probs, t):
    """確率を対数に戻して温度で割り、softmax で確率に戻す。"""
    logits = {k: math.log(max(v, EPS)) / t for k, v in probs.items()}
    top = max(logits.values())
    exps = {k: math.exp(v - top) for k, v in logits.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


def nll(items, t):
    """items は (確率の辞書, 正解ラベル) の列。"""
    return -sum(
        math.log(max(apply_temperature(p, t).get(gold, 0.0), EPS)) for p, gold in items
    ) / len(items)


def fit_temperature(items, lo=0.05, hi=10.0, iters=60):
    """検証用データで NLL が最小になる温度を 1 変数で探す。

    ponytail: NLL が温度について単峰であると仮定した三分探索。
    実際に多峰だったら、まず粗い格子で最小の区間を選んでから同じ探索をかける。
    """
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if nll(items, m1) < nll(items, m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2
