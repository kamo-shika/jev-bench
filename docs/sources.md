# 配布元とライセンス

確認日: 2026-09-19。すべて公式ページを直接確認した。記憶では書いていない。
「未検証:」が付いた行は、公式ページで裏が取れなかったもの。

## データセット

| 名前 | 配布元 URL | ライセンス | 業務利用の可否 | ファイル名とおおよそのサイズ | 確認日 |
| --- | --- | --- | --- | --- | --- |
| JGLUE / JNLI | https://github.com/yahoojapan/JGLUE | CC BY-SA 4.0 | 可（継承条件あり。派生物も同じライセンスで公開する必要がある） | `datasets/jnli-v1.1/` の JSON。train 20,073 / dev 2,434 件。未検証: バイト数 | 2026-09-19 |
| JGLUE / JCommonsenseQA | https://github.com/yahoojapan/JGLUE | CC BY-SA 4.0 | 可（継承条件あり） | `datasets/jcommonsenseqa-v1.1/` の JSON。train 8,939 / dev 1,119 件。未検証: バイト数 | 2026-09-19 |
| JGLUE / JSTS | https://github.com/yahoojapan/JGLUE | CC BY-SA 4.0 | 可（継承条件あり） | `datasets/jsts-v1.1/` の JSON。train 12,451 / dev 1,457 件。未検証: バイト数 | 2026-09-19 |
| livedoor ニュースコーパス | https://www.rondhuit.com/download.html | CC BY-ND 2.1 JP（表示 – 改変禁止） | **要注意**。ND なので改変の配布ができない | `ldcc-20140209.tar.gz`。未検証: サイズ（公式ページに記載なし） | 2026-09-19 |
| WRIME | https://github.com/ids-cv/wrime | CC BY-NC-ND 4.0 | **不可**。NC（非営利）なので業務利用にあたる使い方はできない | `wrime-ver1.tsv`（43,200 件）/ `wrime-ver2.tsv`（35,000 件）。未検証: サイズ | 2026-09-19 |

JGLUE のライセンス表記は「This work is licensed under a Creative Commons
Attribution-ShareAlike 4.0 International License」。JNLI / JCommonsenseQA / JSTS
それぞれに別のライセンスは書かれておらず、JGLUE 全体の CC BY-SA 4.0 に含まれる扱い。

livedoor の表記は「各記事ファイルにはクリエイティブ・コモンズライセンス『表示 – 改変禁止』が
適用されます」。クレジット表記の条件は記事のカテゴリごとに違い、配布物の各サブディレクトリの
`LICENSE.txt` に書かれている。

## モデル（GGUF が公式に配布されているもの）

| 名前 | 配布元 URL | ライセンス | 業務利用の可否 | ファイル名とおおよそのサイズ | 確認日 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B（1B 級） | https://huggingface.co/Qwen/Qwen3-1.7B-GGUF | Apache-2.0 | 可 | Q8_0 で 1.83 GB。未検証: Q4_K_M など他の量子化のサイズ | 2026-09-19 |
| LFM2.5-1.2B-Instruct（1B 級） | https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF | LFM Open License v1.0（`lfm1.0`） | **条件付き**。年間売上 1,000 万米ドル以上の法人は別途契約が必要 | `LFM2.5-1.2B-Instruct-Q4_K_M.gguf` 696 MB / `-Q8_0.gguf` 1.25 GB / `-F16.gguf` 2.34 GB | 2026-09-19 |
| Qwen3-4B（4B 級） | https://huggingface.co/Qwen/Qwen3-4B-GGUF | Apache-2.0 | 可 | Q4_K_M 2.5 GB / Q8_0 4.28 GB。未検証: 正確なファイル名 | 2026-09-19 |
| Gemma 3 4B it（QAT, 4B 級） | https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf | Gemma Terms of Use（https://ai.google.dev/gemma/terms） | **条件付き**。商用は可だが、禁止用途ポリシーの順守義務が下流にも及ぶ。ダウンロードに連絡先の提供への同意が要る | Q4_0 の GGUF 1 本、3.16 GB。未検証: 正確なファイル名 | 2026-09-19 |
| Qwen3-8B（8B 級） | https://huggingface.co/Qwen/Qwen3-8B-GGUF | Apache-2.0 | 可 | Q4_K_M 5.03 GB / Q8_0 8.71 GB。未検証: 正確なファイル名 | 2026-09-19 |
| Llama-3-ELYZA-JP-8B（8B 級・日本語強化） | https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B-GGUF | Meta Llama 3 Community License | **条件付き**。月間アクティブユーザー 7 億超の事業者は Meta に別途申請が要る。「Built with Meta Llama 3」の表示義務あり | `Llama-3-ELYZA-JP-8B-q4_k_m.gguf` 4.92 GB | 2026-09-19 |

LFM の条項は「"Threshold" shall mean annual revenue of 10 million United States
dollars ($10,000,000) or more」（第 1 条）および「The rights granted under this
License for Commercial Use are conditioned upon You or Your Legal Entity not
exceeding the Threshold」（第 5 条 a）。

Llama 3 の条項は「If, on the Meta Llama 3 version release date, the monthly active
users of the products or services made available by or for Licensee, or Licensee's
affiliates, is greater than 700 million monthly active users in the preceding
calendar month, you must request a license from Meta」（第 2 条）と
「prominently display "Built with Meta Llama 3"」（第 1 条 b i）。
出典: https://developer.meta.com/ai/llama3/license/（`https://www.llama.com/llama3/license/` から転送される）

## ランタイム

| 名前 | 配布元 URL | ライセンス | 業務利用の可否 | 入手方法 | 確認日 |
| --- | --- | --- | --- | --- | --- |
| llama.cpp | https://github.com/ggml-org/llama.cpp | MIT | 可 | Homebrew の formula `llama.cpp` が homebrew/core にある（stable 0.4.1、まだ未インストール） | 2026-09-19 |

### 確率の読み出し

`llama-server` の `POST /completion` に `n_probs` がある。公式 README の記述は
「If greater than 0, the response also contains the probabilities of top N tokens
for each generated token given the sampling settings」。

応答は `probs` 配列で、各要素が `id` / `logprob` / `token` / `bytes` と、
最大 `n_probs` 個の `top_logprobs`（同じく `id` / `logprob` / `token` / `bytes`）を持つ。
`post_sampling_probs: true` を付けると `logprob` が `prob`（0.0〜1.0）に、
`top_logprobs` が `top_probs` に変わる。`n_predict` とは独立に効く。

OpenAI 互換の `/v1/completions` 側には、トークン確率を返す記述がない。
**このスパイクでは `/completion`（llama.cpp 固有のエンドポイント）を使う。**

出典: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

## 本家 Jev（TypeSafe）の API

出典: https://docs.typesafe.ai/api

`POST https://api.typesafe.ai/v1/systemone`、`Authorization: Bearer <API_KEY>`。
リクエストは `state` / `model` / `questions`。**`questions` は配列ではなく
質問 ID をキーにしたマップ**。質問は `type`（`noul` / `choice` / `score`）と
`instructions` / `criteria` を持つ。`criteria` は型ごとに形が違う。

- `noul`: `criteria` は `{"true": ..., "false": ...}`。答えは `noul`（0〜1 の確率）
- `choice`: `criteria` は選択肢名 → 説明のマップ。答えは `choice` / `probabilities` / `confidence`
- `score`: `criteria` は順序付き配列（2 段階以上）。答えは `score` / `legend` / `probabilities`（キーは段階の番号の文字列）/ `confidence`

応答は `model` / `answers`（質問 ID → 答え）/ `usage`（`input_tokens` / `output_tokens`）。

### deprecated / sunset の記載

上記のどのページにも、deprecated・sunset・配布停止の記載は見つからなかった。
