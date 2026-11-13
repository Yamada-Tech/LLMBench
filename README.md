# LLMBench

OpenAI互換APIで動いているLLMの **精度** と **速度** を実測するベンチマークツールです（GUI + CLI）。
外部の公開APIでも、手元で動いている `http://127.0.0.1:8080/v1` でも、接続先を変えるだけで計測できます
（サーバーの起動・管理は本ツールの範囲外）。

- **GUI** は tkinter のみ（追加インストール不要）
- **CLI** は Python 3.11+ の標準ライブラリのみで動作（pip インストール不要）

## クイックスタート（GUI）

```bat
python -m llmbench gui        :: gui.bat をダブルクリックでも可
```

1. **ベースURL**・**モデル名**・（必要なら）**APIキー** を入力
2. 「接続テスト」で1往復して疎通確認 → 「▶ Run Benchmark」で計測開始
3. 結果タブで確認: **結果**（速度表・精度表）/ **ログ**（進行）/ **レポート (Markdown)**
4. 結果は `results/<label>-<日付>.json` と `.md` に自動保存
5. 「保存済み結果の比較表」で複数の結果JSONを選んでモデル比較表を作成

実行中はプログレスバーで進捗を表示し、「■ 中止」でいつでも中断できます。
設定は `~/.llmbench/gui_settings.json` に保存されます（**APIキーは保存にチェックした時のみ**書き込みます）。

### GUI の入力項目

| 項目 | 説明 |
|---|---|
| ベースURL | 計測対象のAPIアドレス（下記「接続先の指定」参照） |
| モデル名 | そのサーバーが受け付けるモデル名 |
| APIキー | 空なら `Authorization` ヘッダは付与しない |
| タイアウト秒 | リクエスト単位（既定 120） |
| max_tokens / temperature / seed | 生成条件（既定 512 / 0 / 42。再現性のため seed は残す） |
| 速度の反復 / ウォームアップ | 空ならプロファイル既定（short 5回・medium 3回・long 2回） |
| 速度プロファイル | short（約30tok）/ medium（約400tok）/ long（約2000tok） |
| 追加コンテキスト長 | `1024,4096` のように指定すると長文promptの処理速度を追加計測 |
| 精度カテゴリ / 難易度 | 未選択なら全32問を実行 |
| 証明書検証をスキップ | 自己署名証明書のHTTPS向け（`--insecure` 相当） |
| thinking無効化 | Qwen3系などが思考トークンで `max_tokens` を使い切る対策 |

## 接続先の指定

GUI の「ベースURL」／CLI の `--base-url` は、末尾の形が違っても同じ場所へ解決されます。

| 指定 | 実際に叩くURL |
|---|---|
| `http://127.0.0.1:8080` | `http://127.0.0.1:8080/v1/chat/completions` |
| `http://127.0.0.1:8080/v1` | `http://127.0.0.1:8080/v1/chat/completions` |
| `https://host:port/v1` | `https://host:port/v1/chat/completions` |
| `https://host/v1/chat/completions` | そのまま |

## CLI の使い方

```bat
:: 速度 + 精度の通し測定
python -m llmbench run --base-url http://127.0.0.1:8080/v1 --model my-model

:: APIキー付きのリモートサーバー（自己署名HTTPS）
python -m llmbench run --base-url https://example.org:11072/v1 --model my-model ^
    --api-key sk-xxxx --insecure

:: 速度のみ / 長文コンテキストの処理速度
python -m llmbench speed --base-url http://127.0.0.1:8080/v1 --model my-model --contexts 1024,4096

:: 精度のみ / カテゴリ・難易度で絞り込み
python -m llmbench accuracy --base-url http://127.0.0.1:8080/v1 --model my-model ^
    --categories math,logic --difficulties easy,medium

:: 複数ランの比較（glob 指定可）
python -m llmbench compare "results/*.json"

:: 依存チェック（ネットワーク不要・Mockで全機能確認）
python -m llmbench selftest
```

主なオプション: `--api-key` / `--temperature`（既定 0）/ `--seed`（既定 42）/
`--max-tokens`（既定 512）/ `--repeats` / `--warmup` / `--profiles short,medium,long` /
`--contexts` / `--repeat`（精度の1問反復）/ `--tasks my_tasks.json` /
`--out results` / `--label 名` / `--insecure` / `--no-thinking` / `--backend mock`（セルフテスト用）

## 何を測るか

**速度ベンチ**（プロファイル別に反復）
- TTFT（最初のトークンまでの時間）p50/p90
- 生成速度 tok/s（平均・中央値・p90）
- E2E 1往復時間、prompt 処理速度
- サーバーが返すトークン統計（OpenAI互換 `usage`、llama.cpp 系の `timings` 拡張）を優先使用し、
  報告が無い場合のみテキストからの概算にフォールバック

**精度ベンチ**（内蔵32問: math / arithmetic / logic / knowledge / extraction / code / reasoning）
- temperature=0・固定seed で実行し、**決定論的に採点**（LLM judge 不使用・再現性あり）
- code 問題は生成されたPythonを `python -I`（isolated）のサブプロセスで実行して採点。ホスト環境を汚しません

**レポート**: JSON（全生データ＋設定）と Markdown（サマリ表）。`compare` でラン間比較表。
APIキーはレポートに `***` とマスキングして記録されます。

## 出力例

```
| プロファイル | TTFT p50 | tok/s 平均 | tok/s p90 | E2E 平均 |
|---|---:|---:|---:|---:|
| short  | 900 ms | 90.3 | 88.1 | 1.6s |
| medium | 1160 ms | 80.5 | 78.9 | 4.3s |

総合正答率: 70.0% (7/10)
```

## 問題集の拡張

`llmbench/data/tasks.json` に問題を追加、または `--tasks my_tasks.json` で自作問題集を指定
（GUI は同梱の問題集のみ）。scorer タイプ: `numeric`（tolerance可）/ `choice` / `exact` /
`json` / `code`（テスト呼び出しを定義）。

## つまずいたとき

| 症状 | 対処 |
|---|---|
| 404 | ベースURLに `/v1` を付ける（または `/v1/chat/completions` まで直接指定） |
| 401 / 403 | APIキーが必要または不正。GUI の APIキー欄／`--api-key` を確認 |
| `CERTIFICATE_VERIFY_FAILED` | 証明書をOSの信頼ストアに登録、または証明書検証をスキップ |
| 応答が空 / 0 tok/s | thinking が `max_tokens` を使い切っている可能性。thinking無効化か `--max-tokens` 増で解決 |
| `getaddrinfo failed` | DNS/VPN/ファイアウォール。IP直指定（例 `http://192.168.1.189:11072`）で確認 |
| code 問題だけが落ちる | 強い量子化ではインデントが壊れたコードが返ることがあります（モデル起因の正当な失点） |
| 1回目だけ極端に遅い | モデルロード中の可能性。ウォームアップを増やす |

## 開発・テスト

```bat
python -X utf8 -m unittest discover -s tests
:: GUI のルートをそのまま実サーバーへ通す確認（結果表・保存ファイルまで検証）
python -X utf8 _gui_real_check.py http://127.0.0.1:8080/v1 my-model
```

GUI が起動しない場合は `python -m tkinter` で tkinter の存在を確認してください。

## 測定値の注意

- `tok/s` はサーバー報告ベース。異なるサーバー構成（GPU/量子化/バッチ）で値は変動します
- 日本語トークン化はモデル間で差が大きく、異モデル間の速度比較は同じプロファイル表で見てください
- 精度は32問と小さく、カテゴリ別の数%の差は統計的に有意ではありません（`--repeat 3` で分散確認可）
