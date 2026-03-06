# Two-Point Orientation Discrimination App

JVP dome を用いた two-point orientation discrimination の検査者補助アプリです。

## ファイル
- `two_point_orientation_discrimination_streamlit_app.py`
- `requirements.txt`

## 実行方法
```bash
pip install -r requirements.txt
streamlit run two_point_orientation_discrimination_streamlit_app.py
```

## 使う dome サイズ
Blank dome は使いません。

```text
0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0, 12.0 mm
```

## 練習
- 8 mm から開始
- 縦 / 横は 50% ずつのランダム
- 5問連続正答で PASS
- 8 mm で 2問誤答したら 10 mm に上げる
- 10 mm で 2問誤答したら 12 mm に上げる
- 12 mm で 2問誤答したら FAIL

## 本番
- 8 mm から開始
- 系列は `系列1` / `系列2` / `ランダム` から選択
- 2-down 1-up
  - 2連続正答で 1 段階小さく
  - 1回誤答で 1 段階大きく
- step は常に隣接する dome サイズ
- 終了条件
  - 0.35 mm で 4問連続正答 → PASS
  - 12 mm で 2問連続誤答 → FAIL
  - 10 reversals 到達 → 収束完了
  - 100 trial 到達 → 収束不良
- 暫定閾値は最後の 6 reversal の中央値で表示

## 事後
- 練習と同じルール

## 出力
- すべての phase をまとめた CSV ログをダウンロード可能
- 本番で使用した系列もテキストでダウンロード可能

## 画面
- 被検者情報の入力欄は省略
- 上段に phase ごとの要約カード
- その下に trial / 現在 mm / reversals などの集計
- さらにその下に大きく `次に使う dome` と `次の向き` を表示
- `4.5 mm` などの小数表示も折り返さないように調整
- 下段で患者の回答 `縦 / 横` を入力
