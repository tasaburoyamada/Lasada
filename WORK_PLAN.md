# Lasada プロジェクト作業計画書 (WORK_PLAN.md)

## 概要
`Lasada` は、試案1（検証性・安全重視 Lean 4 / `lbir` 完結型パイプライン定義 & C++/CUDA コード生成）に基づき、専門化（ドメイン蒸留・LoRA化）を行う前の基盤ローカルLLMを構築するためのシステムです。

以下のフルスタックエコシステムを統合し、形式検証・制御プレーン・中間表現・最適化出力を一貫して実行します。

1. **`Symbol32`**: 固定幅（4bit/32bit）文字・シンボル変換体系の直接連携
2. **`lbir`**: Lean 4 ベースの検証可能中間表現（LBIR）基盤
3. **`nomos`**: 不変条件（Contract / Laws）の決定論的数理検証フレームワーク
4. **`Lyceum`**: 形式検証済み LLM 制御プレーン / 低ビット量子化・推論コンテキスト基盤

---

## パイプライン構成 (3ステップ + 統治)

1. **自作アジア優先トークナイザー (`Lasada/Tokenizer.lean`)**
   - 日本語 > 中・韓・越 > その他アジア > 欧州 の階層化語彙予約・BPE結合ルール定義
   - `Symbol32` 空間との相互変換型規約
   - `Nomos.Contract` による可逆性・パウンダリ不変条件の数理検証
2. **Gemma 4 ホワイトボックス蒸留 (`Lasada/DistillWB.lean`)**
   - 教師モデル (Gemma 4 E4B / 31B) の隠れ状態と生徒モデル中間表現の低ランク射影行列 $\mathbf{W}_{\text{proj}}$ アライメント
   - `Lyceum.Types` / `Lyceum.Core` 量子化コンテキストとのテンソル互換性型検証
3. **日本語半ブラックボックス蒸留 (`Lasada/DistillHB.lean`)**
   - 日本語特化モデルからの Logit 転写および DPO 損失計算グラフ
   - `Nomos.Laws` に基づく損失上界・挙動境界の不変条件チェック
4. **C++/Triton 実行コード生成モジュール (`Lasada/CodeGen.lean`)**
   - `lbir` バイナリ表現および C++20 / Triton GPU ネイティブコード生成

---

## 工程フロー

1. **リポジトリ構造・`lakefile.toml` の依存更新** (`lbir`, `nomos`, `lyceum`)
2. **`Nomos` & `Lyceum` 連携型モジュールの更新**
3. **`lake build` および `tests/TestLasada.lean` の自己自動テストパス確認**
4. **Git コミット**
