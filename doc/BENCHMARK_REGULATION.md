# Lasada 全体ベンチマーク・レギュレーション仕様書 (BENCHMARK_REGULATION.md)

## 1. 概要
本ドキュメントは、`Lasada` 基盤LLM（1bit BitMoE / Symbol32 アジア優先トークナイザ構成）の理論的知能、高度推論、プログラミング、数学、エージェント能力、および日本語アライメント品質を厳密・客観的に定量評価するための公式ベンチマークレギュレーションを定義する。

---

## 2. 評価対象ベンチマーク指標スイート (8大国際指標 + 日本語総合特化)

| カテゴリ | ベンチマーク名 | 測定対象・目的 | 評価形式・メトリクス |
| :--- | :--- | :--- | :--- |
| **1. 総合教養** | **MMLU / MMLU-Pro** | 57分野（STEM、人文、社会科学、ビジネス等）の多択知識問題 | 4択/10択 Accuracy (%) |
| **2. 高度思考** | **GPQA** | 博士課程レベル（物理・化学・生物）の超高難易問集 | Zero-shot Accuracy (%) |
| **3. 高等数学** | **MATH** | 高校〜数学オリンピックレベルの記述式問題 | Exact Match / Pass@1 (%) |
| **4. 文章題数学** | **GSM8K** | 小〜中学生レベルの多段階文章題（Chain of Thought 評価） | Pass@1 (%) |
| **5. コード生成** | **HumanEval** | Python 関数自動生成・ユニットテスト通過力 | Pass@1 (%) |
| **6. 自律開発** | **SWE-bench / Verified** | OSS リアル GitHub Issue に対する自律的コード修正能力 | Resolve Rate (%) |
| **7. 会話・対話** | **MT-Bench** | 多段階対話・複雑指示における回答品質評価 | GPT-4 / LLM Judge (1-10点) |
| **8. 多模態思考** | **MMMU** | 高度な大学レベルのマルチモーダル（図表・数式・図面理解）問題 | Accuracy (%) |
| **特化 (日本語)** | **JGLUE / Elyza-Tasks-100** | 日本語総合理解・敬語・文化文脈追従性 | Accuracy / Human Judge (%) |

---

## 3. 総合評価スコア (Comprehensive Lasada Score) の算出定式

全 8 大国際指標および日本語特化指標の正規化平均値として **Comprehensive Lasada Score (CLS)** を算出する。

\[
\text{CLS} = \frac{1}{9} \left( S_{\text{MMLU}} + S_{\text{GPQA}} + S_{\text{MATH}} + S_{\text{GSM8K}} + S_{\text{HumanEval}} + S_{\text{SWE-bench}} + S_{\text{MT-Bench}} \times 10 + S_{\text{MMMU}} + S_{\text{Japanese}} \right)
\]

---

## 4. 実行レギュレーション規約

1. **再現性保証**:
   - `lm-evaluation-harness` (`lm_eval`) または専用評価ハーネスを用い、`seed = 1234` に固定して測定する。
2. **ゼロショット / Few-shot 規定**:
   - 原則 `num_fewshot = 0` (Zero-shot) とし、一部標準比較対象において 5-shot を併記する。
3. **量子化非依存**:
   - 生データ F32 / F16 スコアと 1bit BitMoE 三値化スコアを双方物理測定し、精度維持率 ($\Delta \text{Accuracy}$) を評価する。

---

## 5. 関連物理ファイル
- **ベンチマーク仕様書**: `doc/BENCHMARK_REGULATION.md`
- **評価実行スクリプト**: [`scripts/run_hf_eval.py`](file:///home/tasaburoyamada/sandbox/Lasada/scripts/run_hf_eval.py)
- **Lean 4 検証プログラム**: [`Evaluate.lean`](file:///home/tasaburoyamada/sandbox/Lasada/Evaluate.lean)
