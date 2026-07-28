---
@doc_governor: GOVERNANCE
@priority: CRITICAL
@override: TRUE
---

# Nomos & Lyceum 統治・ガバナンス仕様書 (GOV_NOMOS_CONTRACTS.md)

## 1. Nomos 数理不変条件 (Contracts & Laws)

### 1.1 トークナイザ不変条件契約 (`checkTokenizerContract`)
- すべての SymbolRange において、`startCode < endCode` かつ `reservedVocabCount > 0` を充たさなければならない。
- `Symbol32` 空間における非負バウンダリおよび境界重複不在を事前検証する。

### 1.2 蒸留損失減衰法則 (`verifyLossReductionLaw`)
- 状態遷移 `DistillState`（ステップ数, 現在損失）に対し、ステップ進行にともない損失が設定上限 `maxLoss` 以下へ収束することを静的に定義。

---

## 2. Lyceum 制御プレーン & MCP ガバナンス

### 2.1 制御プレーン統合
- 生成された C++/CUDA ネイティブコードに `Lyceum` の Context オブジェクトを直接アタッチ。
- MCP (Model Context Protocol) JSON-RPC リクエスト経由で安全に推論オプション（`LlmRequestOptions`）を指示。

### 2.2 エラーハンドリング・境界防護
- 異常状態発生時は `Lyceum.AppError` 型に準拠し、標準化された MCP エラーレスポンスコード `32000` としてシリアライズ出力する。
