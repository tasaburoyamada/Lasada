# GEMINI.md - Lasada 統治憲法およびポインタ

## プロジェクト概要
`Lasada` は、アジア優先トークナイザ、Gemma 4 ホワイトボックス射影蒸留、日本語半ブラックボックスアライメントを統合した基盤LLM生成システムである。
`Symbol32`, `lbir`, `nomos`, `Lyceum` をフルスタックで活用する。

---

## 設計書ポインタ (@doc_governor)

以下は、`doc-governor` スキルに基づき管理される「最高位の真実」となる物理設計ドキュメント群である。

- **[全体アーキテクチャ設計書](file:///home/tasaburoyamada/sandbox/Lasada/doc/ARCH_LASADA_SYSTEM.md)**: `doc/ARCH_LASADA_SYSTEM.md`
- **[機能・蒸留仕様書](file:///home/tasaburoyamada/sandbox/Lasada/doc/SPEC_TOKENIZER_AND_DISTILLATION.md)**: `doc/SPEC_TOKENIZER_AND_DISTILLATION.md`
- **[Nomos/Lyceum ガバナンス仕様書](file:///home/tasaburoyamada/sandbox/Lasada/doc/GOV_NOMOS_CONTRACTS.md)**: `doc/GOV_NOMOS_CONTRACTS.md`

---

## 統治原則 (HV-CAD / FDOD)
- 実装コードと設計書に乖離が生じた場合、常に上記設計書の記述を絶対的な正解として優先する。
