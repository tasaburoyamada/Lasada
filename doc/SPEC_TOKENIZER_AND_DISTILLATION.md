---
@doc_governor: SPECIFICATION
@priority: CRITICAL
@override: TRUE
---

# 自作アジア優先トークナイザ & 蒸留パイプライン機能仕様書 (SPEC_TOKENIZER_AND_DISTILLATION.md)

## 1. アジア優先トークナイザ仕様 (`Lasada.Tokenizer`)

### 1.1 初期語彙アロケーション表
`Symbol32` 空間における Unicode ブロック別初期予約範囲：

| 言語区分 (`LangPriority`) | コードポイント範囲 | 予約開始語彙 ID | 予約語彙数 | マージ重み倍率 |
| :--- | :--- | :--- | :--- | :--- |
| **Japanese (Hiragana)** | `0x3040` - `0x309F` | 256 | 8,192 | 10 |
| **Japanese (Katakana)** | `0x30A0` - `0x30FF` | 8,448 | 8,192 | 10 |
| **CJKV (Hanzi/Kanji)** | `0x4E00` - `0x9FFF` | 16,640 | 16,384 | 5 |
| **OtherAsia (Thai等)** | `0x0E00` - `0x0E7F` | 33,024 | 4,096 | 2 |
| **European (ASCII)** | `0x0020` - `0x007F` | 37,120 | 2,048 | 1 |

---

## 2. Gemma 4 ホワイトボックス射影蒸留仕様 (`Lasada.DistillWB`)

### 2.1 テンソルアライメント構造
- **教師側構造**: Gemma 4 E4B (Dim: 3584) または Gemma 4 31B (Dim: 8192)
- **生徒側構造**: Lasada ベースモデル (Dim: 2048)
- **中間潜在空間**: Low-Rank Latent Space (Dim: 256)

### 2.2 整合性検証ルール (`validateAlignment`)
1. `teacherShape.dim == cfg.teacherDim`
2. `studentShape.dim == cfg.studentDim`
3. `teacherShape.batch == studentShape.batch`

---

## 3. 日本語半ブラックボックス蒸留仕様 (`Lasada.DistillHB`)

### 3.1 教師モデル選定 (試案B 採択)
- **日本語教師モデル**: `llm-jp/llm-jp-4-32b-a3b-thinking` (NII主導 12兆トークン事後学習 / Thinking 熟考アーキテクチャ搭載)
- 日本語ネイティブな文脈把握、敬語の距離感、日本の法制度・社会構造への純粋アライメントを最高優先度として全移植。

### 3.2 Soft-Label / Logit 転写パラメーター
- **Top-K**: 20
- **Temperature**: 0.7
- **DPO Beta**: 0.1

### 3.3 DPO 選好データ構造 (`DPOPair`)
- **Prompt**: 入力テキスト
- **Chosen**: 日本語アライメント済みの回答
- **Rejected**: 欧米系ポリコレ/直訳の不自然な回答

---

## 4. BitMoE アーキテクチャ仕様 (`Lasada.ModelSpec` 拡張)

### 4.1 BitMoE 構成原則 (全面統一標準)
- **Dense Gate Router**: 浮動小数点精度 (FP16 / BF16) の決定論的 Top-2 ルーター。
- **1-bit Experts**: すべての生徒モデルで各 MoE レイヤー配下に 8 〜 16 個の 1bit (BitNet b1.58) 専門化 Expert を配置。
- **全ターゲットラインナップ**:
  1. `Lasada-BitMoE-E4B-Base`: 4B クラス 1bit BitMoE (8 Experts / Top-2 Active)
  2. `Lasada-BitMoE-E4B-40B`: 40B クラス 1bit BitMoE (8 Experts / Top-2 Active)
  3. `Lasada-BitMoE-31B-40B`: 40B クラス 1bit BitMoE (8 Experts / Top-2 Active)
  4. `Lasada-BitMoE-31B-70B`: 70B クラス 1bit BitMoE (16 Experts / Top-2 Active)


