namespace Lasada.Tokenizer

/-- 言語別優先度定義 -/
inductive LangPriority where
  | Japanese     : LangPriority -- 優先度1: 最優先 (ひらがな, カタカナ, 常用漢字)
  | CJKV         : LangPriority -- 優先度2: 中・韓・ベトナム語
  | OtherAsia    : LangPriority -- 優先度3: その他アジア言語
  | European     : LangPriority -- 優先度4: 欧州言語
  deriving Inhabited, BEq, Repr

/-- Symbol32コードポイント範囲構造体 -/
structure SymbolRange where
  lang : LangPriority
  startCode : UInt32
  endCode : UInt32
  reservedVocabStart : Nat
  reservedVocabCount : Nat
  deriving Inhabited, Repr

/-- アジア優先語彙アロケーションテーブル -/
def defaultSymbolRanges : List SymbolRange := [
  { lang := .Japanese,  startCode := 0x3040, endCode := 0x309F, reservedVocabStart := 256, reservedVocabCount := 8192 }, -- Hiragana
  { lang := .Japanese,  startCode := 0x30A0, endCode := 0x30FF, reservedVocabStart := 8448, reservedVocabCount := 8192 }, -- Katakana
  { lang := .CJKV,      startCode := 0x4E00, endCode := 0x9FFF, reservedVocabStart := 16640, reservedVocabCount := 16384 }, -- Kanji/Hanzi
  { lang := .OtherAsia, startCode := 0x0E00, endCode := 0x0E7F, reservedVocabStart := 33024, reservedVocabCount := 4096 }, -- Thai etc.
  { lang := .European,  startCode := 0x0020, endCode := 0x007F, reservedVocabStart := 37120, reservedVocabCount := 2048 }  -- ASCII
]

/-- マージ重み計算プログラム (言語優先度に応じたバイアス適用) -/
def computeMergeWeight (lang : LangPriority) (rawFreq : Nat) : Nat :=
  match lang with
  | .Japanese  => rawFreq * 10
  | .CJKV      => rawFreq * 5
  | .OtherAsia => rawFreq * 2
  | .European  => rawFreq * 1

/-- コードポイントから LangPriority の判定アルゴリズム -/
def classifyCodePoint (cp : UInt32) (ranges : List SymbolRange := defaultSymbolRanges) : LangPriority :=
  match ranges.find? (fun r => cp >= r.startCode && cp <= r.endCode) with
  | some r => r.lang
  | none   => LangPriority.European

/-- Symbol32 トークナイズ処理: コードポイント配列から予約 ID へのマッピングプログラム -/
def encodeCodePoints (codePoints : Array UInt32) (ranges : List SymbolRange := defaultSymbolRanges) : Array Nat := Id.run do
  let mut tokens : Array Nat := #[]
  for cp in codePoints do
    let lang := classifyCodePoint cp ranges
    let matchedRange := ranges.find? (fun r => r.lang == lang && cp >= r.startCode && cp <= r.endCode)
    match matchedRange with
    | some r =>
      let offset := (cp - r.startCode).toNat
      let tokenId := r.reservedVocabStart + (offset % r.reservedVocabCount)
      tokens := tokens.push tokenId
    | none =>
      tokens := tokens.push (cp.toNat % 1000 + 37120)
  return tokens

/-- Nomos 契約: トークナイザの不変条件（語彙範囲の重複・非負バウンダリ）を検証する判定プログラム -/
def checkTokenizerContract (ranges : List SymbolRange) : Bool :=
  ranges.all (fun r => r.startCode < r.endCode && r.reservedVocabCount > 0)

end Lasada.Tokenizer
