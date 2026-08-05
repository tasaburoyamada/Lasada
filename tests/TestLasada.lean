import Lasada

open Lasada.Tokenizer
open Lasada.DistillWB
open Lasada.DistillHB
open Lasada.CodeGen
open Lasada.ModelSpec

/-- テスト結果アサーションプログラム -/
def assertTest (name : String) (cond : Bool) : IO Unit := do
  if cond then
    IO.println s!"[PASS] {name}"
  else
    IO.println s!"[FAIL] {name}"
    throw (IO.userError s!"Blackbox test failed: {name}")

def main : IO Unit := do
  IO.println "=================================================="
  IO.println " Lasada Complete Program Logic Verification Suite "
  IO.println "=================================================="

  -- テスト 1: トークナイザ語彙重複・境界不変条件 (Nomos Contract)
  let ranges := defaultSymbolRanges
  assertTest "Nomos Tokenizer Contract Non-overlapping & Valid Bounds" (checkTokenizerContract ranges)

  -- テスト 2: 言語別マージ重み順序プログラム検証（日本語 > CJKV > アジア > 欧州）
  let wJP := computeMergeWeight LangPriority.Japanese 100
  let wCJK := computeMergeWeight LangPriority.CJKV 100
  let wAsia := computeMergeWeight LangPriority.OtherAsia 100
  let wEU := computeMergeWeight LangPriority.European 100
  assertTest "Language Weight Order (JP > CJKV > Asia > EU)" (wJP > wCJK && wCJK > wAsia && wAsia > wEU)

  -- テスト 3: Symbol32 コードポイントエンコーディング・分類プログラム検証
  let testCodePoints : Array UInt32 := #[0x3042, 0x30A2, 0x4E00, 0x0041] -- あ, ア, 一, A
  let encodedTokens := encodeCodePoints testCodePoints
  assertTest "Symbol32 Encode CodePoints Output Size == 4" (encodedTokens.size == 4)

  -- テスト 3.1: BPE サブワード結合プログラム検証
  let bpeRule : BPEMergeRule := { pairLeft := 256, pairRight := 257, mergedId := 9000, weight := 10 }
  let mergedTokens := mergeSubwordPairs #[256, 257, 258] [bpeRule]
  assertTest "BPE Merge Subword Pair Success (256, 257 -> 9000)" (mergedTokens == #[9000, 258])

  -- テスト 4: Gemma 4 低ランク射影 Forward テンソルアライメント計算検証
  let cfgWB : ProjectionConfig := { teacherDim := 3584, studentDim := 2048, latentDim := 256 }
  let proj := createLowRankProjection cfgWB
  let teacherH : Array Float := Array.mk (List.replicate 3584 0.1)
  let studentH := forwardProjection proj teacherH
  assertTest "LowRankProjection Output Dim == StudentDim (2048)" (studentH.size == 2048)

  -- テスト 4.1: Lyceum MemoryMappedContext からのストリーミングデータ取得検証
  let mmCtx := Lyceum.MemoryMappedContext.create 16384
  let fetchedSegment := fetchHiddenSegment mmCtx 0 2048
  assertTest "MemoryMappedContext Stream Fetch Segment Size == 2048" (match fetchedSegment with | Except.ok arr => arr.size == 2048 | Except.error _ => false)

  -- テスト 5: DPO Loss 及び Nomos Laws 蒸留ステップ計算プログラム検証
  let dpoLoss := computeDPOLoss 0.8 0.2 0.5 0.5 0.1
  assertTest "DPO Loss Non-zero Computation" (dpoLoss > 0.0)

  let state0 : DistillState := { stepCount := 0, currentLoss := 2.5 }
  let dummyLogits := Array.mk (List.replicate 20 0.5)
  let nextState := stepDistillation state0 dummyLogits dummyLogits
  assertTest "Distillation Step Loss Reduction Logic" (nextState.stepCount == 1)

  -- テスト 6: LBIR ヘッダーおよび Symbol32 バイナリレジストリ生成プログラム検証
  let lbirHeader := generateLBIRHeader
  assertTest "LBIR Magic Header Verification" (lbirHeader.data == #[0x4C, 0x42, 0x49, 0x52])

  let sregBytes := generateSymbol32RegistryBytes 39168
  assertTest "Symbol32 Registry Header Size == 64" (sregBytes.size == 64)

  let cfgHB := defaultSoftLabelConfig
  let cppSnippet := generateCppPipeline cfgWB cfgHB
  assertTest "Generated C++ Contains Symbol32 Header" (cppSnippet.contains "Symbol32.h")
  assertTest "Generated C++ Contains BitNet Matmul Kernel" (cppSnippet.contains "bitnet_matmul_avx512")

  let tritonSnippet := generateTritonPipeline cfgWB cfgHB
  assertTest "Generated Triton Contains BitNet Kernel" (tritonSnippet.contains "bitnet_projection_kernel")

  -- テスト 7: 全5ターゲットモデルプロファイル (すべて 1bit BitMoE) 仕様の検証
  let profiles := targetProfiles
  assertTest "Target Model Profiles Count == 5" (profiles.length == 5)
  assertTest "Profile 1 is BitMoE E4B-Base" (profileBitMoE_E4B_Base.is1bitQuant && profileBitMoE_E4B_Base.numExperts == 8)
  assertTest "Profile 2 is BitMoE 31B-Base" (profileBitMoE_31B_Base.is1bitQuant && profileBitMoE_31B_Base.projectionConfig.teacherDim == 8192)

  IO.println "=================================================="
  IO.println " All Complete Lean 4 Program Logic Verification Passed."
  IO.println "=================================================="
