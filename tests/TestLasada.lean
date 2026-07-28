import Lasada

open Lasada.Tokenizer
open Lasada.DistillWB
open Lasada.DistillHB
open Lasada.CodeGen
open Lasada.ModelSpec

/-- ブラックボックステスト用ヘルパー：テスト結果アサーション -/
def assertTest (name : String) (cond : Bool) : IO Unit := do
  if cond then
    IO.println s!"[PASS] {name}"
  else
    IO.println s!"[FAIL] {name}"
    throw (IO.userError s!"Blackbox test failed: {name}")

def main : IO Unit := do
  IO.println "=================================================="
  IO.println " Lasada Blackbox & Binary Integration Test Suite"
  IO.println "=================================================="

  -- テスト 1: トークナイザ語彙重複・境界不変条件 (Nomos Contract)
  let ranges := defaultSymbolRanges
  assertTest "Nomos Tokenizer Contract Non-overlapping & Valid Bounds" (checkTokenizerContract ranges)

  -- テスト 2: 言語別マージ重み順序（日本語 > CJKV > アジア > 欧州）
  let wJP := computeMergeWeight LangPriority.Japanese 100
  let wCJK := computeMergeWeight LangPriority.CJKV 100
  let wAsia := computeMergeWeight LangPriority.OtherAsia 100
  let wEU := computeMergeWeight LangPriority.European 100
  assertTest "Language Weight Order (JP > CJKV > Asia > EU)" (wJP > wCJK && wCJK > wAsia && wAsia > wEU)

  -- テスト 3: Gemma 4 ホワイトボックス射影次元アライメント検証
  let cfgWB : ProjectionConfig := { teacherDim := 3584, studentDim := 2048, latentDim := 256 }
  let shapeTeacher : TensorShape := { batch := 2, seqLen := 512, dim := 3584 }
  let shapeStudent : TensorShape := { batch := 2, seqLen := 512, dim := 2048 }
  let shapeMismatch : TensorShape := { batch := 2, seqLen := 512, dim := 1024 }
  assertTest "WB Projection Valid Alignment Pass" (validateAlignment cfgWB shapeTeacher shapeStudent)
  assertTest "WB Projection Mismatch Alignment Fail" (not (validateAlignment cfgWB shapeTeacher shapeMismatch))

  -- テスト 4: FLOPS 計算ロジック非ゼロ検証
  let flops := computeProjectionFlops cfgWB 512
  assertTest "WB Projection FLOPS Non-zero" (flops > 0)

  -- テスト 5: Nomos Laws 蒸留損失下落評価
  let state0 : DistillState := { stepCount := 0, currentLoss := 2.5 }
  assertTest "Nomos Distill Loss Reduction Law" (verifyLossReductionLaw state0 3.0)

  -- テスト 6: LBIR ヘッダーおよび C++ 出力形式テスト (Binary Blackbox Test)
  let lbirHeader := generateLBIRHeader
  assertTest "LBIR Magic Header Verification" (lbirHeader.data == #[0x4C, 0x42, 0x49, 0x52])

  let cfgHB := defaultSoftLabelConfig
  let cppSnippet := generateCppPipeline cfgWB cfgHB
  assertTest "Generated C++ Contains Symbol32 Header" (cppSnippet.contains "Symbol32.h")
  assertTest "Generated C++ Contains Lyceum Context Initializer" (cppSnippet.contains "init_lyceum_mcp_context")

  -- テスト 7: ターゲットモデルプロファイル (E4B, E4B-40B 1bit, 70B) 仕様の型検証
  let profiles := targetProfiles
  assertTest "Target Model Profiles Count == 3" (profiles.length == 3)
  assertTest "Profile 1 is E4B Base" (profileE4B.studentDim == 2048 && not profileE4B.is1bitQuant)
  assertTest "Profile 2 is E4B-40B 1bit Model" (profileE4B_40B_1bit.projectionConfig.teacherDim == 3584 && profileE4B_40B_1bit.is1bitQuant)
  assertTest "Profile 3 is 70B Model" (profile31B_70B.studentDim == 8192 && not profile31B_70B.is1bitQuant)

  IO.println "=================================================="
  IO.println " All 7 Blackbox Test Scenarios Passed Successfully."
  IO.println "=================================================="
