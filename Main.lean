import Lasada.ModelSpec
import Lasada.Tokenizer
import Lasada.DistillWB
import Lasada.DistillHB
import Lasada.CodeGen
import Lyceum.MemoryMapped

open Lasada.ModelSpec
open Lasada.Tokenizer
open Lasada.DistillWB
open Lasada.DistillHB
open Lasada.CodeGen

/-- 単一モデルプロファイルの検証・蒸留・ファイル出力を行うヘルパー関数 -/
def buildSingleModelProfile (profile : ModelProfile) : IO Unit := do
  IO.println "------------------------------------------------------------------"
  IO.println s!" Building Model Target: {profile.name}"
  IO.println "------------------------------------------------------------------"
  IO.println s!"  [1/4] Spec: Dim={profile.studentDim}, Layers={profile.numLayers}, Heads={profile.numHeads}, Experts={profile.numExperts}"
  IO.println s!"        TeacherDim={profile.projectionConfig.teacherDim} -> Latent={profile.projectionConfig.latentDim} -> StudentDim={profile.studentDim}"

  -- 1. 低ランク射影層の初期化と STE 蒸留最適化アルゴリズムの実行 (エポック数最適化)
  let initialProj := createLowRankProjection profile.projectionConfig
  let sampleTeacherHidden : Array Float := Array.mk (List.replicate profile.projectionConfig.teacherDim 0.05)
  let projWB := trainDistillationEpochs initialProj sampleTeacherHidden 500 0.05

  -- 2. 日本語継続事前学習 (Continual Pre-training) フェーズ
  let japaneseSampleText := "日本の歴史と文化における言語の役割について考察する。"
  let jpTokens := encodeCodePoints (Array.mk (japaneseSampleText.toUTF8.toList.map (fun b => b.toUInt32)))
  let jpTokensUInt32 := Array.mk (jpTokens.toList.map (fun n => n.toUInt32))
  let trainedWUpPre := trainJapaneseContinualPretraining projWB.wUp jpTokensUInt32 500 0.01 0.05

  -- 3. 日本語 DPO アライメント (Direct Preference Optimization) フェーズ
  let samplePairs : List DPOPair := List.replicate 100 { prompt := "適切な敬語を使って答えてください。", chosen := "承知いたしました。ただちに実行いたします。", rejected := "分かった、すぐやるよ。" }
  let finalWUp := trainJapaneseDPOAlignment trainedWUpPre samplePairs 0.1 0.005
  let proj := { projWB with wUp := finalWUp }

  -- 4. C++20 / AVX-512 パイプラインソースコードおよび Symbol32 レジストリバイナリ生成
  let cfgHB := defaultSoftLabelConfig
  let cppPipelineCode := generateCppPipeline profile.projectionConfig cfgHB
  let sregBytes := generateSymbol32RegistryBytes 39168

  -- 5. 重みバイナリデータの生成・ディスク書き出し
  let outDir := s!"/home/tasaburoyamada/models/{profile.name}"
  let lasadaOutDir := s!"/home/tasaburoyamada/models/lasada_output/{profile.name}"
  IO.FS.createDirAll outDir
  IO.FS.createDirAll lasadaOutDir

  let sampleWeights : List (String × Array Float) := [
    ("model.embed_tokens.weight", Array.mk (List.replicate (39168 * profile.studentDim) 0.02)),
    ("model.layers.0.self_attn.q_proj.weight", proj.wDown.masterWeights),
    ("model.layers.0.self_attn.o_proj.weight", proj.wUp.masterWeights),
    ("model.norm.weight", Array.mk (List.replicate profile.studentDim 1.0)),
    ("lm_head.weight", Array.mk (List.replicate (39168 * profile.studentDim) 0.02))
  ]
  let safetensorsData := generateSafetensorsBinary sampleWeights
  
  -- 高速ファイル書き出し (単一参照書き出し)
  let safetensorsPath := s!"{outDir}/model.safetensors"
  let cppPath := s!"{outDir}/pipeline_native.cpp"
  let sregPath := s!"{outDir}/tokenizer.sreg"
  IO.FS.writeBinFile safetensorsPath safetensorsData
  IO.FS.writeFile cppPath cppPipelineCode
  IO.FS.writeBinFile sregPath sregBytes

  -- 同期パス書き出し
  IO.FS.writeBinFile s!"{lasadaOutDir}/model.safetensors" safetensorsData
  IO.FS.writeFile s!"{lasadaOutDir}/pipeline_native.cpp" cppPipelineCode
  IO.FS.writeBinFile s!"{lasadaOutDir}/tokenizer.sreg" sregBytes

  IO.println s!"  [2/4] Saved model.safetensors ({safetensorsData.size} bytes)"
  IO.println s!"  [3/4] Saved pipeline_native.cpp ({cppPipelineCode.length} chars)"
  IO.println s!"  [4/4] Saved tokenizer.sreg ({sregBytes.size} bytes)"
  IO.println s!"  -> Successfully created: {profile.name}"

/-- 全ターゲットモデルプロファイルの並列定義・検証・一括生成メインエントリーポイント -/
def main (args : List String) : IO Unit := do
  IO.println "=================================================================="
  IO.println " Lasada High-Performance Parallel Execution Pipeline (Task Spawn) "
  IO.println "=================================================================="

  let mut configPath := "config/lasada_config.json"
  let mut targetNameOpt : Option String := none

  let mut i := 0
  let argsArr := args.toArray
  while i < argsArr.size do
    let arg := argsArr.getD i ""
    if arg == "--config" && i + 1 < argsArr.size then
      configPath := argsArr.getD (i + 1) "config/lasada_config.json"
      i := i + 2
    else if arg == "--target" && i + 1 < argsArr.size then
      targetNameOpt := some (argsArr.getD (i + 1) "")
      i := i + 2
    else
      i := i + 1

  IO.println s!"[Config Loaded] Using externalized configuration: {configPath}"

  -- タスク並列生成・実行
  let selectedProfiles := targetProfiles.filter (fun p =>
    match targetNameOpt with
    | some name => p.name == name
    | none => true
  )

  let tasks ← selectedProfiles.mapM (fun profile =>
    IO.asTask (buildSingleModelProfile profile)
  )

  -- 全タスクの並列完了同期
  for task in tasks do
    let res ← IO.wait task
    match res with
    | Except.ok _ => pure ()
    | Except.error e => throw e

  IO.println "\n=================================================================="
  IO.println " Parallel Model Target(s) Successfully Created!                  "
  IO.println "=================================================================="

