import Lasada.DistillWB

namespace Lasada.ModelSpec

open Lasada.DistillWB

/-- ターゲットモデル種別 -/
inductive TargetModelKind where
  | BitMoE_E4B_Base : TargetModelKind -- E4Bベース 4Bクラス BitMoE 1bit
  | BitMoE_31B_Base : TargetModelKind -- 31Bベース 4Bクラス 超小型 BitMoE 1bit
  | BitMoE_E4B_40B  : TargetModelKind -- E4Bベース 40Bクラス BitMoE 1bit
  | BitMoE_31B_40B  : TargetModelKind -- 31Bベース 40Bクラス BitMoE 1bit
  | BitMoE_31B_70B  : TargetModelKind -- 31Bベース 70Bクラス BitMoE 1bit
  deriving Inhabited, BEq, Repr

/-- ターゲットモデル詳細構造体 -/
structure ModelProfile where
  kind : TargetModelKind
  name : String
  studentDim : Nat
  numLayers : Nat
  numHeads : Nat
  is1bitQuant : Bool
  numExperts : Nat            -- MoE Expert 数
  activeExperts : Nat         -- Top-k アクティブ Expert 数
  projectionConfig : ProjectionConfig
  deriving Inhabited, Repr

/-- 1. BitMoE E4B-Base モデルプロファイル (4Bクラス 1bit BitMoE) -/
def profileBitMoE_E4B_Base : ModelProfile := {
  kind := .BitMoE_E4B_Base,
  name := "Lasada-BitMoE-E4B-Base",
  studentDim := 2048,
  numLayers := 24,
  numHeads := 16,
  is1bitQuant := true,
  numExperts := 8,
  activeExperts := 2,
  projectionConfig := { teacherDim := 3584, studentDim := 2048, latentDim := 256 }
}

/-- 2. BitMoE 31B-Base モデルプロファイル (Gemma 4 31B 教師 4Bクラス 超小型 1bit BitMoE) -/
def profileBitMoE_31B_Base : ModelProfile := {
  kind := .BitMoE_31B_Base,
  name := "Lasada-BitMoE-31B-Base",
  studentDim := 2048,
  numLayers := 24,
  numHeads := 16,
  is1bitQuant := true,
  numExperts := 8,
  activeExperts := 2,
  projectionConfig := { teacherDim := 8192, studentDim := 2048, latentDim := 256 }
}

/-- 3. BitMoE E4B-40B モデルプロファイル (40Bクラス 1bit BitMoE) -/
def profileBitMoE_E4B_40B : ModelProfile := {
  kind := .BitMoE_E4B_40B,
  name := "Lasada-BitMoE-E4B-40B",
  studentDim := 4096,
  numLayers := 32,
  numHeads := 32,
  is1bitQuant := true,
  numExperts := 8,
  activeExperts := 2,
  projectionConfig := { teacherDim := 3584, studentDim := 4096, latentDim := 512 }
}

/-- 4. BitMoE 31B-40B モデルプロファイル (40Bクラス 1bit BitMoE) -/
def profileBitMoE_31B_40B : ModelProfile := {
  kind := .BitMoE_31B_40B,
  name := "Lasada-BitMoE-31B-40B",
  studentDim := 4096,
  numLayers := 32,
  numHeads := 32,
  is1bitQuant := true,
  numExperts := 8,
  activeExperts := 2,
  projectionConfig := { teacherDim := 8192, studentDim := 4096, latentDim := 512 }
}

/-- 5. BitMoE 31B-70B モデルプロファイル (70Bクラス 1bit BitMoE) -/
def profileBitMoE_31B_70B : ModelProfile := {
  kind := .BitMoE_31B_70B,
  name := "Lasada-BitMoE-31B-70B",
  studentDim := 8192,
  numLayers := 64,
  numHeads := 64,
  is1bitQuant := true,
  numExperts := 16,
  activeExperts := 2,
  projectionConfig := { teacherDim := 8192, studentDim := 8192, latentDim := 1024 }
}

/-- ターゲットモデルの一覧 (全5モデル) -/
def targetProfiles : List ModelProfile := [
  profileBitMoE_E4B_Base,
  profileBitMoE_31B_Base,
  profileBitMoE_E4B_40B,
  profileBitMoE_31B_40B,
  profileBitMoE_31B_70B
]

end Lasada.ModelSpec
