import Lasada.DistillWB

namespace Lasada.ModelSpec

open Lasada.DistillWB

/-- ターゲットモデル種別 -/
inductive TargetModelKind where
  | E4BBase       : TargetModelKind -- E4Bベース軽量モデル
  | E4B_40B_1bit  : TargetModelKind -- E4Bベース 40B 1bit量子化モデル (試案1)
  | B31_70B       : TargetModelKind -- 31Bベース 70B大容量モデル
  deriving Inhabited, BEq, Repr

/-- ターゲットモデル詳細構造体 -/
structure ModelProfile where
  kind : TargetModelKind
  name : String
  studentDim : Nat
  numLayers : Nat
  numHeads : Nat
  is1bitQuant : Bool
  projectionConfig : ProjectionConfig
  deriving Inhabited, Repr

/-- E4B ベースモデルプロファイル -/
def profileE4B : ModelProfile := {
  kind := .E4BBase,
  name := "Lasada-E4B-Base",
  studentDim := 2048,
  numLayers := 24,
  numHeads := 16,
  is1bitQuant := false,
  projectionConfig := { teacherDim := 3584, studentDim := 2048, latentDim := 256 }
}

/-- E4B ベース 40B 1bitモデルプロファイル (試案1) -/
def profileE4B_40B_1bit : ModelProfile := {
  kind := .E4B_40B_1bit,
  name := "Lasada-E4B-40B-1bit",
  studentDim := 7168,
  numLayers := 48,
  numHeads := 56,
  is1bitQuant := true,
  projectionConfig := { teacherDim := 3584, studentDim := 7168, latentDim := 512 }
}

/-- 31B ベース 70Bモデルプロファイル -/
def profile31B_70B : ModelProfile := {
  kind := .B31_70B,
  name := "Lasada-31B-70B",
  studentDim := 8192,
  numLayers := 80,
  numHeads := 64,
  is1bitQuant := false,
  projectionConfig := { teacherDim := 8192, studentDim := 8192, latentDim := 1024 }
}

/-- ターゲットモデルの一覧 -/
def targetProfiles : List ModelProfile := [profileE4B, profileE4B_40B_1bit, profile31B_70B]

end Lasada.ModelSpec
