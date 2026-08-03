# Lasada 生徒モデル詳細仕様ドキュメント (MODEL_PROFILES.md)

```
=========================================================================
           Lasada 生徒モデル 詳細仕様・ビルドプロファイル一覧            
=========================================================================
生成日時: 2026-08-03 07:25:56 UTC
共通仕様: アジア優先トークナイザ (Symbol32), BitMoE (Dense Router + 1bit Experts), BitNet b1.58 量子化
-------------------------------------------------------------------------

■ モデル名: Lasada-BitMoE-E4B-Base
  - 教師 Gemma モデル : /home/tasaburoyamada/models/gemma-4-E4B
  - 教師 日本語モデル: /home/tasaburoyamada/models/llm-jp-4-8b-instruct
  - 隠れ層次元 (Dim) : 2048
  - レイヤー数 / Head: 24 レイヤー / 16 ヘッド
  - 1bit 量子化 (BitNet): True
  - MoE エキスパート数: 8 Experts (Top-2 Active)
  - 出力ディレクトリ  : /home/tasaburoyamada/models/lasada_output/Lasada-BitMoE-E4B-Base

■ モデル名: Lasada-BitMoE-E4B-40B
  - 教師 Gemma モデル : /home/tasaburoyamada/models/gemma-4-E4B
  - 教師 日本語モデル: /home/tasaburoyamada/models/llm-jp-4-32b-a3b-thinking
  - 隠れ層次元 (Dim) : 4096
  - レイヤー数 / Head: 32 レイヤー / 32 ヘッド
  - 1bit 量子化 (BitNet): True
  - MoE エキスパート数: 8 Experts (Top-2 Active)
  - 出力ディレクトリ  : /home/tasaburoyamada/models/lasada_output/Lasada-BitMoE-E4B-40B

■ モデル名: Lasada-BitMoE-31B-40B
  - 教師 Gemma モデル : /home/tasaburoyamada/models/gemma-4-31B
  - 教師 日本語モデル: /home/tasaburoyamada/models/llm-jp-4-32b-a3b-thinking
  - 隠れ層次元 (Dim) : 4096
  - レイヤー数 / Head: 32 レイヤー / 32 ヘッド
  - 1bit 量子化 (BitNet): True
  - MoE エキスパート数: 8 Experts (Top-2 Active)
  - 出力ディレクトリ  : /home/tasaburoyamada/models/lasada_output/Lasada-BitMoE-31B-40B

■ モデル名: Lasada-BitMoE-31B-70B
  - 教師 Gemma モデル : /home/tasaburoyamada/models/gemma-4-31B
  - 教師 日本語モデル: /home/tasaburoyamada/models/llm-jp-4-32b-a3b-thinking
  - 隠れ層次元 (Dim) : 8192
  - レイヤー数 / Head: 64 レイヤー / 64 ヘッド
  - 1bit 量子化 (BitNet): True
  - MoE エキスパート数: 16 Experts (Top-2 Active)
  - 出力ディレクトリ  : /home/tasaburoyamada/models/lasada_output/Lasada-BitMoE-31B-70B

=========================================================================
```
