# 👁️ V1AI (Primary Vision AI)

**A lightweight "early vision AI" that automatically extracts target objects from any image without relying on semantic knowledge.**

V1AI is a lightweight, general-purpose vision system inspired by the early visual cortex (V1) in biological vision.

Without pre-memorizing specific object categories, it uses a unique mechanism to observe and integrate images from multiple views (multi-view), automatically extracting key objects and boundaries from any unseen image.
<img width="4377" height="1675" alt="image" src="https://github.com/user-attachments/assets/fc9a373d-8f90-4d68-901c-5a2435e62722" />

* ✋ **Human-labeled data**: 0 (trained exclusively on programmatically auto-generated data)
* 📦 **Model size**: ~5MB (Total parameters under 1.3M: ColorExtractor ~115K, BoundaryFilter ~941K, ObjectExtractor ~306K)
* ⚡ **Execution Speed**: Processes full-image multi-view pipelines in seconds, running at high speed on edge GPUs (such as RTX 2070)

---

## 💡 Why

Current vision models heavily depend on:

* Large-scale data collection
* Human annotation
* Massive model architectures

In contrast, biological systems perceive "where boundaries are" before understanding "what an object is (semantics)."

V1AI implements this early vision processing as an **independent, general-purpose module** decoupled from specific semantics. It provides a "common eye" that operates at high speed on edge devices and mobile terminals without network connectivity, breaking away from server-centralized large models.

---

## 🎯 What It Does

### 1. Multi-View Attention Without Semantic Knowledge

Observes the same image from multiple viewpoints (Full screen, Top-left, Top-right, Bottom-left, Bottom-right) to form independent background hypotheses. While different targets are emphasized depending on the viewpoint, integrating only the boundaries that appear stably across all views (via logical AND-intersection) allows the system to autonomously extract primary objects without prior knowledge.

### 2. Generic Boundary Recognition

Eliminates the need for fine-tuning to memorize specific object classes (like cats or cars). It constructs a $9\times9$ general-purpose boundary recognition filter that physically and geometrically captures light boundaries (edges) and color contrasts common to any image.

### 3. Autonomous Training

Despite being a supervised learning approach, it requires zero human annotations or manual image collection. Training is completed entirely using programmatically and geometrically auto-generated data, achieving early vision acquisition in a fully autonomous manner.

### 4. Edge Device Execution

Keeps the model size under 5MB (under 1.3 million parameters in total) and executes full-image and multi-view filter applications with exceptional efficiency on hardware like the RTX 2070.

---

## 🧠 Core Concept

Human visual recognition follows a hierarchical structure:

```text
Light/Color ➔ Boundary ➔ Object ➔ Semantics

```

V1AI covers the domain leading up to semantics—specifically **"Light ➔ Boundary ➔ Object Candidate."** It replicates the function of the "eye (early visual cortex)" rather than the brain using a compact, general-purpose deep learning approach.

---

## ⚙️ Architecture

```text
[ Input Image (W x H) ]
        │
        ├─────────────────────────────────────────────┐
        ▼                                             ▼
 [ Multi-View Cropper ]                       [ Patch Extractor (9x9) ]
- Full Screen (Col 1)                              │
- Top-Right (Col 2)                                │
- Top-Left (Col 3)                                 │
- Bottom-Right (Col 4)                             │
- Bottom-Left (Col 5)                              │
        │                                             │
        ▼                                             │
 [ RegionalColorExtractor (32x32) ]                   │
- Resize (32x32)                                   │
- Radius-16 circular logic mask separation         │
- Extract 5 object colors / 5 BG colors + flags    │
        │                                             │
        └──────────────────────┬──────────────────────┘
                               ▼
               [ BoundaryFilter & ObjectExtractor ]
- 9x9 patch input + 10-color query
- Infer center pixel probability from local info
                               │
                               ▼
               [ Common Boundary Intersection ]
- Logical AND-intersection of 5 viewpoint maps
- Identify common stable boundaries and core objects

```

### Key Components

1. **RegionalColorExtractor ($32\times32$)**
* Crops input images across viewpoint modes (Full, Top-Right, Top-Left, Bottom-Right, Bottom-Left) and resizes them to $32\times32$.
* Outputs a total of 40 dimensions (5 object colors × 4, 5 background colors × 4) via a lightweight CNN backbone and fully connected layers (with Sigmoid activation).
* Each color consists of 4 dimensions [R, G, B, Exists Flag], eliminating the Hungarian algorithm through luminance sorting and existence flags.
* Applies a circular logical mask (radius 16) to extract color candidates (inside as object, outside as background).
<img width="1000" alt="regional_extractor_32x32_v2_samples" src="https://github.com/user-attachments/assets/55b5d71e-5190-487f-a6dc-d4a2df6b3e7a" />

2. **BoundaryFilter ($9\times9$)**
* Uses `torch.unfold` to extract $9\times9$ patches (input dimension 243) sliding across the entire image with a stride of 1.
* Combines up to 10 color query vectors (30 dimensions) with patch features and feeds them into an MLP.
* Applies a dynamic quantile threshold (e.g., top 20%, lower bound 0.40) to generate stable edge maps.
<img width="1000" alt="training_samples_visualization" src="https://github.com/user-attachments/assets/906d64ac-69f3-4b32-a22f-3fb96dd0b39b" />

3. **ObjectExtractorFilter ($9\times9$)**
* Combines patch features (243 dimensions) with 30 query dimensions (5 object colors [15 dim] + 5 background colors [15 dim]) for a total of 273 dimensions, processed through fully connected layers.
* Outputs background probabilities and RGB predictions for non-background regions.
<img width="1000" alt="object_extractor_training_samples_visualization" src="https://github.com/user-attachments/assets/0a67244a-52d7-4bce-815b-75e121cd06b4" />

---

## 🚀 Usage

The system can be operated interactively via a menu launched by a command or directly via command-line arguments (CLI).

### 1. Launch Command

```bash
v1ai

```

### 2. Interactive Menu

Upon launching, the following menu appears. Enter the number corresponding to the desired operation:

```text
═════════════════════════════════════════════════════════════════
    🚀 Query-Driven MLP Boundary Filter & Color Extractor System
═════════════════════════════════════════════════════════════════
 [ Operations ]
    [1] Train RegionalColorExtractor (Inside r<=32 vs Outside r>32 Colors)
    [2] Train BoundaryFilter (Edge Confidence Classifier)
    [3] Train ObjectExtractorFilter (Object Extraction & Background Removal)
    [4] Run BoundaryVisualizer (Confidence Map & Segmentation)
-----------------------------------------------------------------
 [ Others ]
    [9] Initialize Workspace (Reset)
    [q] Quit
═════════════════════════════════════════════════════════════════
Enter choice: 

```

### 3. Direct CLI Execution

You can also run tasks directly by passing arguments:

* **[1] Train Color Extractor**: `v1ai train-color`
* **[2] Train Boundary Filter**: `v1ai train-boundary`
* **[3] Train Object Extractor**: `v1ai train-extractor`
* **[4] Run Visualizer Pipeline**: `v1ai visual`
* **[9] Initialize Workspace**: `v1ai init`

<img width="3775" height="2985" alt="pipeline_4row_5col_grid_10" src="https://github.com/user-attachments/assets/4e50d5be-5673-4e28-a00e-72d8eb07a30a" />

---

## 🎯 Use Cases & Expected Effects

* **Advanced Preprocessing Module for CNNs**: As a general-purpose proposal generator for various vision tasks.
* **Fully Automated, High-Quality Training Data Collection Pipeline**: Automatically extracting objects from web images and articles to generate quality training data without human intervention.
* **Real-Time Boundary and Object Extraction on Mobile/Edge Devices**
* **General-Purpose / Unknown Anomaly Detection**

---
---

# 👁️ V1AI (Primary Vision AI)

**いかなる画像に対しても、知識に依存せずに対象物体を自動抽出する軽量「初期視覚AI」**

V1AIは、生物の視覚における初期視覚域（V1）に着想を得た軽量汎用ビジョンシステムです。

特定の物体カテゴリを事前に記憶せず、マルチビュー（複数の視点）から画像を観察・統合する独自の仕組みにより、**あらゆる未知の画像に対しても主要な物体や境界を自律的に抽出します。**
<img width="4377" height="1675" alt="image" src="https://github.com/user-attachments/assets/fc9a373d-8f90-4d68-901c-5a2435e62722" />

* ✋ **Human-labeled data**: 0枚（プログラムによる完全自動生成データのみで学習）
* 📦 **Model size**: 約5MB（各モデルのパラメータ数：ColorExtractor 約115K、BoundaryFilter 約941K、ObjectExtractor 約306K、合計約1.3Mパラメータ）
* ⚡ **Execution Speed**: 複数視点を含む画像全体のパイプライン処理で数秒、エッジGPU（RTX 2070等）上で高速動作を実現

---

## 💡 Why (なぜ作ったのか)

現在のビジョンモデルは、主に以下の要素に依存しています。

* 大規模なデータ収集
* 人手によるアノテーション
* 巨大なモデル構造

一方、生物は「物体が何であるか（意味）」を理解する前に、まず「どこに境界があるか」を知覚しています。

V1AIはこの初期視覚処理を、特定のセマンティクスから切り離された**独立した汎用モジュール**として実装しています。サーバー集権型の大規模モデルに依存せず、ネットワークのないエッジデバイスやモバイル端末上でも高速に動作する「共通の目」を提供します。

---

## 🎯 What It Does (いかなる画像からも物体・境界を抽出する仕組み)

### 1. Multi-View Attention Without Semantic Knowledge (マルチビュー統合による注視物体の発見)

同一画像を、全体・左上・右上・左下・右下の複数視点から観察し、それぞれ独立した背景仮説を形成します。視点が変わることで異なる対象に注目しつつ、全ての視点で安定して現れる境界のみを統合（AND交差）することで、知識を持たずに主要物体を自律抽出します。

### 2. Generic Boundary Recognition (汎用境界認識)

特定の物体クラスを事前に覚えるファインチューニングを不要とし、あらゆる画像に共通する光の境界（エッジ）や色の対比を物理的・幾何学的に捉える $9\times9$ の汎用境界認識フィルタを構築しています。

### 3. Autonomous Training (完全自律型データによる学習)

教師あり学習でありながら、人間によるアノテーションや手動の画像収集を一切行っていません。プログラムによって幾何学的に自動生成されたデータのみを用いて学習を完結させ、完全自律型で初期視覚の獲得に成功しています。

### 4. Edge Device Execution (エッジデバイスでの高速実行)

モデルサイズを約5MB（計130万パラメータ未満）に抑え、RTX 2070等の環境において画像全体やマルチビューのフィルタ適用処理を極めて高い効率で実行します。

---

## 🧠 Core Concept (仕組みの根底)

人間の視覚認識は、以下のような段階構造になっています。

```text
光・色 ➔ 境界 (Boundary) ➔ 物体 (Object) ➔ 意味理解 (Semantics)

```

V1AIはこのうち、知識ベースに至る手前の **「光 ➔ 境界 ➔ 物体候補」** の領域を担当します。脳ではなく「目（初期視覚域）」の機能を小型の汎用ディープラーニングで再現しています。

---

## ⚙️ Architecture (アーキテクチャ)

```text
[ Input Image (W x H) ]
        │
        ├─────────────────────────────────────────────┐
        ▼                                             ▼
 [ Multi-View Cropper ]                       [ Patch Extractor (9x9) ]
- Full Screen (Col 1)                              │
- Top-Right (Col 2)                                │
- Top-Left (Col 3)                                 │
- Bottom-Right (Col 4)                             │
- Bottom-Left (Col 5)                              │
        │                                             │
        ▼                                             │
 [ RegionalColorExtractor (32x32) ]                   │
- リサイズ (32x32)                                 │
- 半径16の円論理マスクによる切り分け               │
- 物体側5色 / 背景側5色 ＋ 存在フラグ抽出          │
        │                                             │
        └──────────────────────┬──────────────────────┘
                               ▼
               [ BoundaryFilter & ObjectExtractor ]
- 9x9パッチ入力 ＋ 10色色クエリ
- 局所情報から中心ピクセルの確率を推論
                               │
                               ▼
               [ Common Boundary Intersection ]
- 5つの視点マップの論理積（AND交差）
- 共通する安定した境界・最重要注視物体の特定

```

### 主要コンポーネント

1. **RegionalColorExtractor ($32\times32$)**
* 入力画像を各視点モード（全体、右上、左上、右下、左下）でクロップし、$32\times32$ にリサイズします。
* 軽量CNNバックボーンと全結合層により、計40次元（物体側5色×4、背景側5色×4）を出力します。
* 各色は [R, G, B, 存在フラグ] の4次元で構成され、輝度ソートによる順序固定と存在フラグの導入によってハンガリアン法を廃止しています。
* 中心半径16の円論理マスクを適用し、円内を物体側、円外を背景側として色候補を抽出します。
<img width="1000" alt="regional_extractor_32x32_v2_samples" src="https://github.com/user-attachments/assets/55b5d71e-5190-487f-a6dc-d4a2df6b3e7a" />

2. **BoundaryFilter ($9\times9$)**
* `torch.unfold` を用いて、パッチサイズ $9\times9$（入力次元 $243$）のパッチを画像全体に1ストライドで展開します。
* 最大10色分（30次元）の色クエリベクトルとパッチ特徴量を結合し、MLPに入力します。
* 動的分位点しきい値（例: 上位20%、下限0.40）を適用して安定したエッジマップを生成します。
<img width="1000" alt="training_samples_visualization" src="https://github.com/user-attachments/assets/906d64ac-69f3-4b32-a22f-3fb96dd0b39b" />

3. **ObjectExtractorFilter ($9\times9$)**
* パッチ特徴量（$243$次元）と物体側5色（15次元）・背景側5色（15次元）の合計30次元のクエリを入力し、全結合層で処理します。
* 背景確率と非背景領域のRGB予測を行います。
<img width="1000" alt="object_extractor_training_samples_visualization" src="https://github.com/user-attachments/assets/0a67244a-52d7-4bce-815b-75e121cd06b4" />

---

## 🚀 使い方 (Usage)

本システムは、インストール後にコマンドを実行するとインタラクティブなメニューが起動し、番号を選択して操作することができます。

### 1. 起動コマンド

```bash
v1ai

```

### 2. インターフェースとメニュー選択

起動すると、以下のメニューが表示されます。実行したい項目の番号を入力してください。

```text
═════════════════════════════════════════════════════════════════
    🚀 Query-Driven MLP Boundary Filter & Color Extractor System
═════════════════════════════════════════════════════════════════
 [ Operations ]
    [1] Train RegionalColorExtractor (Inside r<=32 vs Outside r>32 Colors)
    [2] Train BoundaryFilter (Edge Confidence Classifier)
    [3] Train ObjectExtractorFilter (Object Extraction & Background Removal)
    [4] Run BoundaryVisualizer (Confidence Map & Segmentation)
-----------------------------------------------------------------
 [ Others ]
    [9] Initialize Workspace (Reset)
    [q] Quit
═════════════════════════════════════════════════════════════════
Enter choice: 

```

### 3. 各メニューの機能とCLIからの直接実行

メニューの各項目に対応するコマンドライン引数（CLI）を直接指定して実行することも可能です。

* **[1] 色抽出器の訓練** (`v1ai train-color`)
* **[2] 境界フィルターの訓練** (`v1ai train-boundary`)
* **[3] 物体抽出フィルターの訓練** (`v1ai train-extractor`)
* **[4] ビジュアライザ・推論パイプラインの実行** (`v1ai visual`)
* **[9] ワークスペースの初期化** (`v1ai init`)

<img width="3775" height="2985" alt="pipeline_4row_5col_grid_10" src="https://github.com/user-attachments/assets/4e50d5be-5673-4e28-a00e-72d8eb07a30a" />


---

## 🎯 ユースケースと期待される効果

* **CNNの高度な前処理モジュール**: あらゆるビジョンタスクの汎用プロポーザルとして。
* **完全自動・高品質な訓練データ収集パイプライン**: Web画像と記事から物体を自動抽出し、人手を介さずに良質な訓練データを生成する基盤。
* **モバイル・エッジ端末におけるリアルタイム境界・物体抽出**
* **汎用・未知の異物検知**
