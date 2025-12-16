# Essay Stance Structure Analysis - Complete Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Analysis](#dataset-analysis)
3. [Phase 1: Data Pipeline](#phase-1-data-pipeline)
4. [Phase 5: Essay-Level Analysis](#phase-5-essay-level-analysis)
   - [Feature Engineering](#feature-engineering)
   - [Exploratory Data Analysis](#exploratory-data-analysis)
   - [Clustering Analysis](#clustering-analysis)
   - [Predictive Models](#predictive-models)
5. [How to Run](#how-to-run)
6. [Results Summary](#results-summary)
7. [Key Decisions and Changes](#key-decisions-and-changes)
8. [Limitations](#limitations)

---

## Project Overview

### Goal
Analyze persuasive essays to understand and predict their **stance structure** - how arguments are organized to support or oppose a position.

### What We Built
A complete pipeline that:
1. **Parses** BRAT annotation files (argument components and relations)
2. **Extracts** 36 numerical features from each essay
3. **Clusters** essays into 4 distinct argument patterns
4. **Predicts** essay-level stance characteristics from raw text

### Dataset
**Argument Annotated Essays Corpus (AAEC) v2.0**
- 402 persuasive essays written by students
- Professionally annotated with argument structure

---

## Dataset Analysis

### Key Discovery During Planning

When we first examined the dataset, we discovered something important that changed our approach:

**Original Assumption**: Stance labels (For/Against) are sparse and need to be imputed.

**Reality**: 
- Stance labels exist **ONLY on Claims** (not MajorClaims or Premises)
- **ALL Claims** have stance labels (100% coverage)
- There's no missing data to impute!

This discovery came from reading `annotation.conf`:
```
[attributes]
Stance    Arg:Claim, Value:For|Against
```

### Dataset Statistics

| Metric | Count | Notes |
|--------|-------|-------|
| Total Essays | 402 | |
| Train/Val/Test | 274 / 48 / 80 | Official split preserved |
| **Components** | 6,089 | |
| - MajorClaims | 751 | Main thesis statements |
| - Claims | 1,506 | Supporting/opposing claims |
| - Premises | 3,832 | Evidence for claims |
| **Stance Labels** | 1,506 | Only on Claims |
| - For | 1,228 (81.5%) | Supporting the thesis |
| - Against | 278 (18.5%) | Counter-arguments |
| **Relations** | 3,832 | |
| - Supports | 3,613 (94%) | Premise supports Claim |
| - Attacks | 219 (6%) | Component attacks another |

### Class Imbalance Challenge

The dataset is heavily imbalanced:
- **81.5% of claims are "For"** (supporting the thesis)
- **Only 18.5% are "Against"** (counter-arguments)

This meant our models struggle with predicting "Against" or "balanced" essays.

---

## Phase 1: Data Pipeline

### Why This Approach?

We needed to convert BRAT annotation files into Python objects that are easy to work with.

### File: `src/data/brat_parser.py`

**What it does**: Parses `.ann` files containing argument annotations.

**How it works**:

1. **T-lines** (Components): `T1  Claim 123 456  Competition builds character`
   - Extracts: ID, Type, Start position, End position, Text
   
2. **A-lines** (Stance): `A1  Stance T1 For`
   - Links stance label to the component ID
   
3. **R-lines** (Relations): `R1  supports Arg1:T2 Arg2:T1`
   - Extracts: Source component attacks/supports Target component

**Key Data Structures**:

```python
@dataclass
class Component:
    id: str          # e.g., "T1"
    type: str        # MajorClaim, Claim, or Premise
    start: int       # Character position in essay
    end: int
    text: str        # The actual text
    stance: str      # For, Against, or None

@dataclass
class Relation:
    id: str          # e.g., "R1"
    type: str        # supports or attacks
    source_id: str   # The component making the argument
    target_id: str   # The component being supported/attacked

@dataclass
class Essay:
    essay_id: str
    text: str        # Full essay text
    prompt: str      # Essay topic/question
    split: str       # TRAIN or TEST
    components: Dict[str, Component]
    relations: List[Relation]
```

**Convenient Properties** on Essay class:
- `essay.for_claims` → All claims with stance="For"
- `essay.against_claims` → All claims with stance="Against"
- `essay.support_relations` → All "supports" relations
- `essay.attack_relations` → All "attacks" relations

### File: `src/data/dataset_builder.py`

**What it does**: Combines parsed annotations with prompts and splits.

**Steps**:
1. Load prompts from `prompts.csv` (semicolon-separated)
2. Load official train/test split from `train-test-split.csv`
3. Parse all 402 essays using BratParser
4. Create validation set (15% of training data)
5. Attach prompt and split to each Essay object

**Change Made**: Had to handle `UnicodeDecodeError` in prompts.csv by using `errors='replace'` encoding.

---

## Phase 5: Essay-Level Analysis

### Why Essay-Level Instead of Component-Level?

Original plan was to:
1. Classify component stance (For/Against)
2. Classify component type (MajorClaim/Claim/Premise)
3. Extract relations

But we discovered:
- All Claims already have stance → No prediction needed
- Component type is already annotated → No prediction needed
- Relations are already annotated → No extraction needed

**So we pivoted** to analyzing patterns at the **essay level**:
- What types of argument structures exist?
- Can we predict structure from raw text?

---

### Feature Engineering

**File**: `src/analysis/essay_features.py`

We extract **36 features** from each essay, organized into 5 categories:

#### 1. Stance Features (5 features)
Measure how the essay balances For vs Against claims.

| Feature | Formula | Meaning |
|---------|---------|---------|
| `for_count` | Count of For claims | How many supporting claims |
| `against_count` | Count of Against claims | How many counter-arguments |
| `for_ratio` | for_count / (for + against) | Fraction supporting thesis |
| `against_ratio` | 1 - for_ratio | Fraction opposing thesis |
| `stance_balance` | \|for - against\| / total | 0 = balanced, 1 = one-sided |

**Example**: Essay with 4 For claims and 1 Against claim:
- `for_ratio = 4/5 = 0.80`
- `stance_balance = |4-1|/5 = 0.60`

#### 2. Structural Features (9 features)
Measure the argument structure.

| Feature | Formula | Meaning |
|---------|---------|---------|
| `total_components` | All components | Essay complexity |
| `major_claim_count` | Count of MajorClaims | Number of thesis statements |
| `claim_count` | Count of Claims | Number of supporting/opposing points |
| `premise_count` | Count of Premises | Amount of evidence |
| `evidence_density` | premises / claims | Evidence per claim |
| `claim_density` | claims / total | Claim proportion |
| `major_claim_ratio` | major_claims / total | Thesis proportion |

**Example**: Essay with 2 MajorClaims, 4 Claims, 12 Premises:
- `evidence_density = 12/4 = 3.0` (3 pieces of evidence per claim)
- `total_components = 18`

#### 3. Relation Features (6 features)
Measure how components connect.

| Feature | Formula | Meaning |
|---------|---------|---------|
| `total_relations` | All relations | Connectivity |
| `support_count` | Count of "supports" | How much evidence links |
| `attack_count` | Count of "attacks" | How much disagreement |
| `support_ratio` | supports / total | Fraction supporting |
| `attack_ratio` | attacks / total | Fraction attacking |
| `relations_per_component` | relations / components | Average connections |

**Example**: Essay with 10 supports and 2 attacks:
- `attack_ratio = 2/12 = 0.167` (17% of relations are attacks)

#### 4. Graph Features (6 features)
Treat the essay as a graph where components are nodes and relations are edges.

| Feature | Meaning |
|---------|---------|
| `tree_depth` | Longest path from any component to MajorClaim |
| `tree_width` | Max components at any depth level |
| `avg_branching_factor` | Average number of children per node |
| `avg_path_to_major_claim` | Average distance from components to thesis |
| `num_root_components` | Components with no incoming relations |
| `num_leaf_components` | Components with no outgoing relations |

**Why These Matter**: A deep, narrow argument tree suggests linear reasoning. A wide, shallow tree suggests multiple parallel arguments.

**Change Made**: Originally `tree_depth` became `inf` for essays without paths to MajorClaim. Fixed to return 0 instead.

#### 5. Positional Features (8 features)
Where do For/Against claims appear in the essay?

| Feature | Meaning |
|---------|---------|
| `for_in_first_third` | For claims in introduction |
| `for_in_middle_third` | For claims in body |
| `for_in_last_third` | For claims in conclusion |
| `against_in_first_third` | Against claims early |
| `against_in_middle_third` | Against claims in body |
| `against_in_last_third` | Against claims late |
| `first_against_position` | Normalized position (0-1) of first counter-argument |
| `last_against_position` | Normalized position of last counter-argument |

**Why These Matter**: Good essays often introduce counter-arguments in the middle, then refute them.

#### 6. Derived Category
Based on `for_ratio`, we classify essays:

```python
if for_ratio >= 0.8:
    stance_category = "mostly_for"  # One-sided supporting
elif for_ratio <= 0.2 and against_count > 0:
    stance_category = "mostly_against"  # One-sided opposing
else:
    stance_category = "balanced"  # Mix of both
```

**Distribution**: 70% mostly_for, 28% balanced, 2% mostly_against

---

### Exploratory Data Analysis

**File**: `src/analysis/eda.py`

Generated visualizations to understand the data:

1. **`stance_distribution.png`**: Histogram of for_ratio across essays
2. **`component_structure.png`**: Distribution of component counts
3. **`positional_analysis.png`**: Where stances appear (by thirds)
4. **`correlation_heatmap.png`**: Feature correlations
5. **`stance_by_structure.png`**: How structure relates to stance

**Key Findings**:
- Most essays are one-sided (for_ratio > 0.8)
- Evidence density ranges from 1-6 premises per claim
- Counter-arguments (Against) typically appear in middle third
- Strong correlation between premise_count and evidence_density

---

### Clustering Analysis

**File**: `src/analysis/clustering.py`

**Goal**: Find natural groupings of essays by argument structure.

#### Approach

1. **Feature Selection**: Used 10 key features:
   - for_ratio, against_ratio, stance_balance
   - evidence_density, claim_count, premise_count
   - attack_ratio, support_ratio
   - tree_depth, avg_path_to_major_claim

2. **Standardization**: Scaled all features to mean=0, std=1

3. **Optimal K Selection**:
   - Tested K from 2 to 10
   - Used Elbow Method (look for bend in inertia curve)
   - Used Silhouette Score (measure cluster separation)
   - **Chose K=4** based on both metrics

4. **Algorithms**:
   - **K-Means**: Main clustering
   - **Hierarchical**: For dendrogram visualization

5. **Visualization**:
   - **PCA**: Reduce to 2D, color by cluster
   - **t-SNE**: Non-linear projection for better separation

#### Results: 4 Distinct Essay Patterns

| Cluster | Count | For Ratio | Evidence Density | Attack Ratio | Interpretation |
|---------|-------|-----------|------------------|--------------|----------------|
| **one-sided_for** | 158 (39%) | 0.970 | 2.18 | 0.007 | Strongly supports thesis, minimal counter-args |
| **one-sided_for_evidence_rich** | 104 (26%) | 0.871 | 3.89 | 0.054 | Supports thesis with lots of evidence |
| **dialectical** | 86 (21%) | 0.550 | 2.56 | 0.014 | Balanced, considers both sides |
| **balanced_attack_heavy** | 54 (13%) | 0.661 | 2.26 | 0.276 | Uses many attack relations |

---

### Predictive Models

**Goal**: Can we predict essay structure from **raw text alone** (without annotations)?

#### Why This Matters
If we can predict structure from text, we could:
- Automatically assess essay quality
- Provide feedback to student writers
- Analyze large corpora without manual annotation

#### Traditional ML Baseline

**File**: `src/models/essay_predictor_baseline.py`

**Approach**:

1. **Feature Extraction**: TF-IDF on essay text
   - Unigrams, bigrams, trigrams
   - Max 3,000 features
   - Sublinear TF scaling (dampens frequent words)
   - Stop words removed

2. **Models Tried**:
   - Ridge Regression (for for_ratio)
   - Random Forest Regressor
   - **SVR (Support Vector Regression)** ← Best for regression
   - Logistic Regression (for stance_category)
   - Random Forest Classifier
   - **SVC (Support Vector Classifier)** ← Best for classification

3. **Hyperparameters**:
   - SVR: kernel='rbf', C=1.0, epsilon=0.1
   - SVC: kernel='rbf', C=1.0, class_weight='balanced'

**Why SVM?**: Works well with high-dimensional sparse data (TF-IDF produces ~3000 features for ~300 essays).

**Why class_weight='balanced'?**: Dataset is 70% "mostly_for". Without balancing, model would predict everything as "mostly_for".

#### Transformer Model

**File**: `src/models/essay_predictor_transformer.py`

**Approach**:

1. **Model**: DistilBERT (smaller, faster than BERT)
   - Pretrained on large English corpus
   - Fine-tuned on our data

2. **Input Format**: `[Prompt] [SEP] [Essay Text]`
   - Max 512 tokens (essays truncated)

3. **Architecture**:
   - DistilBERT encoder → [CLS] token embedding
   - Regression head: Linear → ReLU → Dropout → Linear → scalar
   - Classification head: Linear → 3-class softmax

4. **Training**:
   - Learning rate: 2e-5
   - Batch size: 8
   - Epochs: 5
   - AdamW optimizer with linear warmup

**Why DistilBERT over RoBERTa?**:
- 40% smaller, 60% faster
- Similar performance on most tasks
- Our dataset is small (274 training examples)

**Challenge**: Essays average ~280 words but DistilBERT only handles 512 tokens. We truncate, which loses some information.

#### Model Results

**For Ratio Prediction (Regression)**

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Baseline (SVR) | 0.175 | 0.199 | 0.044 |
| Transformer | 0.171 | 0.195 | 0.084 |

**Interpretation**: 
- MAE of 0.17 means predictions are off by 17% on average
- R² near 0 means model barely beats predicting the mean
- This is a **hard task** - raw text doesn't strongly signal stance ratio

**Stance Category Classification**

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Baseline (SVC) | 73.8% | 40.3% |
| Transformer | 68.8% | 27.2% |

**Interpretation**:
- Baseline beats transformer! (Surprising)
- Low Macro-F1 due to failing on minority classes
- "mostly_against" class (2% of data) almost never predicted correctly

**Why Baseline Beats Transformer?**:
1. Small training data (274 essays) - transformers need more data
2. TF-IDF captures argument-specific vocabulary (e.g., "however", "although")
3. Transformer truncates essays - loses information

---

## How to Run

### Setup

```bash
# Navigate to project
cd argument-stance-mining

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Inference on Test Essays

```bash
# Test a specific essay
python -m src.inference --essay_id essay004

# Test multiple essays from test split
python -m src.inference --test_sample 10

# Show essay text snippet
python -m src.inference --essay_id essay005 --show_text

# Run with sentence importance analysis
python -m src.inference --essay_id essay004 --attribution
```

### Sentence Attribution Analysis (NEW)

The `--attribution` flag analyzes which sentences contribute most to the model's prediction:

```bash
python -m src.inference --essay_id essay005 --attribution
```

**Sample Output:**
```
🔍 SENTENCE IMPORTANCE ANALYSIS
   (Which sentences drive the transformer's prediction?)

   Top 5 Most Important Sentences:
   1. [0.114] "Living and studying overseas - It is every student's desire to study..."
   5. [0.113] "Compared to the peers studying in the home country, it will be more..."
   4. [0.110] "First, studying at an overseas university gives individuals the opp..."
```

This also generates an HTML visualization at `reports/attribution_<essay_id>.html` with color-coded sentences (darker blue = more important).

### Rhetorical Role Analysis (NEW)

Analyze what rhetorical role each component plays:

```bash
python -m src.inference --essay_id essay004 --rhetorical
```

**Sample Output:**
```
🎭 RHETORICAL ROLE ANALYSIS
   (What role does each component play in the argument?)

   Predicted Role Distribution:
      EVIDENCE            : 8
      MAIN_ARGUMENT       : 2
      COUNTER_ARGUMENT    : 1

   Prediction Accuracy: 5/11 (45.5%)

   Sample Component Predictions:
      [EVIDENCE          ] ✓ "Studies show that renewable energy creates more jobs..."
      [MAIN_ARGUMENT     ] ✓ "Competition builds essential life skills..."
      [COUNTER_ARGUMENT  ] ✓ "However, some argue that costs are too high..."
```

### Full Analysis

Run all analyses together:

```bash
python -m src.inference --essay_id essay005 --full
```

This runs both sentence attribution AND rhetorical role analysis.

### Example Output

```
======================================================================
ESSAY ANALYSIS: essay005
======================================================================

📝 ESSAY INFO
   Prompt: The idea of going overseas for university study is an exciting prospect...
   Text length: 1654 characters, ~262 words
   Split: TEST

📊 GROUND TRUTH (from BRAT annotations)
   For Claims: 3
   Against Claims: 1
   Total Claims: 4
   For Ratio: 0.750
   Stance Category: mostly_for
   Evidence Density: 1.50 premises/claim
   Attack Ratio: 0.333

🤖 BASELINE MODEL PREDICTIONS (TF-IDF + SVC/SVR)
   For Ratio: 0.802 (error: 0.052)
   Stance Category: mostly_for ✓

🧠 TRANSFORMER PREDICTIONS (DistilBERT)
   For Ratio: 0.795 (error: 0.045)
   Stance Category: mostly_for ✓
```

### Run Full Pipeline

```bash
# Parse and build dataset
python -m src.data.dataset_builder

# Extract features
python -m src.analysis.essay_features

# Run EDA
python -m src.analysis.eda

# Run clustering
python -m src.analysis.clustering

# Train baseline models
python -m src.models.essay_predictor_baseline

# Train transformer models
python -m src.models.essay_predictor_transformer
```

---

## Results Summary

### What We Learned

1. **Most student essays are one-sided** (65% strongly support their thesis)
2. **Counter-arguments are rare** (only 18% of claims are "Against")
3. **4 distinct writing patterns exist**:
   - One-sided with minimal evidence
   - One-sided with rich evidence
   - Balanced/dialectical
   - Attack-heavy
4. **Predicting structure from text is hard** - models achieve only ~74% accuracy
5. **Traditional ML beats transformers** on small datasets

### Generated Outputs

| Output | Location | Description |
|--------|----------|-------------|
| Parsed data | `data/processed/essays_parsed.json` | All essays as JSON |
| Features | `data/processed/essay_features.json` | 36 features per essay |
| Clustered essays | `reports/essays_with_clusters.csv` | Essay IDs with cluster labels |
| Baseline results | `reports/baseline_results.json` | Model metrics |
| Transformer results | `reports/transformer_results.json` | Model metrics |
| Visualizations | `reports/figures/*.png` | 12 analysis plots |
| Final report | `reports/FINAL_REPORT.md` | Summary findings |

---

## Key Decisions and Changes

### Decision 1: Focus on Essay-Level Analysis

**Original Plan**: Train models for component-level stance classification.

**Change**: Since all Claims already have stance labels, this was unnecessary. We pivoted to essay-level analysis which provides more interesting insights.

### Decision 2: Use K=4 for Clustering

**Options Considered**: K=2 to K=10

**Choice**: K=4 based on:
- Elbow method showed diminishing returns after K=4
- Silhouette score peaked around K=4
- 4 clusters have interpretable meanings

### Decision 3: Traditional ML as Primary Baseline

**Why Not Start with Transformers?**
- Small dataset (274 training examples)
- TF-IDF + SVM is a strong baseline for text classification
- Faster to iterate and debug

**Result**: Baseline actually outperformed transformer on classification!

### Decision 4: SVR/SVC over Random Forest

**Models Tested**: Ridge, Random Forest, SVR, SVC

**Choice**: SVM variants performed best because:
- High-dimensional sparse data (TF-IDF)
- Margin-based classifier handles this well
- Balanced class weights help with imbalance

### Change: Fixed Tree Depth Calculation

**Problem**: Essays without clear paths to MajorClaim got `inf` depth.

**Fix**: Return 0 instead, representing "no hierarchical structure".

### Change: Adjusted Stance Category Thresholds

**Original**: 
- mostly_for: for_ratio > 0.7
- mostly_against: for_ratio < 0.3

**Revised**:
- mostly_for: for_ratio >= 0.8
- mostly_against: for_ratio <= 0.2 AND against_count > 0

**Why**: Better aligned with actual data distribution.

---

## Limitations

### 1. Small Dataset
Only 402 essays → not enough for deep learning to shine.

### 2. Severe Class Imbalance
70% of essays are "mostly_for" → models biased toward majority class.

### 3. Text Truncation
Essays exceed 512 tokens → transformer loses information.

### 4. No Component-Level Importance
Current models predict essay-level aggregates, not which sentences matter most.

### 5. Limited Generalization
Models trained on student essays may not transfer to professional writing.

---

## File Structure

```
argument-stance-mining/
├── src/
│   ├── data/
│   │   ├── brat_parser.py       # Parse BRAT annotations
│   │   └── dataset_builder.py   # Build unified dataset
│   ├── analysis/
│   │   ├── essay_features.py    # Extract 36 features
│   │   ├── eda.py               # Exploratory analysis
│   │   ├── clustering.py        # K-means, hierarchical
│   │   └── insights.py          # Generate reports
│   ├── models/
│   │   ├── essay_predictor_baseline.py    # TF-IDF + SVM
│   │   └── essay_predictor_transformer.py # DistilBERT
│   └── inference.py             # Run predictions
├── models/                       # Saved model weights
├── reports/                      # Results and figures
├── data/processed/               # Processed JSON data
├── dataset/                      # Raw AAEC data
├── requirements.txt              # Python dependencies
└── PROJECT_DOCUMENTATION.md      # This file
```

---

*Documentation generated: December 2024*
*Project: Essay Stance Structure Analysis*
*Course: Advanced NLP*

