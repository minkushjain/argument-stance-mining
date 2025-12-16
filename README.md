# Argument Stance Mining in Persuasive Essays

**A computational approach to analyzing argument structure and stance patterns in student essays**

## Authors

- **Minkush Jain** — UC Berkeley
- **Aratrik Paul** — UC Berkeley
- **Vikramsingh Rathod** — UC Berkeley

## Overview

This project investigates how arguments are structured in persuasive essays by analyzing stance patterns, rhetorical strategies, and argument flow. We develop a comprehensive pipeline that:

1. **Parses** argument annotations from the BRAT format
2. **Extracts** 36 numerical features capturing stance, structure, relations, and position
3. **Clusters** essays into 4 distinct argumentation patterns
4. **Predicts** essay-level stance characteristics using both traditional ML and transformer models
5. **Analyzes** sentence-level importance and rhetorical roles

## Installation

### Requirements

- Python 3.10 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/argument-stance-mining.git
cd argument-stance-mining

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Dataset

This project uses the **Argument Annotated Essays Corpus (AAEC) v2.0**.

### Download Instructions

1. Visit the [AAEC dataset page](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422)
2. Download `ArgumentAnnotatedEssays-2.0.zip`
3. Extract to `dataset/ArgumentAnnotatedEssays-2.0/`
4. Ensure the following structure:
   ```
   dataset/
   └── ArgumentAnnotatedEssays-2.0/
       ├── brat-project-final/
       │   ├── essay001.ann
       │   ├── essay001.txt
       │   └── ...
       ├── prompts.csv
       ├── train-test-split.csv
       └── README.txt
   ```

### Dataset Statistics

| Metric | Count |
|--------|-------|
| Total Essays | 402 |
| Train/Val/Test Split | 274 / 48 / 80 |
| Argument Components | 6,089 |
| Claims with Stance | 1,506 |
| Relations | 3,832 |

## Usage

### Running the Full Pipeline

```bash
# Step 1: Parse dataset and build structured data
python -m src.data.dataset_builder

# Step 2: Extract essay-level features
python -m src.analysis.essay_features

# Step 3: Run exploratory data analysis
python -m src.analysis.eda

# Step 4: Perform clustering analysis
python -m src.analysis.clustering

# Step 5: Train baseline model (TF-IDF + SVM)
python -m src.models.essay_predictor_baseline

# Step 6: Train transformer model (DistilBERT)
python -m src.models.essay_predictor_transformer

# Step 7: Train rhetorical role classifier
python -m src.models.rhetorical_classifier_v2
```

### Running Inference

```bash
# Analyze a specific essay
python -m src.inference --essay_id essay005

# Full analysis with sentence attribution and rhetorical roles
python -m src.inference --essay_id essay005 --full

# Analyze multiple test essays
python -m src.inference --test_sample 10

# Show essay text in output
python -m src.inference --essay_id essay005 --show_text
```

### Inference Options

| Flag | Description |
|------|-------------|
| `--essay_id` | Essay ID to analyze (e.g., `essay005`) |
| `--test_sample N` | Analyze N random essays from test set |
| `--show_text` | Display essay text snippet |
| `--attribution` | Run sentence importance analysis |
| `--rhetorical` | Run rhetorical role classification |
| `--full` | Run all analyses |

## Project Structure

```
argument-stance-mining/
├── src/
│   ├── data/
│   │   ├── brat_parser.py          # BRAT annotation parser
│   │   ├── dataset_builder.py      # Dataset construction
│   │   └── rhetorical_labels.py    # Rhetorical role definitions
│   ├── analysis/
│   │   ├── essay_features.py       # Feature extraction (36 features)
│   │   ├── eda.py                  # Exploratory data analysis
│   │   ├── clustering.py           # K-means & hierarchical clustering
│   │   ├── insights.py             # Insight generation
│   │   └── sentence_attribution.py # Sentence importance analysis
│   ├── models/
│   │   ├── essay_predictor_baseline.py    # TF-IDF + SVM models
│   │   ├── essay_predictor_transformer.py # DistilBERT fine-tuning
│   │   ├── rhetorical_classifier.py       # Component role classifier
│   │   └── rhetorical_classifier_v2.py    # Improved classifier
│   └── inference.py                # Unified inference interface
├── requirements.txt
├── DOCUMENTATION.md                # Detailed technical documentation
└── README.md
```

## Research Questions

This project addresses three main research questions:

### RQ1: What argument structure patterns exist in persuasive essays?

We identified **4 distinct clusters** of essay writing patterns:
- **One-sided For** (39%): Strongly supports thesis with minimal counter-arguments
- **Evidence-rich For** (26%): Supports thesis with extensive evidence
- **Dialectical** (21%): Balanced consideration of both sides
- **Attack-heavy** (13%): Heavy use of attack relations

### RQ2: Can we predict essay stance structure from raw text?

| Task | Baseline (TF-IDF + SVM) | Transformer (DistilBERT) |
|------|------------------------|--------------------------|
| For-Ratio Regression (MAE) | 0.175 | 0.171 |
| Stance Classification (Accuracy) | 73.8% | 68.8% |

**Key Finding**: Traditional ML outperforms transformers on this small dataset due to TF-IDF's ability to capture argument-specific vocabulary.

### RQ3: Which sentences drive stance predictions?

Using gradient-based attribution, we identify sentence-level contributions to model predictions, revealing which rhetorical moves most influence perceived stance.

## Key Results

1. **Most essays are one-sided**: 65% strongly support their thesis
2. **Counter-arguments are rare**: Only 18.5% of claims oppose the thesis
3. **Structure is hard to predict**: Models achieve ~74% accuracy on stance category
4. **Evidence density varies**: 1-6 premises per claim across essays
5. **Position matters**: Counter-arguments typically appear in the middle third

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@inproceedings{jain2024argument,
  title={Argument Stance Mining in Persuasive Essays: 
         A Multi-Level Analysis of Structure and Rhetorical Strategies},
  author={Jain, Minkush and Paul, Aratrik and Rathod, Vikramsingh},
  booktitle={UC Berkeley Advanced NLP Course Project},
  year={2024}
}
```

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 Minkush Jain, Aratrik Paul, Vikramsingh Rathod

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgments

- **Dataset**: [Argument Annotated Essays Corpus](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422) by Stab & Gurevych (2017)
- **Course**: INFO 256 Applied Natural Language Processing, UC Berkeley

