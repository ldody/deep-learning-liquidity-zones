# Deep Learning for Liquidity Zone Detection and Prediction

Research code for identifying **high-liquidity zones in full limit order book data**, studying their relationship with subsequent price dynamics, and predicting their future location from **OHLCV data alone** using machine learning and deep learning.

The project combines market microstructure, unsupervised learning, time-series deep learning, statistical evaluation, and execution-oriented backtesting. It was developed as part of academic research and forms the basis of a research article currently in the publication process.

> **Portfolio note:** this repository is intended to showcase the methodology, software architecture, and technical skills behind the research. Empirical results from the associated manuscript are intentionally not reported here.

## Project Overview

The workflow is organized around three main research components:

1. **Liquidity-zone identification** — reconstruct and preprocess full order book data, then detect dense liquidity regions with HDBSCAN.
2. **Liquidity-zone prediction** — learn to predict those latent order-book structures using historical OHLCV information and a hybrid deep-learning architecture.
3. **Economic validation** — evaluate predicted liquidity zones through an execution-oriented backtesting framework and compare the approach with benchmark methods.

The codebase also includes statistical evaluation of the detected clusters, clustering analysis, hyperparameter optimization, classical comparison models, and Slurm scripts for HPC execution.

## Main Technical Skills Demonstrated

- **Market microstructure:** full order book reconstruction, limit order book processing, liquidity analysis
- **Unsupervised learning:** HDBSCAN density-based clustering
- **Deep learning:** TensorFlow / Keras, convolutional layers, bidirectional GRU, attention / Transformer components
- **Machine learning benchmarks:** classification models and technical-analysis baselines
- **Time-series processing:** OHLCV and high-frequency market data
- **Statistical analysis:** ANOVA, cluster evaluation, correlation-based analysis
- **Model evaluation:** precision, recall, F1-oriented metrics and interval-overlap measures
- **Hyperparameter optimization:** Optuna
- **Economic validation:** execution-oriented backtesting
- **Large-scale computing:** Slurm job arrays and HPC workflows
- **Data engineering:** compressed market-data ingestion, preprocessing, Parquet-based intermediate datasets

## Methodology

### 1. Full Order Book Reconstruction

The raw market-data pipeline is implemented under `src/data/`.

`FOBDBM.py` manages the Full Order Book database, while `prepro_FOB.py` transforms the raw event data into datasets used by the downstream analysis.

The preprocessing pipeline separates and reconstructs market information such as:

- limit order book states,
- filled orders,
- bid/ask information,
- prices and order sizes.

Because the original market data are licensed, the raw datasets are **not distributed with this repository**.

### 2. Liquidity-Zone Identification

Liquidity zones are identified from order-book information using **HDBSCAN**, a density-based clustering algorithm.

The clustering workflow is implemented primarily in:

```text
src/models/clustering.py
src/models/clustering_main.py
src/models/clustering_preprocessing.py
src/models/clustering_postprocessing.py
```

The preprocessing stage transforms order-book observations before clustering, while the post-processing stage converts detected clusters into liquidity-zone representations suitable for analysis and prediction.

### 3. Statistical Evaluation and Liquidity Analysis

The repository contains dedicated modules for evaluating the clustering procedure and studying the relationship between identified liquidity zones and subsequent market behavior:

```text
src/evaluation/
src/analysis/
```

This includes cluster-separation evaluation and downstream analysis of the detected liquidity structures.

### 4. Liquidity-Zone Prediction from OHLCV

The second part of the project treats liquidity-zone prediction as a supervised learning problem.

Instead of requiring order-book information at prediction time, the model uses historical **OHLCV sequences** to infer the location of latent liquidity zones.

The main neural architecture combines:

- convolutional feature extraction,
- bidirectional GRU layers for sequential dependencies,
- attention / Transformer components for longer-range interactions,
- dense output layers for liquidity-zone prediction.

The implementation is located mainly in:

```text
src/models/regression_ANN.py
src/models/regression_main.py
src/models/regression_preprocessing.py
```

Hyperparameter optimization is performed with **Optuna**.

### 5. Benchmark Models

The deep-learning framework is compared with alternative approaches implemented under:

```text
src/comp_models/
```

These include classical classification methods and a pivot-point technical-analysis baseline.

### 6. Execution-Oriented Backtesting

The repository also contains an economic-validation layer:

```text
src/backtest/backtest.py
```

The backtest evaluates whether predicted liquidity information can be incorporated into execution decisions and compares liquidity-aware execution with benchmark execution rules.

The purpose of this component is **execution-quality analysis**, not the presentation of a standalone trading strategy.

## Repository Structure

```text
.
├── requirements.txt
├── deploy_venv.bash
├── slurm/
│   ├── FOB_DB_reinit.bash
│   ├── FOB_prepro.bash
│   ├── clustering.bash
│   ├── clustering_eval.bash
│   ├── clustering_analysis.bash
│   ├── opti.bash
│   ├── pred.bash
│   ├── ANN.bash
│   ├── classification.bash
│   ├── PivotPoints.bash
│   └── backtest.bash
├── src/
│   ├── data/
│   │   ├── FOBDBM.py
│   │   └── prepro_FOB.py
│   ├── models/
│   │   ├── clustering.py
│   │   ├── clustering_main.py
│   │   ├── clustering_preprocessing.py
│   │   ├── clustering_postprocessing.py
│   │   ├── regression_ANN.py
│   │   ├── regression_CNN.py
│   │   ├── regression_main.py
│   │   └── regression_preprocessing.py
│   ├── evaluation/
│   │   └── clustering_eval.py
│   ├── analysis/
│   │   ├── clustering_analysis.py
│   │   ├── clustering_analysis_main.py
│   │   └── clustering_analysis_preprocessing.py
│   ├── comp_models/
│   │   ├── classification.py
│   │   └── pivot_points.py
│   └── backtest/
│       └── backtest.py
└── tests/
```

Generated data, trained models, intermediate results, and licensed market datasets are intentionally not included in the public repository.

## Pipeline

```text
Euronext Full Order Book
          │
          ▼
  FOB reconstruction
          │
          ▼
 Order-book preprocessing
          │
          ▼
   HDBSCAN clustering
          │
          ├──────────────► Statistical evaluation
          │
          ▼
  Liquidity-zone labels
          │
          ├──────────────► Liquidity-zone analysis
          │
          ▼
OHLCV ──► Deep-learning prediction
          │
          ├──────────────► Benchmark comparison
          │
          ▼
 Execution-oriented backtest
```

## Environment

The original HPC workflow uses **Python 3.10.4**.

Create the virtual environment at the root of the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main dependencies include:

```text
pandas
numpy
tensorflow
scikit-learn
hdbscan
optuna
pyarrow
fastparquet
matplotlib
tqdm
ta
```

### Important Repository-Name Compatibility Note

Several original research scripts locate the project root by searching for a directory named:

```text
PhD_article_2
```

The **GitHub repository itself can be renamed** for a cleaner portfolio presentation, but the local checkout should retain the directory name expected by the original code.

For example, if the public repository is named `deep-learning-liquidity-zones`:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL> PhD_article_2
cd PhD_article_2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This preserves the original source code without changing its path assumptions.

## Data Requirements

The research pipeline expects market data under the project `data/` directory, including Full Order Book and OHLCV inputs.

Conceptually:

```text
data/
├── assets.csv
├── raw/
│   ├── FOB/
│   └── OHLCV/
└── processed/
    └── FOB/
        ├── LOB/
        └── FO/
```

The exact datasets used for the research are not redistributed. Full Order Book data are proprietary/licensed market data, and the OHLCV research inputs are likewise not included in the repository.

The associated manuscript uses high-frequency market data for CAC 40 equities, combining Euronext limit-order-book information with OHLCV data.

## Running the Research Pipeline

The repository was designed primarily for the original research/HPC environment rather than as a packaged end-user application. The principal stages are exposed through the Python entry points and their corresponding Slurm scripts.

Typical stages include:

```bash
# Reinitialize the Full Order Book database
python src/data/FOBDBM.py --reinit True

# Preprocess Full Order Book data
python src/data/prepro_FOB.py -sa True --job_id <JOB_ID>

# Identify liquidity zones
python src/models/clustering_main.py -sa True --job_id <JOB_ID>

# Evaluate clustering
python src/evaluation/clustering_eval.py

# Analyze detected liquidity zones
python src/analysis/clustering_analysis_main.py -sa True --job_id <JOB_ID>

# Hyperparameter optimization
python src/models/regression_main.py -ba True --job_id <JOB_ID>

# Generate deep-learning predictions
python src/models/regression_main.py -p True

# Run benchmark models
python src/comp_models/classification.py -sa True
python src/comp_models/pivot_points.py -sa True

# Economic validation
python src/backtest/backtest.py
```

These commands document the original entry points. Successful reproduction requires the corresponding licensed datasets, expected intermediate files, asset metadata, and computing environment.

## HPC / Slurm Workflow

The `slurm/` directory contains the job scripts used for the original large-scale experiments.

Examples include:

```text
FOB_DB_reinit.bash
FOB_prepro.bash
clustering.bash
clustering_eval.bash
clustering_analysis.bash
opti.bash
pred.bash
classification.bash
PivotPoints.bash
backtest.bash
```

The scripts contain **institution-specific HPC configuration**, including module loading, resource allocation, and original filesystem paths. They are preserved as part of the research workflow and must be adapted before use on another cluster.

## Reproducibility Scope

This repository provides the **research implementation and computational workflow**, but it is not intended as a turnkey reproduction package.

Reproduction depends on:

- access to the original licensed Full Order Book data,
- compatible OHLCV inputs,
- the expected asset metadata and intermediate datasets,
- sufficient computational resources,
- adaptation of the original Slurm environment when running on another HPC system.

The source code is preserved in its original research form to maintain consistency with the experiments on which the associated manuscript is based.

## Research Manuscript

This project forms the computational basis of the research manuscript:

**“Deep Learning and High-Liquidity Zones in Market Depth”**

The manuscript is **currently in the publication process**.

It studies a framework for identifying liquidity zones from full order book data, predicting those zones from historical OHLCV information, and assessing their economic relevance through execution-oriented analysis.

> Results from the manuscript are intentionally not reproduced in this README while the article is in the publication process.

## Data Source

Full Order Book data used in the research were obtained from **Euronext**. Access to the underlying historical market data is subject to the data provider's licensing and distribution conditions.

## Disclaimer

This repository is provided for **research, academic, and portfolio purposes**. It does not constitute investment advice, a recommendation to trade, or a production-ready trading system.

The code reflects the computational environment and research workflow used during the project. External users may need to adapt paths, data interfaces, HPC settings, and environment-specific configuration.

## Author

**Léo Dody**

Research interests: financial machine learning, market microstructure, deep learning, high-frequency data, liquidity modeling, and quantitative finance.
