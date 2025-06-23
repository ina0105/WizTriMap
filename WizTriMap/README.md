# movie-recommender

This project is a movie recommendation system that utilizes collaborative filtering techniques to suggest movies to users based on their preferences and interactions with the dataset.

## Project Structure

```
movie-recommender
├── src
│   ├── data
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── features
│   │   ├── __init__.py
│   │   ├── user_features.py
│   │   ├── item_features.py
│   │   └── advanced_features.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── splitter.py
│   ├── models
│   │   └── ranking_models.py
│   ├── preprocess.py
│   ├── train_advanced.py
│   ├── train.py
│   ├── evaluate_mmr_advanced.py
│   ├── evaluate.py
│   ├── evaluate_mmr.py
│   ├── baseline.py
│   └── evaluate_advanced.py
├── data
│   ├── ratings.dat
│   ├── movies.dat
│   └── users.dat
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

To install the required dependencies, run:
```
conda env create -f environment.yml
conda activate recommender
```
Or install the requirements:

```
pip install -r requirements.txt
```

## Full Pipeline

The movie recommendation system follows a complete pipeline:

1. **Data Loading**: Load datasets from `data/` directory using [`src/data/loader.py`](src/data/loader.py)
2. **Data Preprocessing**: Clean and prepare data using [`src/data/preprocessor.py`](src/data/preprocessor.py)
3. **Feature Engineering**: Generate user and item features using [`src/features/`](src/features/)
4. **Data Splitting**: Split data chronologically into training/validation/test sets
5. **Model Training**: Train recommendation models with different approaches
6. **Evaluation**: Assess model performance using Precision@K, Recall@K, NDCG@K, Diversity, Novelty and Coverage@K metrics for ranking 

## Training Options

The system supports three different training modes, each with increasing complexity:

### 1. Normal Train (Basic LightGBM + LambdaRank)
Standard collaborative filtering approach using LightGBM with basic user and item features.

**Features included:**
- Basic user demographics and rating patterns
- Item popularity and genre information
- Simple interaction features

**Models:**
- LightGBM Ranker with lambdarank objective
- NDCG optimization for ranking quality

**Usage:**
```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

**Output:**
- LightGBM Model: `models/lgbm_model.txt`
- Metrics: `models/metrics.pkl`
- Feature importance: `models/feature_importance.csv`

### 2. Advanced Features Train (Enhanced Feature Engineering + Ranking)
Incorporates advanced feature engineering and learning-to-rank techniques for state-of-the-art recommendations.

**Advanced features include:**
- **Temporal Features**: Time-based patterns, seasonality, interaction sequences
- **Statistical Features**: Advanced user/item rating distributions and percentiles
- **Matrix Factorization**: SVD-based latent factor features
- **Graph Features**: User-item interaction network properties
- **Sequence Features**: Rating trends and session-based patterns
- **Interaction Features**: Complex user-item relationship modeling

**Models:**
- LightGBM Ranker with lambdarank objective
- NDCG optimization for ranking quality

**Usage:**
```bash
python src/train_advanced.py
python src/evaluate_advanced.py
```

**Output:**
- Advanced processed data: `../data/processed/advanced_preprocessed_data.pkl`
- Ranking model: `models/lgb_ranker.txt`
- Advanced evaluation results: `models/advanced_*_metrics_at_10.pkl`

### 3.MMR Re-ranking Pipeline

The system includes an advanced MMR (Maximal Marginal Relevance) re-ranking pipeline that balances relevance and diversity in recommendations. This addresses the common problem where recommendation systems suggest very similar items.

What is MMR?

MMR re-ranking optimizes the trade-off between:
- **Relevance**: How well items match user preferences (from LightGBM predictions)
- **Diversity/Novelty**: How different/novel the recommended items are

The MMR formula: `MMR = λ × Relevance + (1-λ) × Diversity`

Where λ controls the balance:
- λ = 0.3: More diversity-focused recommendations
- λ = 0.5: Balanced approach
- λ = 0.7: More relevance-focused recommendations

**Usage:**
```bash
python src/evaluate_mmr.py
python src/evaluate_mmr_advanced.py
```
## Data
The datasets used in this project are:

- [`data/ratings.dat`](data/ratings.dat): Contains user ratings for movies
- [`data/movies.dat`](data/movies.dat): Contains information about movies
- [`data/users.dat`](data/users.dat): Contains user demographic information

## Evaluation Metrics

All models are evaluated using:
- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended  
- **NDCG@K**: Normalized Discounted Cumulative Gain (ranking quality)
- **Diversity**: Measures how different the recommended items are from each other
- **Novelty**: Assesses how unexpected or less popular the recommended items are
- **Coverage@K**: Proportion of unique items recommended across all users

## Model Outputs

Trained models and results are saved in the `models/` directory:
- Basic model files and metrics
- Advanced ranking models and evaluation results

## Perfecto Solana Morenilla
