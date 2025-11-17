# SkillGapPipeline
A project that allows users input skills they currently have for a particular job role and recommends additional skills needed to perform their tasks at those jobs efficiently.

## Recommendation Model

This project includes a skill-based job recommendation model that matches user skills to relevant job postings. The model supports two approaches:

1. **TF-IDF + Nearest Neighbors** (scikit-learn) - Fast keyword-based matching
2. **Embeddings + Nearest Neighbors** (sentence-transformers) - Semantic understanding with better context

### Quick Start

#### Basic Usage

```python
from src.recommendation_model import SkillRecommendationModel
import pandas as pd

# Load your job data
df = pd.read_csv('data/all_job_post.csv')

# Create and train TF-IDF model
model = SkillRecommendationModel(method='tfidf')
model.fit(df)

# Get recommendations
user_skills = ['python', 'pandas', 'sql', 'machine learning']
recommendations = model.recommend(
    user_skills, 
    n_recommendations=5, 
    return_scores=True
)

print(recommendations[['job_title', 'category', 'similarity_score']])
```

#### Using Embeddings Model

```python
# For better semantic understanding
embedding_model = SkillRecommendationModel(method='embeddings')
embedding_model.fit(df)

recommendations = embedding_model.recommend(user_skills, n_recommendations=5)
```

#### Save and Load Models

```python
# Save trained model
model.save('models/my_model.pkl')

# Load saved model
loaded_model = SkillRecommendationModel.load('models/my_model.pkl')
recommendations = loaded_model.recommend(user_skills)
```

### Training Scripts

Train models using the provided scripts:

```bash
# Train both TF-IDF and embeddings models
python src/train_model.py

# Or use the example script
python src/example_usage.py
```

### Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `scikit-learn>=1.2` - For TF-IDF and nearest neighbors
- `sentence-transformers>=2.2.2` - For embeddings (optional but recommended)
- `pandas>=1.5` - Data handling
- `numpy>=1.24` - Numerical operations

### Notebook Usage

See `etl/extract.ipynb` for a complete example with EDA and model training.
