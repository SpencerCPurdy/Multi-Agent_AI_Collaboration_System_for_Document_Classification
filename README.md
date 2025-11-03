# Multi-Agent AI Collaboration System for Document Classification

A machine learning system that implements genuine multi-agent collaboration for document classification. Three specialized ML models (agents) with different architectures work together through ensemble methods to classify documents into 20 categories from the newsgroups dataset.

## About

This portfolio project demonstrates multi-agent machine learning by training three distinct models that collaborate to achieve better classification performance than individual models alone. Each agent specializes in different aspects of text analysis, and their predictions are combined through ensemble methods.

**Author:** Spencer Purdy  
**Development Environment:** Google Colab Pro (A100 GPU, High RAM)

## Features

- **Three Specialized Agents**:
  - TF-IDF Agent: Uses statistical text features with Logistic Regression
  - Embedding Agent: Leverages semantic embeddings with a neural network
  - XGBoost Agent: Handles mixed features with gradient boosting
- **Ensemble Coordination**: Weighted voting and stacking meta-learner
- **Agent Voting System**: Shows individual agent predictions and consensus
- **Interactive Interface**: Gradio web application with real-time classification
- **Comprehensive Evaluation**: Performance metrics, confusion matrix, and agent comparison
- **Visualization**: Confidence scores and prediction distributions

## Dataset

- **Source:** 20 Newsgroups Dataset (via scikit-learn)
- **License:** Public domain
- **Total Documents:** ~18,000 newsgroup posts
- **Categories:** 20 (technology, sports, politics, religion, science, etc.)
- **Task:** Multi-class text classification
- **Preprocessing:** Removal of headers, footers, and quotes

## System Performance

Performance on held-out test set (3,770 documents):

| Model | Accuracy | F1-Score (Weighted) |
|-------|----------|---------------------|
| TF-IDF Agent | 66.18% | 0.6589 |
| Embedding Agent | 72.45% | 0.7224 |
| XGBoost Agent | 61.44% | 0.6147 |
| Weighted Voting | ~71% | ~0.70 |
| Stacking Ensemble | **73%** | **0.73** |

**Best Performance:** Stacking ensemble achieves 73% accuracy by learning optimal agent weighting

**Training Set Size:** 12,060 documents  
**Validation Set Size:** 3,016 documents  
**Test Set Size:** 3,770 documents

## Agent Architectures

### TF-IDF Agent
- **Feature Extraction:** 5,000 TF-IDF features with bigrams
- **Model:** Logistic Regression with L2 regularization
- **Training Time:** ~16.53 seconds
- **Strengths:** Fast, interpretable, keyword-based classification

### Embedding Agent
- **Feature Extraction:** 384-dimensional sentence embeddings (all-MiniLM-L6-v2)
- **Model:** 2-layer neural network (384 → 256 → 128 → 20)
- **Training Time:** ~7.74 seconds
- **Strengths:** Captures semantic similarity, handles paraphrasing

### XGBoost Agent
- **Features:** Combined TF-IDF + embeddings + metadata
- **Model:** Gradient boosting (200 estimators, max depth 6)
- **Training Time:** ~632.16 seconds
- **Strengths:** Robust with mixed features, handles complex patterns

### Meta-Learner (Stacking)
- **Input:** Predictions from all three agents
- **Model:** Logistic Regression
- **Purpose:** Learns optimal combination of agent predictions

## Technical Stack

- **ML Frameworks:** scikit-learn, PyTorch, XGBoost
- **NLP:** sentence-transformers, nltk
- **Data Processing:** pandas, numpy
- **Class Balancing:** imbalanced-learn (SMOTE)
- **UI Framework:** Gradio
- **Visualization:** matplotlib, seaborn, plotly
- **Development:** Google Colab Pro with A100 GPU

## Setup and Usage

### Running in Google Colab

1. Clone this repository or download the notebook file
2. Upload `Multi-Agent AI Collaboration System for Document Classification.ipynb` to Google Colab
3. Select Runtime > Change runtime type > A100 GPU (or T4 GPU for free tier)
4. Run all cells sequentially

The notebook will automatically:
- Install required dependencies
- Download the 20 Newsgroups dataset
- Train all three agents
- Train ensemble methods
- Evaluate on test set
- Launch a Gradio interface with a shareable link

### Running Locally

```bash
# Clone the repository
git clone https://github.com/SpencerCPurdy/Multi-Agent_AI_Collaboration_System_for_Document_Classification.git
cd Multi-Agent_AI_Collaboration_System_for_Document_Classification

# Install dependencies
pip install scikit-learn==1.3.0 numpy==1.24.3 pandas==2.0.3 torch==2.1.0 transformers==4.35.0 gradio==4.7.1 sentence-transformers==2.2.2 imbalanced-learn==0.11.0 xgboost==2.0.1 plotly==5.18.0 seaborn==0.13.0 nltk==3.8.1

# Run the notebook
jupyter notebook "Multi-Agent AI Collaboration System for Document Classification.ipynb"
```

**Note:** Training takes approximately 10-15 minutes depending on hardware.

## Project Structure

```
├── Multi-Agent AI Collaboration System for Document Classification.ipynb
├── README.md
├── LICENSE
└── .gitignore
```

The notebook contains the following components:

1. **Configuration & Setup**: System parameters and reproducibility settings
2. **Data Loading**: 20 Newsgroups dataset with preprocessing
3. **Feature Engineering**: TF-IDF, embeddings, and metadata features
4. **Agent Training**: Three specialized models trained independently
5. **Ensemble Methods**: Voting and stacking implementation
6. **Evaluation**: Comprehensive metrics and visualizations
7. **Gradio Interface**: Interactive web application

## Key Implementation Details

- **Reproducibility:** All random seeds set to 42 for deterministic results
- **Cross-Validation:** 5-fold stratified cross-validation for model selection
- **Feature Engineering:** Combined TF-IDF (5,000 features), sentence embeddings (384-d), and document metadata
- **Class Balancing:** SMOTE applied to handle class imbalance
- **Neural Network:** Dropout (0.3) and early stopping (patience: 3 epochs) to prevent overfitting

## Performance by Category

The system achieves varying performance across categories:

**Strong Performance (>85% precision):**
- rec.sport.hockey: 94% precision
- rec.sport.baseball: 89% precision
- comp.windows.x: 87% precision

**Moderate Performance (70-85% precision):**
- sci.crypt: 84% precision
- sci.med: 83% precision
- comp.graphics: 70% precision

**Challenging Categories (<60% precision):**
- talk.religion.misc: 38% precision
- comp.os.ms-windows.misc: Lower performance due to overlapping topics

## Limitations

### Domain Specificity
- Trained on newsgroup data; may not generalize well to significantly different domains (e.g., legal documents, medical reports)

### Performance Constraints
- 73% accuracy is solid but not state-of-the-art for text classification
- Performance degrades on very short documents (<50 words)
- Ambiguous documents covering multiple topics may be misclassified

### Known Issues
- Training data bias reflected in model predictions
- English text only
- Very long documents (>10,000 words) may lose context
- Sarcasm and irony not reliably detected

### Uncertainty Indicators
- Confidence <50%: Highly uncertain prediction, consider human review
- Close top-2 predictions: Document may belong to multiple categories
- Agent disagreement: Complex or ambiguous document

## Ensemble Strategy

The system uses two ensemble approaches:

1. **Weighted Voting**: Combines predictions based on validation performance
   - Simple and interpretable
   - Each agent weighted by validation accuracy

2. **Stacking**: Meta-learner optimally combines agent predictions
   - Learns complex agent interaction patterns
   - Achieves best performance (~73% accuracy)
   - Meta-learner uses Logistic Regression with 5-fold cross-validation

## Use Cases

This multi-agent approach is applicable to:
- Customer support ticket routing
- Email categorization
- Content moderation
- Document management systems
- News article classification

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- 20 Newsgroups dataset creators
- scikit-learn team for dataset hosting
- Hugging Face for sentence-transformers
- Open-source ML community

## Contact

**Spencer Purdy**  
GitHub: [@SpencerCPurdy](https://github.com/SpencerCPurdy)

---

*This is a portfolio project developed to demonstrate multi-agent machine learning and ensemble methods. The system is designed for educational and demonstrational purposes. Performance metrics reflect results on the specific dataset used.*
