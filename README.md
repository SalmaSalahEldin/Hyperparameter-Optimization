# Academic Performance Analytics API

A comprehensive machine learning API for predicting student math scores based on various academic and demographic factors. This project demonstrates the **critical importance of hyperparameter optimization** in machine learning by comparing baseline models (default parameters) with optimized models (hyperparameter tuning).

## 🎯 **Why Hyperparameter Optimization Matters**

This project provides **concrete evidence** that proper hyperparameter tuning can significantly improve model performance. We maintain two separate sets of models:

1. **Baseline Models**: Trained with default scikit-learn parameters
2. **Optimized Models**: Trained with GridSearchCV hyperparameter optimization

### **Performance Improvements Achieved**

| Model | Baseline R² | Optimized R² | Improvement | % Improvement |
|-------|-------------|--------------|-------------|---------------|
| **Decision Tree** | 0.7313 | **0.8241** | **+0.0927** | **+12.68%** |
| **Lasso Regression** | 0.8253 | **0.8809** | **+0.0556** | **+6.73%** |
| **Random Forest** | 0.8488 | **0.8550** | **+0.0061** | **+0.72%** |
| **K-Neighbors** | 0.7771 | **0.7816** | **+0.0045** | **+0.58%** |
| **Ridge Regression** | 0.8805 | **0.8805** | +0.0000 | +0.00% |

### **Key Insights**

- **Decision Tree**: Most dramatic improvement (+12.68%) - shows how default parameters can severely underperform
- **Lasso Regression**: Significant improvement (+6.73%) - demonstrates the impact of proper regularization tuning
- **Random Forest**: Moderate improvement (+0.72%) - shows how ensemble parameters matter
- **Ridge Regression**: Already optimal - demonstrates when default parameters work well

## 🏗️ **Project Structure**

```
mlproject/
├── app.py                      # FastAPI application
├── retrain_model.py            # Comprehensive model training script
├── requirements.txt            # Python dependencies
├── artifacts/                  # All trained models and metrics
│   ├── baseline_models/        # Models with default parameters
│   │   ├── Linear_Regression.pkl
│   │   ├── Ridge_Regression.pkl
│   │   ├── Lasso_Regression.pkl
│   │   ├── Random_Forest.pkl
│   │   ├── KNeighbors.pkl
│   │   └── Decision_Tree.pkl
│   ├── optimized_models/       # Models with hyperparameter tuning
│   │   ├── Ridge_Regression.pkl
│   │   ├── Lasso_Regression.pkl
│   │   ├── Random_Forest.pkl
│   │   ├── KNeighbors.pkl
│   │   └── Decision_Tree.pkl
│   ├── baseline_preprocessors/ # Preprocessors for baseline models
│   │   └── preprocessor.pkl
│   ├── optimized_preprocessors/ # Preprocessors for optimized models
│   │   └── preprocessor.pkl
│   ├── metrics/                # Detailed performance metrics
│   │   ├── baseline_metrics.json      # Baseline model detailed metrics
│   │   ├── optimized_metrics.json     # Optimized model detailed metrics
│   │   ├── model_comparison.json      # Performance comparison analysis
│   │   └── training_summary.json      # Training summary and recommendations
│   └── best_model.pkl         # Best performing optimized model
├── src/                        # Source code
│   ├── components/             # ML pipeline components
│   ├── pipeline/               # Prediction pipeline
│   └── utils.py                # Utility functions
└── notebook/                   # Jupyter notebooks
```

## 🚀 **Quick Start**

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models (Baseline + Optimized)
```bash
python retrain_model.py
```

This will:
- Train 6 baseline models with default parameters
- Train 5 optimized models with hyperparameter tuning
- Generate comprehensive performance metrics
- Save everything in organized directories

### 3. Run the API
```bash
python app.py
```

The API will be available at `http://localhost:5001`

## 📊 **Detailed Model Performance**

### **Baseline Models (Default Parameters)**

| Model | Training R² | Test R² | Training RMSE | Test RMSE | Overfitting Score |
|-------|-------------|---------|---------------|-----------|-------------------|
| Linear Regression | 0.8743 | 0.8804 | 5.32 | 5.39 | -0.0061 |
| Ridge Regression | 0.8743 | 0.8805 | 5.32 | 5.39 | -0.0062 |
| Lasso Regression | 0.8071 | 0.8253 | 6.59 | 6.52 | -0.0182 |
| Random Forest | 0.9761 | 0.8488 | 2.32 | 6.07 | **+0.1272** |
| K-Neighbors | 0.8620 | 0.7771 | 5.58 | 7.36 | **+0.0849** |
| Decision Tree | 0.9997 | 0.7313 | 0.28 | 8.09 | **+0.2683** |

### **Optimized Models (Hyperparameter Tuning)**

| Model | CV Score | Test R² | Test RMSE | Overfitting Score | Best Parameters |
|-------|----------|---------|-----------|-------------------|-----------------|
| Ridge Regression | 0.8687 | 0.8805 | 5.39 | -0.0062 | alpha=1.0, solver='auto' |
| Lasso Regression | 0.8686 | 0.8809 | 5.38 | -0.0066 | alpha=0.01, max_iter=1000 |
| Random Forest | 0.8382 | 0.8550 | 5.94 | 0.0738 | n_estimators=200, max_depth=20 |
| K-Neighbors | 0.7942 | 0.7816 | 7.29 | 0.2180 | n_neighbors=9, weights='distance' |
| Decision Tree | 0.7966 | 0.8241 | 6.54 | 0.0308 | max_depth=5, min_samples_leaf=4 |

## 🔍 **What the Metrics Tell Us**

### **Overfitting Analysis**
- **Baseline Decision Tree**: Severe overfitting (0.2683) - memorizes training data
- **Optimized Decision Tree**: Much better generalization (0.0308) - learns patterns
- **Random Forest**: Baseline overfits (0.1272), optimization reduces it (0.0738)

### **Cross-Validation Benefits**
- **Ridge Regression**: CV score (0.8687) closely matches test performance (0.8805)
- **Lasso Regression**: CV score (0.8686) predicts test performance (0.8809) well
- **Decision Tree**: CV score (0.7966) prevents overfitting, improves test performance

### **Parameter Optimization Impact**
- **Lasso alpha**: Changed from 1.0 (default) to 0.01 (optimal) - **+6.73% improvement**
- **Decision Tree depth**: Limited from unlimited to 5 - **+12.68% improvement**
- **Random Forest**: Increased trees from 100 to 200, optimized depth - **+0.72% improvement**

## 🤖 **API Model Information**

### **Which Model Does the API Use?**

**The API uses the OPTIMIZED BEST MODEL** that achieves the highest accuracy:

- **Model**: `artifacts/best_model.pkl` (Ridge Regression with optimized parameters)
- **Preprocessor**: `artifacts/optimized_preprocessors/preprocessor.pkl`
- **Performance**: **R² = 0.8805** (Test), **CV Score = 0.8687**

### **Why This Matters:**

1. **Best Performance**: Uses the model that achieved the highest cross-validation score
2. **Optimized Parameters**: Trained with hyperparameter tuning, not default values
3. **Overfitting Prevention**: Uses cross-validation to ensure reliable performance
4. **Production Ready**: Automatically selects the best performing model

### **Model Selection Process:**

The training script automatically:
1. Trains 6 baseline models with default parameters
2. Trains 5 optimized models with hyperparameter tuning
3. Selects the best model based on cross-validation performance
4. Saves it as `artifacts/best_model.pkl`
5. The API automatically loads this best model

**Current Best Model**: Ridge Regression with alpha=1.0, solver='auto'

## 🎯 **API Endpoints**

### **Health Check**
```bash
GET /health
```

### **Get Available Options**
```bash
GET /options
```

### **Make Prediction**
```bash
POST /predict
```

**Example Request:**
```json
{
  "gender": "female",
  "ethnicity": "group A",
  "parental_level_of_education": "bachelor's degree",
  "lunch": "standard",
  "test_preparation_course": "completed",
  "reading_score": 75.0,
  "writing_score": 66.0
}
```

**Example Response:**
```json
{
  "prediction": 57.67,
  "input_data": { ... },
  "status": "success",
  "message": "Prediction completed successfully"
}
```

## 📈 **Model Training and Evaluation**

### **Training Process**
1. **Data Preparation**: 1000 samples, 14 features (7 original + 7 encoded)
2. **Baseline Training**: 6 models with default parameters
3. **Hyperparameter Optimization**: GridSearchCV with 5-fold cross-validation
4. **Performance Comparison**: Comprehensive metrics analysis
5. **Model Selection**: Best model automatically identified

### **Evaluation Metrics**
- **R² Score**: Coefficient of determination (higher is better)
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **Overfitting Score**: Training R² - Test R² (closer to 0 is better)

## 🏆 **Best Practices Demonstrated**

1. **Always Use Cross-Validation**: Prevents overfitting and provides reliable performance estimates
2. **Hyperparameter Tuning**: Can significantly improve model performance
3. **Performance Monitoring**: Track multiple metrics, not just accuracy
4. **Overfitting Detection**: Monitor training vs test performance gaps
5. **Model Comparison**: Maintain baseline models for performance comparison

## 📁 **Available Models**

### **For Production Use**
- **Best Model**: `artifacts/best_model.pkl` (Ridge Regression)
- **Best Preprocessor**: `artifacts/optimized_preprocessors/preprocessor.pkl`

### **For Research/Comparison**
- **Baseline Models**: `artifacts/baseline_models/`
- **Optimized Models**: `artifacts/optimized_models/`
- **Detailed Metrics**: `artifacts/metrics/`

## 🔄 **Retraining Models**

To retrain all models with fresh data:

```bash
python retrain_model.py
```

This will:
- Remove old models
- Train new baseline and optimized models
- Generate fresh performance metrics
- Update all artifacts

## 📊 **Performance Visualization**

The training script generates comprehensive JSON files with all metrics:

- **`baseline_metrics.json`**: Complete baseline model performance
- **`optimized_metrics.json`**: Complete optimized model performance  
- **`model_comparison.json`**: Side-by-side performance comparison
- **`training_summary.json`**: Training summary and recommendations

## 🎓 **Learning Outcomes**

This project demonstrates:

1. **The Value of Hyperparameter Tuning**: Up to 12.68% performance improvement
2. **Overfitting Prevention**: Cross-validation and regularization techniques
3. **Model Selection**: How to choose the best model for production
4. **Performance Monitoring**: Comprehensive evaluation metrics
5. **Professional ML Workflow**: Organized model management and documentation

## 🤝 **Contributing**

Feel free to contribute by:
- Adding new models to the training pipeline
- Improving hyperparameter search spaces
- Enhancing the evaluation metrics
- Optimizing the API performance

## 📄 **License**

This project is open source and available under the MIT License.