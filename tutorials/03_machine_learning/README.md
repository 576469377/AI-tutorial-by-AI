# Machine Learning Basics

Master traditional machine learning algorithms and techniques that form the foundation of data science and AI applications.

## 🎯 Learning Track

**📊 You are in: Machine Learning Track**  
**⏱️ Estimated time**: 3-4 weeks  
**🎯 Best for**: Data analysis, business intelligence, interpretable AI solutions

## What You'll Learn

- Traditional machine learning concepts and terminology
- Supervised vs unsupervised learning approaches
- Classification and regression algorithms
- Model evaluation and validation techniques
- Feature engineering and data preprocessing
- Practical applications in business and research

## Prerequisites

- ✅ Completed [AI Fundamentals](../00_ai_fundamentals/README.md)
- ✅ Completed [Python Basics](../01_basics/README.md) 
- ✅ Completed [Data Visualization](../02_data_visualization/README.md)
- 📊 Understanding of statistics and probability (helpful but not required)

## 🎓 Learning Objectives

By the end of this tutorial, you will:
- Understand when to use traditional ML vs deep learning approaches
- Implement core ML algorithms from scratch and with scikit-learn
- Evaluate model performance using appropriate metrics
- Apply feature engineering techniques to improve model performance
- Choose the right algorithm for different problem types

## 🔍 Traditional ML vs Deep Learning

**When to use Traditional Machine Learning:**
- ✅ Small to medium datasets (< 100K samples)
- ✅ Need interpretable/explainable models
- ✅ Limited computational resources
- ✅ Structured/tabular data
- ✅ Quick prototyping and fast training

**When to consider Deep Learning:**
- 🧠 Large datasets (> 100K samples)
- 🧠 Complex patterns (images, text, audio)
- 🧠 Unstructured data
- 🧠 Computational resources available
- 🧠 State-of-the-art performance needed

*This tutorial focuses on traditional ML - for deep learning, see [Neural Networks](../04_neural_networks/README.md)*

## Topics Covered

### 1. Introduction to Machine Learning
- What is machine learning?
- Types of machine learning (supervised, unsupervised, reinforcement)
- Common applications and use cases

### 2. Supervised Learning
- **Classification**: Predicting categories (e.g., spam/not spam)
- **Regression**: Predicting continuous values (e.g., house prices)
- Popular algorithms: Linear Regression, Decision Trees, Random Forest

### 3. Model Evaluation
- Training vs testing data
- Cross-validation
- Metrics: accuracy, precision, recall, F1-score, R²

### 4. Feature Engineering
- Data preprocessing
- Handling missing values
- Feature scaling and normalization
- Feature selection

## Algorithms Covered

| Algorithm | Type | Use Case | Pros | Cons |
|-----------|------|----------|------|------|
| Linear Regression | Regression | Simple relationships | Interpretable, fast | Limited to linear relationships |
| Decision Tree | Both | Clear rules | Easy to understand | Can overfit |
| Random Forest | Both | General purpose | Robust, handles overfitting | Less interpretable |
| K-Means | Clustering | Customer segmentation | Simple, efficient | Requires choosing K |

## Examples

Explore machine learning in action:
- **Notebook**: [03_machine_learning.ipynb](../../notebooks/03_machine_learning.ipynb)
- **Script**: [03_ml_examples.py](../../examples/03_ml_examples.py)

## Sample Projects

1. **House Price Prediction** - Regression example
2. **Email Spam Classification** - Binary classification
3. **Customer Segmentation** - Clustering analysis
4. **Sales Forecasting** - Time series prediction

## 🚀 Next Steps: Choose Your Path

### Option A: Continue with Advanced AI 🚀
**Recommended if you want to**: Build state-of-the-art AI systems
- **→ [Large Language Models](../06_large_language_models/README.md)** - Jump to LLM training
- **Prerequisites**: Strong Python skills, willingness to learn complex architectures

### Option B: Deep Learning Track 🧠  
**Recommended if you want to**: Understand neural networks and modern AI
- **→ [Neural Networks](../04_neural_networks/README.md)** - Start with neural network fundamentals
- **→ [PyTorch](../05_pytorch/README.md)** - Learn deep learning frameworks
- **Prerequisites**: Linear algebra knowledge helpful

### Option C: Deepen ML Knowledge 📊
**Recommended if you want to**: Become an ML expert
- Practice with real datasets from Kaggle
- Explore advanced ML topics: ensemble methods, feature selection
- Study specific domains: time series, recommendation systems

### 📚 Additional Resources
- **Practical Projects**: Try the sample projects in this tutorial
- **Competitions**: Participate in Kaggle competitions
- **Real Datasets**: Apply these techniques to your own data
- **Learning Path Guide**: See [complete learning paths](../../docs/learning_paths.md)