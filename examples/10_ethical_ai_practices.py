#!/usr/bin/env python3
"""
Ethical AI and Responsible Machine Learning Examples
==================================================

This module demonstrates ethical AI practices, bias detection and mitigation,
fairness metrics, and responsible AI development workflows.

Key concepts covered:
- Data bias detection and analysis
- Fairness metrics and evaluation
- Model bias mitigation techniques
- Explainable AI for transparency
- Privacy-preserving machine learning basics
- Robust model validation

Author: AI Tutorial by AI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def create_biased_dataset():
    """
    Create a synthetic dataset that demonstrates potential bias issues
    
    Returns:
        pd.DataFrame: Dataset with potential bias
    """
    print("🔍 Creating synthetic dataset with potential bias...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create features
    age = np.random.normal(35, 10, n_samples)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, 
                                p=[0.4, 0.35, 0.2, 0.05])
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
    race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples,
                           p=[0.6, 0.15, 0.15, 0.1])
    
    # Create approval decision with bias
    approval = []
    for i in range(n_samples):
        prob = 0.5
        
        # Add bias factors based on demographics
        if gender[i] == 'Male':
            prob += 0.1
        if race[i] == 'White':
            prob += 0.15
        elif race[i] == 'Asian':
            prob += 0.05
        if education[i] in ['Master', 'PhD']:
            prob += 0.2
        elif education[i] == 'Bachelor':
            prob += 0.1
        if age[i] > 30:
            prob += 0.1
        
        # Add some noise for realistic data
        prob += np.random.normal(0, 0.1)
        prob = np.clip(prob, 0, 1)
        
        approval.append(1 if np.random.random() < prob else 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'education': education,
        'gender': gender,
        'race': race,
        'approved': approval
    })
    
    print(f"✅ Created dataset with {len(df)} samples")
    print(f"   Overall approval rate: {df['approved'].mean():.3f}")
    
    return df

def analyze_bias_in_data(df):
    """
    Analyze potential bias in the dataset
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    print("\n🔍 BIAS ANALYSIS")
    print("=" * 50)
    
    # Overall statistics
    overall_approval = df['approved'].mean()
    print(f"📊 Overall approval rate: {overall_approval:.3f}")
    
    # Analyze by protected attributes
    protected_attrs = ['gender', 'race', 'education']
    
    bias_analysis = {}
    
    for attr in protected_attrs:
        print(f"\n📈 Analysis by {attr}:")
        group_stats = df.groupby(attr)['approved'].agg(['count', 'mean', 'std']).round(3)
        print(group_stats)
        
        # Calculate disparate impact
        group_rates = df.groupby(attr)['approved'].mean()
        max_rate = group_rates.max()
        min_rate = group_rates.min()
        disparate_impact = min_rate / max_rate if max_rate > 0 else 0
        
        bias_analysis[attr] = {
            'group_rates': group_rates.to_dict(),
            'disparate_impact': disparate_impact,
            'max_rate': max_rate,
            'min_rate': min_rate
        }
        
        print(f"   Disparate Impact Ratio: {disparate_impact:.3f}")
        if disparate_impact < 0.8:
            print("   ⚠️  Potential bias detected (ratio < 0.8)")
        else:
            print("   ✅ No significant bias detected")
    
    return bias_analysis

def visualize_bias(df):
    """
    Create visualizations to show bias in the data
    
    Args:
        df (pd.DataFrame): Dataset to visualize
    """
    print("\n📊 Creating bias visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Bias Analysis Visualization', fontsize=16, fontweight='bold')
    
    # 1. Approval rates by gender
    gender_rates = df.groupby('gender')['approved'].mean()
    axes[0, 0].bar(gender_rates.index, gender_rates.values, color=['lightblue', 'lightpink'])
    axes[0, 0].set_title('Approval Rates by Gender')
    axes[0, 0].set_ylabel('Approval Rate')
    axes[0, 0].set_ylim(0, 1)
    
    # Add value labels
    for i, v in enumerate(gender_rates.values):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 2. Approval rates by race
    race_rates = df.groupby('race')['approved'].mean()
    axes[0, 1].bar(race_rates.index, race_rates.values, color=['lightcoral', 'lightgreen', 'lightyellow', 'lightgray'])
    axes[0, 1].set_title('Approval Rates by Race')
    axes[0, 1].set_ylabel('Approval Rate')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(race_rates.values):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 3. Approval rates by education
    edu_rates = df.groupby('education')['approved'].mean()
    axes[1, 0].bar(edu_rates.index, edu_rates.values, color=['orange', 'yellow', 'green', 'blue'])
    axes[1, 0].set_title('Approval Rates by Education')
    axes[1, 0].set_ylabel('Approval Rate')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(edu_rates.values):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 4. Correlation heatmap
    df_encoded = df.copy()
    le_gender = LabelEncoder()
    le_race = LabelEncoder()
    le_education = LabelEncoder()
    
    df_encoded['gender_encoded'] = le_gender.fit_transform(df['gender'])
    df_encoded['race_encoded'] = le_race.fit_transform(df['race'])
    df_encoded['education_encoded'] = le_education.fit_transform(df['education'])
    
    correlation_matrix = df_encoded[['age', 'gender_encoded', 'race_encoded', 'education_encoded', 'approved']].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
    axes[1, 1].set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('ethical_ai_bias_analysis.png', dpi=300, bbox_inches='tight')
    print("💾 Bias analysis visualization saved as 'ethical_ai_bias_analysis.png'")
    
    return fig

def implement_fairness_constraints(df, sensitive_attr='gender'):
    """
    Implement fairness-aware machine learning techniques
    
    Args:
        df (pd.DataFrame): Dataset
        sensitive_attr (str): Sensitive attribute to ensure fairness for
    
    Returns:
        dict: Results from different fairness approaches
    """
    print(f"\n🎯 IMPLEMENTING FAIRNESS CONSTRAINTS FOR '{sensitive_attr}'")
    print("=" * 60)
    
    # Prepare data
    df_encoded = df.copy()
    le_gender = LabelEncoder()
    le_race = LabelEncoder()
    le_education = LabelEncoder()
    
    df_encoded['gender_encoded'] = le_gender.fit_transform(df['gender'])
    df_encoded['race_encoded'] = le_race.fit_transform(df['race'])
    df_encoded['education_encoded'] = le_education.fit_transform(df['education'])
    
    # Features (excluding the sensitive attribute from model features for fairness)
    feature_cols = ['age', 'education_encoded']
    if sensitive_attr != 'gender':
        feature_cols.append('gender_encoded')
    if sensitive_attr != 'race':
        feature_cols.append('race_encoded')
    
    X = df_encoded[feature_cols]
    y = df_encoded['approved']
    sensitive_features = df_encoded[f'{sensitive_attr}_encoded']
    
    # Split data
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive_features, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # 1. Baseline model (potentially biased)
    print("📊 Training baseline model...")
    baseline_model = LogisticRegression(random_state=42)
    baseline_model.fit(X_train_scaled, y_train)
    baseline_pred = baseline_model.predict(X_test_scaled)
    
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    results['baseline'] = {
        'model': baseline_model,
        'predictions': baseline_pred,
        'accuracy': baseline_accuracy,
        'scaler': scaler,
        'feature_cols': feature_cols
    }
    
    # 2. Fairness through data preprocessing (simplified)
    print("⚖️  Implementing fairness through balanced sampling...")
    
    # Create balanced training set by sensitive attribute
    unique_sensitive = sensitive_train.unique()
    min_group_size = min([(sensitive_train == group).sum() for group in unique_sensitive])
    
    balanced_indices = []
    for group in unique_sensitive:
        group_indices = np.where(sensitive_train == group)[0]
        balanced_indices.extend(np.random.choice(group_indices, min_group_size, replace=False))
    
    X_train_balanced = X_train_scaled[balanced_indices]
    y_train_balanced = y_train.iloc[balanced_indices]
    
    balanced_model = LogisticRegression(random_state=42)
    balanced_model.fit(X_train_balanced, y_train_balanced)
    balanced_pred = balanced_model.predict(X_test_scaled)
    
    balanced_accuracy = accuracy_score(y_test, balanced_pred)
    results['balanced'] = {
        'model': balanced_model,
        'predictions': balanced_pred,
        'accuracy': balanced_accuracy,
        'scaler': scaler,
        'feature_cols': feature_cols
    }
    
    # 3. Fairness evaluation
    print("\n📈 Evaluating fairness metrics...")
    
    for model_name, model_data in results.items():
        print(f"\n{model_name.upper()} MODEL FAIRNESS:")
        pred = model_data['predictions']
        
        # Calculate group-specific metrics
        for group in unique_sensitive:
            group_mask = sensitive_test == group
            group_accuracy = accuracy_score(y_test[group_mask], pred[group_mask])
            group_positive_rate = pred[group_mask].mean()
            
            print(f"  Group {group}: Accuracy={group_accuracy:.3f}, Positive Rate={group_positive_rate:.3f}")
        
        # Calculate disparate impact
        group_positive_rates = []
        for group in unique_sensitive:
            group_mask = sensitive_test == group
            group_positive_rates.append(pred[group_mask].mean())
        
        disparate_impact = min(group_positive_rates) / max(group_positive_rates) if max(group_positive_rates) > 0 else 0
        print(f"  Disparate Impact: {disparate_impact:.3f}")
        
        results[model_name]['disparate_impact'] = disparate_impact
        results[model_name]['group_positive_rates'] = group_positive_rates
    
    return results

def demonstrate_explainable_ai(df, fairness_results):
    """
    Demonstrate explainable AI techniques for transparency
    
    Args:
        df (pd.DataFrame): Dataset
        fairness_results (dict): Results from fairness analysis
    """
    print("\n🔍 EXPLAINABLE AI DEMONSTRATION")
    print("=" * 50)
    
    # Use the balanced model for explanation
    balanced_model = fairness_results['balanced']['model']
    scaler = fairness_results['balanced']['scaler']
    feature_cols = fairness_results['balanced']['feature_cols']
    
    # Prepare data for explanation
    df_encoded = df.copy()
    le_gender = LabelEncoder()
    le_race = LabelEncoder()
    le_education = LabelEncoder()
    
    df_encoded['gender_encoded'] = le_gender.fit_transform(df['gender'])
    df_encoded['race_encoded'] = le_race.fit_transform(df['race'])
    df_encoded['education_encoded'] = le_education.fit_transform(df['education'])
    
    X = df_encoded[feature_cols]
    feature_names = [col.replace('_encoded', '').replace('_', ' ').title() for col in feature_cols]
    
    # Get feature importance (for logistic regression, use coefficients)
    if hasattr(balanced_model.coef_, 'flatten'):
        feature_importance = np.abs(balanced_model.coef_.flatten())
    else:
        feature_importance = np.abs(balanced_model.coef_[0])
    
    # Ensure we have the right number of features
    if len(feature_importance) != len(feature_names):
        feature_importance = feature_importance[:len(feature_names)]
    
    print("📊 Feature Importance Analysis:")
    for i, (name, importance) in enumerate(zip(feature_names, feature_importance)):
        print(f"  {name}: {importance:.4f}")
    
    # Create explanation visualization
    plt.figure(figsize=(10, 6))
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'][:len(feature_names)]
    bars = plt.bar(feature_names, feature_importance, color=colors)
    plt.title('Feature Importance in Fair Model', fontsize=14, fontweight='bold')
    plt.ylabel('Absolute Coefficient Value')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, importance in zip(bars, feature_importance):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{importance:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('explainable_ai_feature_importance.png', dpi=300, bbox_inches='tight')
    print("💾 Feature importance visualization saved as 'explainable_ai_feature_importance.png'")
    
    # Demonstrate individual prediction explanation
    print("\n🔍 Individual Prediction Explanation:")
    sample_idx = 0
    sample_features = X.iloc[sample_idx:sample_idx+1]
    sample_features_scaled = scaler.transform(sample_features)
    sample_prediction = balanced_model.predict_proba(sample_features_scaled)[0]
    
    print(f"Sample prediction probability: {sample_prediction[1]:.3f}")
    print(f"Feature contributions:")
    
    for i, (name, value, coef) in enumerate(zip(feature_names, sample_features.iloc[0], balanced_model.coef_[0])):
        contribution = value * coef
        print(f"  {name} = {value:.2f} → contribution: {contribution:.4f}")

def create_ethical_ai_checklist():
    """
    Create an ethical AI development checklist
    
    Returns:
        dict: Comprehensive ethical AI checklist
    """
    checklist = {
        "Data Collection & Preparation": [
            "✅ Ensure data represents diverse populations",
            "✅ Document data sources and collection methods",
            "✅ Check for historical biases in training data",
            "✅ Implement privacy protection measures",
            "✅ Obtain proper consent for data usage"
        ],
        "Model Development": [
            "✅ Use fairness-aware algorithms when appropriate",
            "✅ Implement bias detection and mitigation techniques",
            "✅ Ensure model transparency and explainability",
            "✅ Test for robustness and edge cases",
            "✅ Document model limitations and assumptions"
        ],
        "Evaluation & Testing": [
            "✅ Evaluate performance across different demographic groups",
            "✅ Measure and report fairness metrics",
            "✅ Test for adversarial attacks and robustness",
            "✅ Validate on diverse test sets",
            "✅ Assess real-world performance vs. lab performance"
        ],
        "Deployment & Monitoring": [
            "✅ Implement continuous monitoring for bias drift",
            "✅ Provide clear explanations for AI decisions",
            "✅ Establish human oversight and intervention mechanisms",
            "✅ Create feedback loops for model improvement",
            "✅ Regularly audit and update the system"
        ],
        "Governance & Ethics": [
            "✅ Establish clear AI ethics guidelines",
            "✅ Involve diverse stakeholders in decision-making",
            "✅ Ensure accountability and responsibility",
            "✅ Comply with relevant regulations and standards",
            "✅ Plan for long-term impact and sustainability"
        ]
    }
    
    return checklist

def main():
    """Main demonstration function"""
    print("🤖 ETHICAL AI AND RESPONSIBLE MACHINE LEARNING")
    print("=" * 60)
    print("This demonstration covers key aspects of ethical AI development")
    print("including bias detection, fairness metrics, and responsible practices.")
    print()
    
    # 1. Create and analyze biased dataset
    df = create_biased_dataset()
    bias_analysis = analyze_bias_in_data(df)
    
    # 2. Visualize bias
    visualize_bias(df)
    
    # 3. Implement fairness constraints
    fairness_results = implement_fairness_constraints(df, sensitive_attr='gender')
    
    # 4. Demonstrate explainable AI
    demonstrate_explainable_ai(df, fairness_results)
    
    # 5. Show ethical AI checklist
    print("\n📋 ETHICAL AI DEVELOPMENT CHECKLIST")
    print("=" * 50)
    
    checklist = create_ethical_ai_checklist()
    for category, items in checklist.items():
        print(f"\n🎯 {category}:")
        for item in items:
            print(f"  {item}")
    
    # 6. Summary and recommendations
    print("\n🎉 ETHICAL AI DEMONSTRATION COMPLETE")
    print("=" * 50)
    print("📊 Key Findings:")
    
    baseline_di = fairness_results['baseline']['disparate_impact']
    balanced_di = fairness_results['balanced']['disparate_impact']
    
    print(f"  • Baseline model disparate impact: {baseline_di:.3f}")
    print(f"  • Balanced model disparate impact: {balanced_di:.3f}")
    print(f"  • Improvement in fairness: {((balanced_di - baseline_di) / baseline_di * 100):.1f}%")
    
    print(f"\n💡 Recommendations:")
    print(f"  • Always evaluate models for bias across protected groups")
    print(f"  • Implement fairness constraints during model development")
    print(f"  • Use explainable AI techniques for transparency")
    print(f"  • Establish continuous monitoring for bias drift")
    print(f"  • Follow ethical AI development practices")
    
    print(f"\n📁 Generated files:")
    print(f"  • ethical_ai_bias_analysis.png - Bias analysis visualization")
    print(f"  • explainable_ai_feature_importance.png - Feature importance plot")

if __name__ == "__main__":
    main()