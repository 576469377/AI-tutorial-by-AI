#!/usr/bin/env python3
"""
Advanced AI Demos and Interactive Examples
==========================================

This module provides interactive demonstrations of various AI capabilities
including text generation, image classification, and model explanations.

Features:
- Interactive text generation with multiple models
- Real-time model performance comparison
- Model explanation and interpretability tools
- Web-based demonstration interface

Author: AI Tutorial by AI
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class InteractiveAIDemo:
    """Interactive demonstrations of AI capabilities"""
    
    def __init__(self):
        """Initialize the interactive demo system"""
        self.demo_results = []
        self.performance_history = {}
        
        # Create output directory
        os.makedirs("demo_outputs", exist_ok=True)
    
    def text_generation_demo(self, prompts: List[str] = None) -> Dict[str, Any]:
        """
        Demonstrate text generation with different approaches
        
        Args:
            prompts: List of text prompts to use
            
        Returns:
            Dictionary with generation results
        """
        if prompts is None:
            prompts = [
                "The future of artificial intelligence",
                "Machine learning applications in healthcare",
                "Climate change and technology solutions",
                "The importance of data science",
                "Quantum computing breakthrough"
            ]
        
        print("🤖 Interactive Text Generation Demo")
        print("=" * 50)
        
        # Simulate different text generation models
        models = {
            'Simple Markov': self._markov_simulation,
            'Neural Language Model': self._neural_lm_simulation,
            'Transformer': self._transformer_simulation,
            'Large Language Model': self._llm_simulation
        }
        
        results = {}
        
        for model_name, generator in models.items():
            print(f"\n🔮 {model_name} Generation:")
            print("-" * 30)
            
            model_results = []
            for prompt in prompts:
                generated = generator(prompt)
                model_results.append({
                    'prompt': prompt,
                    'generated': generated,
                    'length': len(generated.split()),
                    'quality_score': self._assess_text_quality(generated)
                })
                print(f"Prompt: '{prompt[:30]}...'")
                print(f"Generated: '{generated}'")
                print(f"Quality Score: {model_results[-1]['quality_score']:.2f}")
                print()
            
            results[model_name] = model_results
        
        # Create comparison visualization
        self._create_text_generation_comparison(results)
        
        return results
    
    def model_performance_demo(self) -> go.Figure:
        """
        Demonstrate real-time model performance comparison
        
        Returns:
            Plotly figure with performance comparison
        """
        print("📊 Real-Time Model Performance Demo")
        print("=" * 50)
        
        # Simulate training progress for different models
        models = ['CNN', 'ResNet', 'Transformer', 'BERT', 'Custom Model']
        epochs = 50
        
        performance_data = []
        
        for model in models:
            # Simulate different learning curves
            if model == 'CNN':
                base_acc = 0.6
                learning_rate = 0.02
                noise_level = 0.03
            elif model == 'ResNet':
                base_acc = 0.7
                learning_rate = 0.025
                noise_level = 0.02
            elif model == 'Transformer':
                base_acc = 0.75
                learning_rate = 0.015
                noise_level = 0.025
            elif model == 'BERT':
                base_acc = 0.8
                learning_rate = 0.01
                noise_level = 0.015
            else:  # Custom Model
                base_acc = 0.65
                learning_rate = 0.03
                noise_level = 0.02
            
            for epoch in range(epochs):
                # Simulate learning curve with realistic patterns
                progress = epoch / epochs
                
                # Logarithmic improvement with plateau
                improvement = learning_rate * np.log(1 + epoch * 2)
                accuracy = base_acc + improvement * (1 - progress * 0.3)
                
                # Add realistic noise
                accuracy += np.random.normal(0, noise_level)
                
                # Ensure bounds
                accuracy = np.clip(accuracy, 0, 1)
                
                # Simulate loss (inverse relationship with some offset)
                loss = (1 - accuracy) * 2 + np.random.normal(0, 0.1)
                loss = np.clip(loss, 0.1, 2.0)
                
                performance_data.append({
                    'model': model,
                    'epoch': epoch + 1,
                    'accuracy': accuracy,
                    'loss': loss,
                    'learning_rate': learning_rate * (0.95 ** (epoch // 10))  # LR decay
                })
        
        df = pd.DataFrame(performance_data)
        
        # Create interactive performance dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Accuracy', 'Training Loss', 
                          'Learning Rate Schedule', 'Final Performance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Colors for models
        colors = px.colors.qualitative.Set1[:len(models)]
        
        # 1. Training Accuracy
        for i, model in enumerate(models):
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['epoch'],
                    y=model_data['accuracy'],
                    mode='lines',
                    name=model,
                    line=dict(color=colors[i]),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 2. Training Loss
        for i, model in enumerate(models):
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['epoch'],
                    y=model_data['loss'],
                    mode='lines',
                    name=model,
                    line=dict(color=colors[i]),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Learning Rate Schedule
        for i, model in enumerate(models):
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['epoch'],
                    y=model_data['learning_rate'],
                    mode='lines',
                    name=model,
                    line=dict(color=colors[i]),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Final Performance (Bar Chart)
        final_performance = df.groupby('model')['accuracy'].last().reset_index()
        fig.add_trace(
            go.Bar(
                x=final_performance['model'],
                y=final_performance['accuracy'],
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="🚀 Real-Time Model Performance Dashboard",
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Learning Rate", row=2, col=1)
        
        fig.update_xaxes(title_text="Model", row=2, col=2)
        fig.update_yaxes(title_text="Final Accuracy", row=2, col=2)
        
        # Save the figure
        fig.write_html("demo_outputs/realtime_performance_demo.html")
        print("📊 Real-time performance demo saved as 'demo_outputs/realtime_performance_demo.html'")
        
        return fig
    
    def model_interpretability_demo(self) -> Dict[str, Any]:
        """
        Demonstrate model interpretability and explanation techniques
        
        Returns:
            Dictionary with interpretability results
        """
        print("🔍 Model Interpretability Demo")
        print("=" * 50)
        
        # Simulate feature importance analysis
        features = [
            'Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5',
            'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9', 'Feature_10'
        ]
        
        # Different models with different feature importance patterns
        models_importance = {
            'Random Forest': np.random.exponential(0.1, len(features)),
            'Gradient Boosting': np.random.exponential(0.08, len(features)),
            'Neural Network': np.random.uniform(0.05, 0.15, len(features)),
            'Linear Model': np.random.gamma(2, 0.05, len(features))
        }
        
        # Normalize importances
        for model in models_importance:
            total = sum(models_importance[model])
            models_importance[model] = [imp/total for imp in models_importance[model]]
        
        # Create feature importance comparison
        fig_importance = go.Figure()
        
        x_pos = np.arange(len(features))
        bar_width = 0.2
        
        for i, (model, importances) in enumerate(models_importance.items()):
            fig_importance.add_trace(go.Bar(
                x=[f + i * bar_width for f in x_pos],
                y=importances,
                name=model,
                width=bar_width
            ))
        
        fig_importance.update_layout(
            title="🎯 Feature Importance Comparison Across Models",
            xaxis_title="Features",
            yaxis_title="Importance Score",
            xaxis=dict(
                tickmode='array',
                tickvals=[f + bar_width*1.5 for f in x_pos],
                ticktext=features
            ),
            height=600
        )
        
        fig_importance.write_html("demo_outputs/feature_importance_demo.html")
        
        # Simulate SHAP-like values
        shap_values = np.random.normal(0, 0.1, (100, len(features)))
        
        # Create SHAP summary plot simulation
        fig_shap = go.Figure()
        
        for i, feature in enumerate(features):
            fig_shap.add_trace(go.Box(
                y=shap_values[:, i],
                name=feature,
                boxpoints='outliers'
            ))
        
        fig_shap.update_layout(
            title="📊 SHAP Values Distribution (Simulated)",
            xaxis_title="Features",
            yaxis_title="SHAP Value",
            height=600
        )
        
        fig_shap.write_html("demo_outputs/shap_demo.html")
        
        # Decision boundary visualization (2D example)
        self._create_decision_boundary_demo()
        
        results = {
            'feature_importance': models_importance,
            'shap_simulation': shap_values.tolist(),
            'interpretability_methods': [
                'Feature Importance',
                'SHAP Values',
                'LIME',
                'Permutation Importance',
                'Decision Boundaries'
            ]
        }
        
        print("🔍 Interpretability demos created:")
        print("  - Feature importance comparison")
        print("  - SHAP values simulation")
        print("  - Decision boundary visualization")
        
        return results
    
    def create_comprehensive_demo_dashboard(self) -> go.Figure:
        """
        Create a comprehensive demo dashboard showcasing all capabilities
        
        Returns:
            Plotly figure with comprehensive dashboard
        """
        print("🌟 Creating Comprehensive Demo Dashboard")
        print("=" * 50)
        
        # Create multi-tab dashboard with different AI capabilities
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Model Performance Evolution',
                'Feature Importance Analysis',
                'Text Generation Quality',
                'Training Efficiency',
                'Model Complexity vs Accuracy',
                'AI Capability Matrix'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Sample data for demonstrations
        epochs = list(range(1, 51))
        
        # 1. Model Performance Evolution
        models = ['CNN', 'ResNet', 'Transformer']
        for i, model in enumerate(models):
            accuracy = [0.6 + 0.3 * (1 - np.exp(-epoch/10)) + np.random.normal(0, 0.02) for epoch in epochs]
            accuracy = [min(0.95, max(0.5, acc)) for acc in accuracy]
            
            fig.add_trace(
                go.Scatter(x=epochs, y=accuracy, mode='lines', name=model),
                row=1, col=1
            )
        
        # 2. Feature Importance Analysis
        features = ['F1', 'F2', 'F3', 'F4', 'F5']
        importance = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        fig.add_trace(
            go.Bar(x=features, y=importance, showlegend=False),
            row=1, col=2
        )
        
        # 3. Text Generation Quality
        models_text = ['Markov', 'LSTM', 'Transformer', 'LLM']
        quality_scores = [0.4, 0.65, 0.8, 0.92]
        
        fig.add_trace(
            go.Bar(x=models_text, y=quality_scores, showlegend=False,
                  marker_color=['red', 'orange', 'lightblue', 'green']),
            row=2, col=1
        )
        
        # 4. Training Efficiency (Accuracy vs Time)
        training_times = [120, 300, 600, 1200, 2400]
        accuracies = [0.75, 0.82, 0.88, 0.91, 0.93]
        model_names = ['Simple CNN', 'ResNet-18', 'ResNet-50', 'BERT-Base', 'BERT-Large']
        
        fig.add_trace(
            go.Scatter(
                x=training_times, y=accuracies, mode='markers+text',
                text=model_names, textposition="top center",
                marker=dict(size=10, color=accuracies, colorscale='Viridis'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 5. Model Complexity vs Accuracy
        complexities = [50000, 11000000, 25000000, 110000000, 175000000]
        
        fig.add_trace(
            go.Scatter(
                x=complexities, y=accuracies, mode='markers',
                marker=dict(size=12, color='red'),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # 6. AI Capability Matrix
        capabilities = ['Text', 'Vision', 'Speech', 'Reasoning']
        model_types = ['Traditional ML', 'Deep Learning', 'Transformers']
        capability_matrix = [
            [0.6, 0.8, 0.95],  # Text
            [0.4, 0.9, 0.7],   # Vision
            [0.3, 0.85, 0.8],  # Speech
            [0.5, 0.7, 0.9]    # Reasoning
        ]
        
        fig.add_trace(
            go.Heatmap(
                z=capability_matrix,
                x=model_types,
                y=capabilities,
                colorscale='RdYlGn',
                showscale=False
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="🌟 Comprehensive AI Capabilities Dashboard",
            height=1200,
            showlegend=True
        )
        
        # Update individual subplot labels
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        
        fig.update_xaxes(title_text="Features", row=1, col=2)
        fig.update_yaxes(title_text="Importance", row=1, col=2)
        
        fig.update_xaxes(title_text="Model Type", row=2, col=1)
        fig.update_yaxes(title_text="Quality Score", row=2, col=1)
        
        fig.update_xaxes(title_text="Training Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Accuracy", row=2, col=2)
        
        fig.update_xaxes(title_text="Model Parameters", row=3, col=1)
        fig.update_yaxes(title_text="Accuracy", row=3, col=1)
        
        # Save dashboard
        fig.write_html("demo_outputs/comprehensive_ai_dashboard.html")
        print("🌟 Comprehensive dashboard saved as 'demo_outputs/comprehensive_ai_dashboard.html'")
        
        return fig
    
    def _markov_simulation(self, prompt: str) -> str:
        """Simulate Markov chain text generation"""
        words = prompt.split()
        if len(words) == 0:
            return "Error: Empty prompt"
        
        # Simple simulation
        generated = words[-1] if words else "the"
        suffixes = ["is interesting", "will be revolutionary", "shows promise", 
                   "needs more research", "has great potential"]
        return f"{generated} {np.random.choice(suffixes)}"
    
    def _neural_lm_simulation(self, prompt: str) -> str:
        """Simulate neural language model generation"""
        responses = {
            "future": "will likely involve advanced neural networks and quantum computing systems",
            "machine": "learning algorithms continue to evolve with better accuracy and efficiency",
            "artificial": "intelligence systems are becoming more sophisticated and capable",
            "climate": "change requires innovative technological solutions and global cooperation",
            "data": "science enables organizations to make informed decisions and predictions",
            "quantum": "computing represents a paradigm shift in computational capabilities"
        }
        
        for key, response in responses.items():
            if key in prompt.lower():
                return response
        
        return "presents fascinating opportunities for technological advancement"
    
    def _transformer_simulation(self, prompt: str) -> str:
        """Simulate transformer-based generation"""
        sophisticated_responses = {
            "future": "of artificial intelligence lies in the development of more efficient, interpretable, and ethically-aligned systems that can collaborate effectively with humans",
            "machine": "learning applications in healthcare are revolutionizing diagnostic accuracy, drug discovery, and personalized treatment plans",
            "artificial": "intelligence systems must be designed with careful consideration of bias, fairness, and transparency to ensure beneficial outcomes for society",
            "climate": "change mitigation strategies increasingly rely on AI-powered optimization of energy systems, carbon capture technologies, and sustainable resource management",
            "data": "science methodologies continue to evolve, incorporating advanced statistical techniques and machine learning to extract meaningful insights from complex datasets",
            "quantum": "computing breakthrough could fundamentally transform cryptography, optimization problems, and simulation of quantum mechanical systems"
        }
        
        for key, response in sophisticated_responses.items():
            if key in prompt.lower():
                return response
        
        return "represents a complex domain requiring interdisciplinary collaboration and careful consideration of technological, ethical, and societal implications"
    
    def _llm_simulation(self, prompt: str) -> str:
        """Simulate large language model generation"""
        llm_responses = {
            "future": "of artificial intelligence is characterized by several key trends: the development of more efficient architectures that require less computational power, improved alignment techniques that ensure AI systems behave in accordance with human values, and the emergence of multimodal systems that can seamlessly integrate text, vision, and audio processing capabilities",
            "machine": "learning applications in healthcare represent one of the most promising and impactful areas of AI development, with recent advances in medical imaging analysis achieving superhuman performance in detecting certain cancers, while natural language processing systems are revolutionizing clinical documentation and enabling more personalized patient care through intelligent analysis of electronic health records",
            "artificial": "intelligence systems today face critical challenges around interpretability, robustness, and ethical deployment, requiring careful consideration of algorithmic bias, data privacy, and the potential societal impacts of automation, while simultaneously offering unprecedented opportunities to address complex global challenges in areas such as climate modeling, drug discovery, and educational personalization",
            "climate": "change and technology solutions intersect in fascinating ways, with AI-powered systems optimizing renewable energy grids, machine learning algorithms improving weather prediction models, and advanced materials science accelerated by computational methods leading to more efficient solar panels and energy storage systems",
            "data": "science has evolved from a primarily statistical discipline to a complex field encompassing machine learning, deep learning, and advanced visualization techniques, with modern practitioners requiring skills in programming, domain expertise, and ethical reasoning to navigate the challenges of working with large-scale, often biased datasets",
            "quantum": "computing breakthrough would represent a fundamental shift in computational paradigms, potentially enabling exponential speedups for certain classes of problems including cryptographic analysis, optimization challenges, and quantum system simulation, while requiring the development of entirely new programming models and error correction techniques"
        }
        
        for key, response in llm_responses.items():
            if key in prompt.lower():
                return response
        
        return "encompasses a rich landscape of interdisciplinary research and development, where advances in computational methods, mathematical foundations, and engineering implementations converge to create systems with increasingly sophisticated capabilities for reasoning, pattern recognition, and creative problem-solving across diverse domains"
    
    def _assess_text_quality(self, text: str) -> float:
        """Assess text quality based on length, complexity, and coherence"""
        # Simple quality metrics
        words = text.split()
        
        # Length factor (optimal around 20-30 words)
        length_score = min(1.0, len(words) / 25)
        
        # Complexity factor (average word length)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        complexity_score = min(1.0, avg_word_length / 7)
        
        # Vocabulary diversity
        unique_words = len(set(words))
        diversity_score = unique_words / len(words) if words else 0
        
        # Combined score
        quality = (length_score * 0.4 + complexity_score * 0.3 + diversity_score * 0.3)
        return min(1.0, quality)
    
    def _create_text_generation_comparison(self, results: Dict[str, Any]):
        """Create visualization comparing text generation models"""
        # Extract quality scores
        model_scores = {}
        for model, model_results in results.items():
            scores = [result['quality_score'] for result in model_results]
            model_scores[model] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        
        # Create comparison plot
        fig = go.Figure()
        
        models = list(model_scores.keys())
        means = [model_scores[model]['mean'] for model in models]
        stds = [model_scores[model]['std'] for model in models]
        
        fig.add_trace(go.Bar(
            x=models,
            y=means,
            error_y=dict(type='data', array=stds),
            marker_color=['red', 'orange', 'lightblue', 'green']
        ))
        
        fig.update_layout(
            title="🤖 Text Generation Quality Comparison",
            xaxis_title="Model Type",
            yaxis_title="Average Quality Score",
            height=500
        )
        
        fig.write_html("demo_outputs/text_generation_comparison.html")
        print("📝 Text generation comparison saved as 'demo_outputs/text_generation_comparison.html'")
    
    def _create_decision_boundary_demo(self):
        """Create decision boundary visualization demo"""
        # Generate synthetic 2D data
        np.random.seed(42)
        n_samples = 200
        
        # Create two classes with some overlap
        class_0_x = np.random.normal(2, 1, n_samples//2)
        class_0_y = np.random.normal(2, 1, n_samples//2)
        
        class_1_x = np.random.normal(4, 1, n_samples//2)
        class_1_y = np.random.normal(4, 1, n_samples//2)
        
        # Create decision boundary (simplified)
        x_range = np.linspace(0, 6, 100)
        y_range = np.linspace(0, 6, 100)
        
        fig = go.Figure()
        
        # Add data points
        fig.add_trace(go.Scatter(
            x=class_0_x, y=class_0_y,
            mode='markers',
            name='Class 0',
            marker=dict(color='red', size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=class_1_x, y=class_1_y,
            mode='markers',
            name='Class 1',
            marker=dict(color='blue', size=8)
        ))
        
        # Add decision boundary (simple line)
        fig.add_trace(go.Scatter(
            x=x_range, y=x_range,
            mode='lines',
            name='Decision Boundary',
            line=dict(color='black', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="🎯 Decision Boundary Visualization",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            height=500
        )
        
        fig.write_html("demo_outputs/decision_boundary_demo.html")


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Main demonstration function"""
    print("🌟 Advanced AI Demos and Interactive Examples")
    print("=" * 70)
    
    # Initialize demo system
    demo = InteractiveAIDemo()
    
    # Run text generation demo
    print("\n1️⃣ Text Generation Demo")
    text_results = demo.text_generation_demo()
    
    # Run performance demo
    print("\n2️⃣ Model Performance Demo")
    performance_fig = demo.model_performance_demo()
    
    # Run interpretability demo
    print("\n3️⃣ Model Interpretability Demo")
    interpretability_results = demo.model_interpretability_demo()
    
    # Create comprehensive dashboard
    print("\n4️⃣ Comprehensive Dashboard")
    comprehensive_fig = demo.create_comprehensive_demo_dashboard()
    
    # Summary
    print("\n✅ All demos completed successfully!")
    print("\n📁 Generated Interactive Demos:")
    print("  - demo_outputs/realtime_performance_demo.html")
    print("  - demo_outputs/feature_importance_demo.html")
    print("  - demo_outputs/shap_demo.html")
    print("  - demo_outputs/text_generation_comparison.html")
    print("  - demo_outputs/decision_boundary_demo.html")
    print("  - demo_outputs/comprehensive_ai_dashboard.html")
    
    print("\n🚀 Next Steps:")
    print("  1. Open the HTML files in your browser")
    print("  2. Explore the interactive features")
    print("  3. Use these demos as templates for your own AI projects")
    print("  4. Integrate with real models for live demonstrations")

if __name__ == "__main__":
    main()