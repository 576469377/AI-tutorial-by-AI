# Getting Started Guide

Welcome to the AI Tutorial by AI! This guide will help you set up your environment and start learning.

## Quick Setup

### 1. Clone the Repository
```bash
git clone https://github.com/576469377/AI-tutorial-by-AI.git
cd AI-tutorial-by-AI
```

### 2. Set Up Python Environment
We recommend using a virtual environment to avoid conflicts:

```bash
# Create virtual environment
python -m venv ai_tutorial_env

# Activate it
# On Windows:
ai_tutorial_env\Scripts\activate
# On macOS/Linux:
source ai_tutorial_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Sample Data
```bash
python create_sample_data.py
```

### 5. Start Learning!
Choose your preferred learning method:

#### Option A: Interactive Jupyter Notebooks
```bash
jupyter lab
# Then open notebooks/ai_tutorial_complete.ipynb
```

#### Option B: Run Python Scripts
```bash
# Start with basics
python examples/01_numpy_pandas_basics.py

# Move to visualization
python examples/02_visualization_examples.py

# Try machine learning
python examples/03_ml_examples.py

# Explore neural networks
python examples/04_neural_network_examples.py
```

## Learning Path

### For Beginners
1. **Start Here**: Read [tutorials/01_basics/README.md](tutorials/01_basics/README.md)
2. **Run**: `python examples/01_numpy_pandas_basics.py`
3. **Next**: Work through the Jupyter notebook step by step

### For Intermediate Learners
1. Jump to [tutorials/03_machine_learning/README.md](tutorials/03_machine_learning/README.md)
2. Run the machine learning examples
3. Experiment with different algorithms

### For Advanced Learners
1. Study the neural network implementation
2. Modify the code to try different architectures
3. Apply the concepts to your own datasets

## Troubleshooting

### Common Issues

**ImportError: No module named 'numpy'**
- Make sure you've activated your virtual environment
- Run `pip install -r requirements.txt`

**FileNotFoundError: sample_data/...**
- Run `python create_sample_data.py` to generate sample datasets

**Plots not showing in Jupyter**
- Try running `%matplotlib inline` in a cell

**Permission errors**
- Make sure you have write permissions in the directory
- On Windows, try running as administrator

### Getting Help

- Check the [main README.md](README.md) for project overview
- Look at individual tutorial README files for specific topics
- Open an issue on GitHub if you encounter bugs

## Next Steps

After completing the tutorials:

1. **Practice**: Try the exercises in each tutorial
2. **Experiment**: Modify the code and see what happens
3. **Build**: Create your own AI project using the concepts learned
4. **Share**: Contribute improvements back to the project

## Additional Resources

- **Official Documentation**: Links to NumPy, Pandas, Scikit-learn docs
- **Online Courses**: Recommendations for further learning
- **Books**: Suggested reading for deep dives
- **Communities**: Where to get help and discuss AI topics

Happy learning! 🚀