# Cross-Platform Setup Guide for AI Tutorial

This guide provides detailed, platform-specific instructions for setting up your development environment to run the AI Tutorial examples and notebooks.

## Quick Platform Selection

- [🍎 macOS Setup](#macos-setup)
- [🪟 Windows Setup](#windows-setup)  
- [🐧 Linux Setup](#linux-setup)

---

## macOS Setup

### Prerequisites
- macOS 10.14 (Mojave) or later
- Administrative access to install software

### Step 1: Install Homebrew (Package Manager)
Homebrew makes it easy to install development tools on macOS.

```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add Homebrew to your PATH (for Apple Silicon Macs)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc

# Verify installation
brew --version
```

### Step 2: Install Python
We recommend using Python 3.9 or later.

#### Option A: Using Homebrew (Recommended)
```bash
# Install Python
brew install python

# Verify installation
python3 --version
pip3 --version
```

#### Option B: Using pyenv (For Multiple Python Versions)
```bash
# Install pyenv
brew install pyenv

# Add to shell configuration
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# Install Python 3.10
pyenv install 3.10.9
pyenv global 3.10.9

# Verify
python --version
```

### Step 3: Install Git
```bash
# Install Git
brew install git

# Configure Git (replace with your information)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify
git --version
```

### Step 4: Clone and Setup the Project
```bash
# Clone the repository
git clone https://github.com/576469377/AI-tutorial-by-AI.git
cd AI-tutorial-by-AI

# Create virtual environment
python3 -m venv ai_tutorial_env

# Activate virtual environment
source ai_tutorial_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python create_sample_data.py
```

### Step 5: Install Jupyter Lab
```bash
# Install JupyterLab (if not already installed)
pip install jupyterlab

# Start JupyterLab
jupyter lab
```

### macOS Troubleshooting

**Issue: "command not found: python"**
```bash
# Create an alias for python3
echo 'alias python=python3' >> ~/.zshrc
echo 'alias pip=pip3' >> ~/.zshrc
source ~/.zshrc
```

**Issue: Permission denied when installing packages**
```bash
# Use --user flag to install in user directory
pip install --user package_name
```

**Issue: SSL certificate errors**
```bash
# Update certificates
brew install ca-certificates
```

**Issue: PyTorch installation problems**
```bash
# For Apple Silicon Macs, use conda for better compatibility
brew install miniforge
conda install pytorch torchvision torchaudio -c pytorch
```

### macOS Performance Tips
- Enable GPU acceleration for PyTorch on Apple Silicon:
  ```python
  import torch
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  ```

---

## Windows Setup

### Prerequisites
- Windows 10 or Windows 11
- Administrator privileges

### Step 1: Install Python
#### Option A: From Microsoft Store (Easiest)
1. Open Microsoft Store
2. Search for "Python 3.10" or later
3. Click "Get" to install
4. Python will be available as `python` command

#### Option B: From Python.org (More Control)
1. Go to https://www.python.org/downloads/
2. Download Python 3.10+ for Windows
3. **Important**: Check "Add Python to PATH" during installation
4. Choose "Customize installation" and check all optional features
5. Click "Install"

### Step 2: Install Git
#### Option A: Git for Windows
1. Go to https://git-scm.com/download/win
2. Download and run the installer
3. Use default settings (recommended)
4. Open Git Bash or Command Prompt

#### Option B: Using Chocolatey (Package Manager)
```powershell
# First install Chocolatey (run as Administrator)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Install Git
choco install git

# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Step 3: Setup Development Environment
#### Using Command Prompt
```cmd
# Clone the repository
git clone https://github.com/576469377/AI-tutorial-by-AI.git
cd AI-tutorial-by-AI

# Create virtual environment
python -m venv ai_tutorial_env

# Activate virtual environment
ai_tutorial_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python create_sample_data.py
```

#### Using PowerShell
```powershell
# Clone the repository
git clone https://github.com/576469377/AI-tutorial-by-AI.git
Set-Location AI-tutorial-by-AI

# Create virtual environment
python -m venv ai_tutorial_env

# Activate virtual environment
.\ai_tutorial_env\Scripts\Activate.ps1

# If execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python create_sample_data.py
```

### Step 4: Install Additional Tools
#### Visual Studio Code (Recommended Editor)
1. Download from https://code.visualstudio.com/
2. Install Python extension
3. Install Jupyter extension

#### Windows Subsystem for Linux (WSL) - Optional
For a Linux-like experience:
```powershell
# Enable WSL (run as Administrator)
wsl --install

# Restart computer when prompted
# Then follow Linux setup instructions in WSL
```

### Windows Troubleshooting

**Issue: "python is not recognized"**
```cmd
# Check if Python is in PATH
echo %PATH%

# Manually add Python to PATH:
# Control Panel > System > Advanced > Environment Variables
# Add: C:\Users\[username]\AppData\Local\Programs\Python\Python310
# Add: C:\Users\[username]\AppData\Local\Programs\Python\Python310\Scripts
```

**Issue: SSL certificate errors**
```cmd
# Upgrade pip and certificates
python -m pip install --upgrade pip
pip install --upgrade certifi
```

**Issue: Long path names**
```cmd
# Enable long paths in Windows (run as Administrator)
# In Group Policy Editor: Computer Configuration > Administrative Templates > System > Filesystem
# Enable "Enable Win32 long paths"
```

**Issue: Permission errors**
```cmd
# Install packages for current user only
pip install --user package_name
```

**Issue: PyTorch CUDA setup**
```cmd
# For NVIDIA GPU support, install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Windows Performance Tips
- Use SSD storage for better I/O performance
- Enable GPU acceleration if you have NVIDIA GPU
- Consider using WSL2 for better compatibility with Linux-based tools

---

## Linux Setup

### Prerequisites
- Ubuntu 20.04+, Fedora 34+, or similar modern distribution
- sudo access

### Step 1: Update System
#### Ubuntu/Debian
```bash
sudo apt update
sudo apt upgrade -y
```

#### Fedora/CentOS/RHEL
```bash
sudo dnf update -y
# or for older versions:
sudo yum update -y
```

#### Arch Linux
```bash
sudo pacman -Syu
```

### Step 2: Install Python and Development Tools
#### Ubuntu/Debian
```bash
# Install Python and essential tools
sudo apt install -y python3 python3-pip python3-venv python3-dev
sudo apt install -y build-essential git curl wget

# Install additional dependencies for scientific computing
sudo apt install -y libopenblas-dev liblapack-dev gfortran
sudo apt install -y libjpeg-dev libpng-dev
```

#### Fedora/CentOS/RHEL
```bash
# Install Python and essential tools
sudo dnf install -y python3 python3-pip python3-devel
sudo dnf install -y gcc gcc-c++ make git curl wget

# Install additional dependencies
sudo dnf install -y openblas-devel lapack-devel
sudo dnf install -y libjpeg-turbo-devel libpng-devel
```

#### Arch Linux
```bash
# Install Python and essential tools
sudo pacman -S python python-pip python-virtualenv
sudo pacman -S base-devel git curl wget

# Install additional dependencies
sudo pacman -S openblas lapack
sudo pacman -S libjpeg-turbo libpng
```

### Step 3: Setup Project Environment
```bash
# Clone the repository
git clone https://github.com/576469377/AI-tutorial-by-AI.git
cd AI-tutorial-by-AI

# Create virtual environment
python3 -m venv ai_tutorial_env

# Activate virtual environment
source ai_tutorial_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python create_sample_data.py
```

### Step 4: Install Jupyter Lab
```bash
# Install JupyterLab
pip install jupyterlab

# Start JupyterLab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

### Step 5: GPU Support (NVIDIA)
If you have an NVIDIA GPU and want CUDA support:

```bash
# Check if NVIDIA driver is installed
nvidia-smi

# Install CUDA toolkit (Ubuntu)
sudo apt install nvidia-cuda-toolkit

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Linux Troubleshooting

**Issue: "Permission denied" for pip install**
```bash
# Install packages for current user
pip install --user package_name

# Or fix pip permissions
sudo chown -R $USER ~/.local
```

**Issue: "python3-distutils" not found (Ubuntu)**
```bash
sudo apt install python3-distutils python3-apt
```

**Issue: Matplotlib display issues**
```bash
# Install GUI backend
sudo apt install python3-tk

# For headless servers, use Agg backend
export MPLBACKEND=Agg
```

**Issue: Out of memory during compilation**
```bash
# Increase swap space temporarily
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Issue: CUDA version mismatch**
```bash
# Check CUDA version
nvcc --version

# Install compatible PyTorch version
# Check https://pytorch.org/get-started/locally/ for compatibility
```

### Linux Performance Tips
- Use fast SSD storage
- Monitor system resources with `htop`
- Use `screen` or `tmux` for long-running training sessions
- Consider using conda for package management in some cases

---

## Alternative Setup: Using Conda

For all platforms, Conda provides a consistent environment:

### Step 1: Install Conda
- Download Miniconda: https://docs.conda.io/en/latest/miniconda.html
- Follow platform-specific installation instructions

### Step 2: Create Environment
```bash
# Create conda environment
conda create -n ai_tutorial python=3.10

# Activate environment
conda activate ai_tutorial

# Install packages
conda install numpy pandas matplotlib seaborn scikit-learn jupyter
conda install pytorch torchvision torchaudio -c pytorch

# Clone and setup project
git clone https://github.com/576469377/AI-tutorial-by-AI.git
cd AI-tutorial-by-AI
python create_sample_data.py
```

## Alternative Setup: Using Docker

For a completely isolated environment:

### Create Dockerfile
```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y git
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

### Run with Docker
```bash
# Build image
docker build -t ai-tutorial .

# Run container
docker run -p 8888:8888 -v $(pwd):/app ai-tutorial
```

## Verification Steps

After setup on any platform, verify your installation:

```bash
# Test Python installation
python --version

# Test package imports
python -c "import numpy, pandas, matplotlib, sklearn; print('All packages imported successfully!')"

# Test PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Run basic example
python examples/01_numpy_pandas_basics.py

# Test Jupyter
jupyter lab --version
```

## Getting Help

### Common Resources
- **Documentation**: Check README.md in the project root
- **Issues**: Report problems on GitHub Issues
- **Community**: Stack Overflow, Reddit r/MachineLearning

### Platform-Specific Help
- **macOS**: Homebrew documentation, Apple Developer forums
- **Windows**: Microsoft Developer documentation, Windows Subsystem for Linux
- **Linux**: Distribution-specific forums, Package manager documentation

### Performance Monitoring
```bash
# Monitor CPU/Memory usage
htop  # Linux/macOS
# Task Manager (Windows)

# Monitor GPU usage (NVIDIA)
nvidia-smi -l 1

# Check disk space
df -h  # Linux/macOS
# dir C:\ (Windows)
```

---

**Next Steps**: After completing setup, start with the [Getting Started Guide](../getting_started.md) and then explore the tutorials in order!

Happy Learning! 🚀