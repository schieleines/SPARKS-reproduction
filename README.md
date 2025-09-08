# SPARKS Analysis: Neural Hand Position Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 **Project Overview**

This repository contains a complete reproduction of **Figure 3** from the SPARKS paper: *"Expressive latent embeddings and state-of-the-art prediction from somatosensory cortex data"*. The analysis demonstrates SPARKS' superiority over control models in predicting monkey hand position from neural activity in the somatosensory cortex.

## 📊 **Key Results**

| Model | Final R² Score | Performance | Improvement |
|-------|----------------|-------------|-------------|
| **SPARKS** | **0.7756** | 🏆 **Excellent** | **+544%** |
| Conventional Attention | 0.2316 | ⚠️ Moderate | +132% |
| RNN VAE | 0.1584 | ⚠️ Moderate | +58% |

**SPARKS achieves 3.3x better performance** than conventional attention models!

## 🔬 **Technical Implementation**

### **Dataset**
- **Source**: sub-Han from Neural Latents Benchmark (NLB)
- **Brain Area**: Area 2 (somatosensory cortex) - optimal for hand position encoding
- **Data**: 65 neurons, 600 time points per trial, 8-directional reaching task
- **Format**: NWB (Neurodata Without Borders)

### **Models Compared**
1. **SPARKS**: HebbianTransformerEncoder with biologically-inspired attention
2. **Conventional Attention**: Standard TransformerEncoder
3. **RNN VAE**: Recurrent neural network variational autoencoder

### **Key Features**
- **Hebbian Learning**: STDP-based attention mechanisms
- **Multi-head Attention**: Biological plausibility
- **Latent Space Analysis**: DSA and embedding visualization
- **Performance Metrics**: R² scores, co-smoothing, cross-validation

## 📁 **Repository Structure**

```
SPARKS/
├── sparks/                          # Original SPARKS framework
│   ├── sparks/
│   │   ├── models/                  # Model architectures
│   │   ├── data/                    # Data loading utilities
│   │   └── utils/                   # Training and evaluation
│   └── pyproject.toml
├── sparks/sparks/scripts/monkey/    # Analysis scripts
│   ├── monkey_reaching.py          # Main SPARKS implementation
│   ├── controls/                    # Control models
│   │   ├── monkey_conventional_attention.py
│   │   └── monkey_vae.py
│   ├── visuals/                     # Generated figures
│   │   ├── figure3a_reaching_task.png
│   │   ├── figure3b_latent_embeddings.png
│   │   ├── figure3c_dsa_illustration.png
│   │   ├── figure3d_dsa_distances.png
│   │   ├── figure3f_r2_scores.png
│   │   ├── figure3g_cosmoothing.png
│   │   ├── figure3h_unsup_vs_sup.png
│   │   ├── actual_learning_curves.png
│   │   ├── performance_comparison.png
│   │   ├── ANALYSIS_REPORT.md
│   │   └── JOB_APPLICATION_SUMMARY.md
│   ├── create_corrected_figures.py  # Figure generation
│   └── analyze_actual_results.py   # Results analysis
├── paper.pdf                        # Original SPARKS paper
└── README.md                        # This file
```

## 🚀 **Quick Start**

### **Prerequisites**
```bash
pip install torch torchvision torchaudio
pip install scikit-learn tqdm matplotlib seaborn
pip install nlb_tools
```

### **Installation**
```bash
git clone https://github.com/schieleines/SPARKS-Analysis.git
cd SPARKS-Analysis
pip install -e ./sparks
```

### **Run Analysis**
```bash
cd sparks/sparks/scripts/monkey

# Run SPARKS model
python monkey_reaching.py --target_type hand_pos --n_epochs 30 --batch_size 32

# Run control models
python controls/monkey_conventional_attention.py --n_epochs 30 --batch_size 32
python controls/monkey_vae.py --enc_type linear --n_epochs 30 --batch_size 32

# Generate figures
python create_corrected_figures.py
python analyze_actual_results.py
```

## 📈 **Analysis Components**

### **Figure 3 Reproduction**
- **3A**: Reaching task illustration with hand trajectories
- **3B**: Latent space comparison across models
- **3C**: Dynamical systems analysis (DSA)
- **3D**: Distance metrics comparison
- **3F**: Learning curves and performance
- **3G**: Generalization performance (co-smoothing)
- **3H**: Unsupervised vs supervised comparison

### **Additional Analysis**
- **Learning Curves**: Real training progress from trained models
- **Performance Comparison**: Direct model comparison
- **Statistical Analysis**: Cross-validation and significance testing

## 🧠 **Scientific Understanding**

### **Why SPARKS Works Better**
1. **Biological Inspiration**: Hebbian learning mimics real neural plasticity
2. **Attention Mechanisms**: Biologically-constrained attention with STDP
3. **Temporal Processing**: Captures spike timing relationships
4. **Unsupervised Learning**: Learns meaningful patterns without supervision

### **Dataset Choice**
- **Area 2 (Somatosensory Cortex)**: Directly encodes hand position and proprioception
- **Sub-Han Dataset**: High-quality neural recordings from reaching tasks
- **8-Directional Task**: Systematic variation in hand position

## 📊 **Results Interpretation**

### **R² Score (0.776)**
- **Meaning**: SPARKS explains 77.6% of hand position variance
- **Significance**: Excellent prediction accuracy for neural data
- **Comparison**: 3.3x better than conventional attention

### **Co-smoothing Score (0.72)**
- **Purpose**: Measures generalization to new data
- **SPARKS Advantage**: Better generalization due to biological learning
- **Significance**: Model learned meaningful patterns, not just memorization

### **DSA Analysis**
- **Purpose**: Measures temporal structure in latent representations
- **SPARKS Advantage**: Lower distances = better temporal alignment
- **Biological Relevance**: Real neural activity has temporal structure

## 🎯 **Job Application Value**

This analysis demonstrates:
- **Technical Proficiency**: Complete SPARKS implementation
- **Scientific Understanding**: Biological principles and methodology
- **Data Analysis Skills**: Real neural data processing
- **Research Reproducibility**: Exact paper figure replication
- **Comparative Analysis**: Multiple model evaluation
- **Visualization**: Professional scientific figures

## 📚 **References**

- **SPARKS Paper**: [Link to paper]
- **Neural Latents Benchmark**: [https://neurallatents.github.io/](https://neurallatents.github.io/)
- **Sub-Han Dataset**: Area2_Bump from NLB

## 🤝 **Contributing**

This is a reproduction study for research and educational purposes. Feel free to:
- Fork the repository
- Submit issues for bugs or improvements
- Use the code for your own research

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 **Author**

**Ines Sebti** - [@schieleines](https://github.com/schieleines)

---

**Ready for computational neuroscience job applications! 🎉**