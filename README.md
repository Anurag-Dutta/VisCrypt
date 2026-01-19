# VISCRYPT

**Recurrence-Based Signal Visualization for Texture-Driven Differential Cryptanalysis**

[![Paper](https://img.shields.io/badge/Paper-IEEE%20SPL-blue)](paper.pdf)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-green)](https://zenodo.org/records/18223783)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**VISCRYPT** transforms 1D ciphertext-difference sequences into 2D recurrence-plot texture images, enabling superior CNN-based differential cryptanalysis distinguishers on reduced-round Ascon and Speck32/64.

## 🚀 Key Contributions

### 1. **Recurrence Plot Representation**
Transforms 1D ciphertext differences → 2D texture images exposing:
- **Diagonals**: Diffusion strength (long=weak, short=strong)
- **Laminar bands**: Periodic state returns  
- **Isolated pixels**: Mixing quality

### 2. **State-of-the-Art Results**
| Cipher | Max Rounds | VISCRYPT (10⁴) | Prior SOTA | Gain |
|--------|------------|----------------|------------|------|
| **Ascon**  | **3**      | **99.98%**     | 98.61%    | **+1.4%** |
| **Speck32/64** | **7** | **61.90%** | 53.13% | **+8.8%** |

**11% average improvement**
### 3. **No Novel Architectures Needed**
Standard **ResNet{18,34,50,101,152}/DenseNet{121,169,201,264}** achieve SOTA via texture representation

### 4. **Three-Phase Framework**
