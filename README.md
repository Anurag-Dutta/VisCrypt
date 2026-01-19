# VISCRYPT

**Recurrence-Based Signal Visualization for Texture-Driven Differential Cryptanalysis**

[![Paper](https://img.shields.io/badge/Paper-IEEE%20SPL-blue)](paper.pdf)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Contributions

### 1. **Recurrence Plot Representation**
Transforms 1D ciphertext differences → 2D texture images exposing:
- **Diagonals**: Diffusion strength (long=weak, short=strong)
- **Laminar bands**: Periodic state returns  
- **Isolated pixels**: Mixing quality

### 2. **State-of-the-Art Results**
| Cipher | Max Rounds | VISCRYPT | Prior SOTA | Gain |
|--------|------------|----------|------------|------|
| Ascon  | **3**      | **99.98%**| 98.61%    | +1.4%|
| Speck32/64 | **7** | **61.90%**| 53.13% | **+8.8%**

**11% average improvement**, **10x less data** (10⁶ samples)

### 3. **No Novel Architectures Needed**
Standard **ResNet/DenseNet** backbones achieve SOTA via better representation

### 4. **Three-Phase Framework**
