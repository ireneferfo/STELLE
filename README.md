# STELLE 🌟  
**Signal Temporal logic Embedding for Logically-grounded Learning and Explanation**

## Overview
**STELLE** is a neuro-symbolic framework for **interpretable time series classification**.  
It unifies *classification* and *explanation* through direct embedding of trajectories into a space of **Signal Temporal Logic (STL)** concepts.

Each prediction made by STELLE is accompanied by **human-readable logical explanations**, both:
- **Local** — explaining individual predictions  
- **Global** — describing class-level temporal behaviour  

The method was introduced in:  
> *Irene Ferfoglia, Simone Silvetti, Gaia Saveri, Laura Nenzi, and Luca Bortolussi.*  
> **Guided by Stars: Interpretable Concept Learning Over Time Series via Temporal Logic Semantics**  
> *submitted to Journal of Artificial Intelligence Research (JAIR), 2025.*

## Features
- Interpretable-by-design classification using STL formulae  
- Trajectory embedding kernel based on temporal robustness  
- Dual explanations — local and global  
- Compatible with multivariate time series  
- Implemented in PyTorch

## Citation
If you use STELLE in your research, please cite: 

```bibtex
@article{ferfoglia2025stelle,
  title     = {Guided by Stars: Interpretable Concept Learning Over Time Series via Temporal Logic Semantics},
  author    = {Ferfoglia, Irene and Silvetti, Simone and Saveri, Gaia and Nenzi, Laura and Bortolussi, Luca},
  year      = {2025}
}
```