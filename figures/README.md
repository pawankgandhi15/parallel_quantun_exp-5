# Figures & Visualizations Directory (`figures/`)

This directory contains publication-grade, high-resolution scientific diagrams, 16:9 visual renders, and scalable vector graphics (`.svg`) generated from the technical specifications in [`cur_arc.md`](../cur_arc.md), [`cur_imple.md`](../cur_imple.md), and the source research paper (*A Parallel Hybrid Quantum-Classical Convolutional Design Using Parameterized Quantum Circuits for Image Classification*, Liu & Lou, *Quantum Engineering* 2026, Article 6643049).

---

## Complete Visual Assets Catalog

| # | Filename | Format | Primary Reference Section | Description |
|---|---|---|---|---|
| **1** | [`qc_cnn_parallel_architecture.svg`](qc_cnn_parallel_architecture.svg)<br>[`qc_cnn_parallel_architecture.jpg`](qc_cnn_parallel_architecture.jpg) | Vector SVG &<br>High-Res 16:9 | `cur_arc.md` §2.1–§2.7<br>`cur_imple.md` §2 | **Complete Proposed Architecture Dataflow:** Shows input $[B, 1, 28, 28]$, parallel classical Conv2d $[B, 8, 14, 14]$ and quantum PQC $[B, 4, 14, 14]$ branches, channel concatenation $[B, 12, 14, 14]$, flatten ($2352$), and 3-layer dense classification head ($128 \to 64 \to 10$). |
| **2** | [`pqc_circuit11_schematic.svg`](pqc_circuit11_schematic.svg)<br>[`pqc_circuit11_diagram.jpg`](pqc_circuit11_diagram.jpg) | Vector SVG &<br>High-Res 16:9 | `cur_arc.md` §2.4<br>`cur_imple.md` §2.4 | **4-Qubit Circuit 11 Quantum Gate Schematic:** All 4 qubit wires ($q_0 \dots q_3$), Hadamard + $RY(x_i)$ angle encoding, Variational Layer 1 ($RY$ + $CRX$ Circle), Variational Layer 2 ($RY$ + Shifted $CRX$ Circle), and Pauli-$Z$ measurement detectors. |
| **3** | [`shallow_vs_deep_philosophy.svg`](shallow_vs_deep_philosophy.svg)<br>[`shallow_parallel_vs_deep_design.jpg`](shallow_parallel_vs_deep_design.jpg) | Vector SVG &<br>High-Res 16:9 | `cur_arc.md` §1 & §2.8<br>`cur_imple.md` §2.8 | **Design Philosophy Comparison:** Deep Sequential Hybrids suffering from Barren Plateaus ($\text{Var}[\partial \mathcal{L}/\partial \theta] \sim \mathcal{O}(1/2^n)$) vs. Shallow Parallel Hybrids preserving active gradient variance. |
| **4** | [`model_suite_architecture_comparison.svg`](model_suite_architecture_comparison.svg)<br>[`model_suite_architecture_comparison.jpg`](model_suite_architecture_comparison.jpg) | Vector SVG &<br>High-Res 16:9 | `cur_arc.md` §3 | **Extended 6-Model Architecture Comparison Suite:** Side-by-side architectural flow, channel dimensions, and parameter counts for `QCCNNParallel`, `ClassicalCNN`, `ClassicalOnlyCNN`, `QuantumOnlyCNN`, `ClassicalExtendedCNN`, and `ScalableQCCNNParallel`. |
| **5** | [`ablation_and_scalability_study.svg`](ablation_and_scalability_study.svg)<br>[`ablation_and_scalability_study.jpg`](ablation_and_scalability_study.jpg) | Vector SVG &<br>High-Res 16:9 | `cur_arc.md` §4.5 & §4.6 | **Ablation & Scalability Multi-Panel Study:** Panel (a) Experiment 4 Branch Synergy (+2.95% over parameter-matched classical baseline); Panel (b) Experiment 5 Part A Qubit scaling sweet spot at $N=4$; Panel (c) Experiment 5 Part B Barren plateau onset at circuit depth $L \ge 4$. |
| **6** | [`noise_robustness_analysis.svg`](noise_robustness_analysis.svg)<br>[`noise_robustness_comparison.jpg`](noise_robustness_comparison.jpg) | Vector SVG &<br>High-Res 16:9 | `cur_arc.md` §4.4<br>`cur_imple.md` §7 | **Quantum Noise Robustness Evaluation (Tables 5–8):** 4-panel comparison across error rates $p \in \{0.0, 0.1, 0.2, 0.3\}$ under Data Noise, Bit-Flip, Phase-Flip, and Depolarizing channels for Proposed vs. HQNN-Quanv vs. CNN/QNN. |
| **7** | [`model_comparison_barchart.svg`](model_comparison_barchart.svg) | Scalable Vector Graphic | `cur_arc.md` §2.8 & §4.3<br>`cur_imple.md` §6 | **Model Parameter Efficiency vs. Accuracy Bar Chart:** Compares convolutional parameter counts (136 vs. 304–512) and clean MNIST test accuracy (90.05%) across 7 literature architectures. |
| **8** | [`architecture_differences_matrix.svg`](architecture_differences_matrix.svg) | Scalable Vector Graphic | `cur_arc.md` §5 | **Architecture Specification vs. Implementation Matrix:** Comprehensive 8-dimension discrepancy matrix comparing `docs/ARCHITECTURE.md` baseline against the current codebase. |

---

## How to View and Embed

### In Markdown:
```markdown
![Model Suite Comparison](figures/model_suite_architecture_comparison.svg)
![Ablation and Scalability](figures/ablation_and_scalability_study.svg)
```

### In Research Papers & Presentations:
- **Vector SVGs:** Infinite resolution with selectable text, ideal for LaTeX documents and publication PDF compilation.
- **16:9 JPG Renders:** Stylized dark-mode scientific graphics ideal for slide decks, conference presentations, and web dashboards.
