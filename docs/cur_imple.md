# Current Implementation — QC-CNN-Parallel

> **Source paper:** *A Parallel Hybrid Quantum-Classical Convolutional Design Using
> Parameterized Quantum Circuits for Image Classification*
> Haoxuan Liu & Xiaoping Lou — Quantum Engineering (2026), article 6643049
> DOI: 10.1155/que2/6643049

---

## Table of Contents

1. [Experiment Overview](#1-experiment-overview)
2. [Architecture — Full Technical Description](#2-architecture--full-technical-description)
3. [Code Implementation Walkthrough](#3-code-implementation-walkthrough)
4. [Parameter Budget](#4-parameter-budget)
5. [Training Configuration (Paper-Verified)](#5-training-configuration-paper-verified)
6. [Three Experiments in the Paper](#6-three-experiments-in-the-paper)
7. [Key Results from the Paper](#7-key-results-from-the-paper)
8. [How This Implementation Differs from the Paper](#8-how-this-implementation-differs-from-the-paper)
9. [Implementation File Map](#9-implementation-file-map)
10. [Reproduction Checklist](#10-reproduction-checklist)

---

## 1. Experiment Overview

The experiment implements a **hybrid quantum-classical convolutional neural network
(QC-CNN-Parallel)** for grayscale image classification. The core idea is inspired by
Inception-style parallel multi-scale feature extraction:

- A **classical convolutional branch** (4x4 kernel) captures local spatial texture.
- A **quantum convolutional branch** (2x2 sliding window) applies a trainable 4-qubit
  Parameterized Quantum Circuit (PQC) to every non-overlapping image patch.
- Both branches run **in parallel** on the same input. Their outputs are **concatenated**
  channel-wise before a 3-layer fully connected classification head.

**Key innovations over prior work:**

1. Quantum and classical convolutions run **in parallel**, not sequentially, capturing
   complementary local and nonlinear features simultaneously.
2. The PQC (Circuit 11) is selected via a rigorous 3-metric evaluation:
   **Expressibility**, **Meyer-Wallach Entanglement**, and a novel **Discreteness**
   (gradient variance) metric.
3. Only **136 classical + 16 quantum = 152 convolutional parameters** — the fewest of
   any model compared in the paper.

---

## 2. Architecture — Full Technical Description

### 2.1 High-Level Data Flow

```
Input Image: [B, 1, 28, 28]
                    |
          +---------+----------+
          |                    |
   Classical Branch       Quantum Branch
   Conv2d(1->8, 4x4,      2x2 sliding window
   stride=2, padding=1)   stride=2, 4 qubits
   ReLU                   PQC (Circuit 11)
          |                    |
   [B, 8, 14, 14]       [B, 4, 14, 14]
          |                    |
          +------ Concat ------+
                    |
             [B, 12, 14, 14]
                    |
                 Flatten
             [B, 2352]
                    |
        Linear(2352->128) + ReLU
                    |
         Linear(128->64) + ReLU
                    |
           Linear(64->C)
                    |
            Logits: [B, C]
```

For MNIST, C=10. Both branches output 14x14 spatial maps for 28x28 input.

---

### 2.2 Classical Convolutional Branch

A single Conv2d layer followed by ReLU:

| Parameter | Value | Reason |
|---|---|---|
| in_channels | 1 | Grayscale input |
| out_channels | 8 | 8 learnable filters |
| kernel_size | 4x4 | Larger receptive field vs. quantum 2x2 |
| stride | 2 | Downsamples to 14x14 |
| padding | 1 | Ensures 14x14 output for 28x28 input |

Output size: floor((28 + 2*1 - 4) / 2) + 1 = 14

Output shape: [B, 8, 14, 14]

Trainable parameters: 8 * 1 * 4 * 4 + 8 = 128 + 8 = **136** (matches paper Table 4).

---

### 2.3 Quantum Convolutional Branch

Uses a **2x2 sliding window with stride 2** over the input image. For 28x28 images this
creates a 14x14 grid of non-overlapping patches (196 patches total per image).

Each patch is processed:

1. **Extract patch:** P[i,j] = image[b, 0, 2i:2i+2, 2j:2j+2]
2. **Flatten and encode:** angles = flatten(P[i,j]) * pi  (maps [0,1] to [0, pi])
3. **Run PQC:** [z0, z1, z2, z3] = quantum_circuit(angles, weights)  (each zk in [-1, 1])
4. **Store features:** quantum_output[b, :, i, j] = [z0, z1, z2, z3]

Output shape: [B, 4, 14, 14]

The **same 16 PQC weights are shared across all 196 patches**, analogous to classical
kernel weight sharing.

---

### 2.4 Parameterized Quantum Circuit (Circuit 11)

Circuit 11 is a **4-qubit, 2-layer variational circuit with Circle entanglement**.
Selected after evaluating 11 PQC architectures.

The 16 trainable parameters split into 4 groups of 4:

```
weights[0:4]   -> w_rot1  (rotation layer 1: single-qubit RY)
weights[4:8]   -> w_ent1  (entangling layer 1: CRX angles)
weights[8:12]  -> w_rot2  (rotation layer 2: single-qubit RY)
weights[12:16] -> w_ent2  (entangling layer 2: CRX, SHIFTED start)
```

Full circuit gate sequence:

```
Encoding (angle encoding, all 4 qubits):
  For qubit k in {0,1,2,3}:
    H(qk)               <- Hadamard creates superposition
    RY(angles[k], qk)   <- encode pixel as rotation angle

Variational Layer 1:
  RY(w_rot1[k], qk)  for k = 0,1,2,3

Entangling Layer 1 (Circle, starts q0->q1):
  CRX(w_ent1[0], q0 -> q1)
  CRX(w_ent1[1], q1 -> q2)
  CRX(w_ent1[2], q2 -> q3)
  CRX(w_ent1[3], q3 -> q0)

Variational Layer 2:
  RY(w_rot2[k], qk)  for k = 0,1,2,3

Entangling Layer 2 (Circle, SHIFTED starts q1->q2):
  CRX(w_ent2[0], q1 -> q2)
  CRX(w_ent2[1], q2 -> q3)
  CRX(w_ent2[2], q3 -> q0)
  CRX(w_ent2[3], q0 -> q1)

Measurement:
  Return [<Z0>, <Z1>, <Z2>, <Z3>]
```

The **shift** in the second entangling layer (q1 instead of q0 as start) is Circuit 11's
key feature — it breaks symmetry with the first layer, prevents overlapping patterns, and
increases expressive capacity (paper Eq. 24, page 9).

**Why Circuit 11 was selected:**

| Circuit | Params | Expressibility (lower=better) | Entanglement | Discreteness | MNIST Acc |
|---|---:|---:|---:|---:|---:|
| Circuit 10 | 28 | 0.0013 | 0.7180 | 0.0208 | 0.8254 |
| **Circuit 11** | **16** | **0.0071** | **0.5463** | **0.0191** | **0.8057** |
| RY All-to-All | 4 | 0.3454 | 0.4520 | 0.1260 | 0.8006 |

Circuit 11 achieves near-equal accuracy to Circuit 10 with **half the parameters (16 vs 28)**,
making it hardware-efficient under NISQ constraints.

---

### 2.5 Quantum Measurement

After the PQC, the **Pauli-Z expectation value** is measured on each qubit:

  E(Zk) = P(qk=0) - P(qk=1)  in [-1, 1]

Pauli-Z is chosen because:
- Its eigenstates |0> and |1> are the default computational basis of quantum hardware.
- The bounded range [-1, 1] is numerically stable as input to classical dense layers.
- Z-basis measurement is directly supported at the hardware level.

PennyLane returns analytic expectation values (no shot noise) when no `shots` argument
is specified, which is the case in the current implementation.

---

### 2.6 Feature Fusion

```python
x_fused = torch.cat((x_class, x_quant), dim=1)
# [B, 8, 14, 14] + [B, 4, 14, 14] -> [B, 12, 14, 14]
```

Both branches produce aligned 14x14 spatial maps via stride-2 operations.

---

### 2.7 Classification Head

```
Flatten: [B, 12, 14, 14] -> [B, 2352]   (12 * 14 * 14 = 2352)

Layer 1: Linear(2352, 128) -> ReLU
Layer 2: Linear(128, 64)   -> ReLU
Layer 3: Linear(64, C)     -> raw logits (no softmax)
```

CrossEntropyLoss applies log-softmax internally — do not apply softmax before it.

---

### 2.8 Design Philosophy — Shallow, Parallel Hybrid Architecture

> [!NOTE]
> This is the central design principle of the paper. The model increases **width** (parallel
> branches) rather than **depth** (stacked quantum layers) to improve performance. This is a
> deliberate choice to avoid the barren plateau problem that plagues deep PQCs.

The QC-CNN-Parallel architecture is intentionally **shallow and parallel**, not deep and sequential.
This decision stems from two limitations of deeper hybrid quantum architectures:

1. **Barren plateau problem (McClean et al., 2018):** As quantum circuit depth increases,
   gradient variance decreases exponentially — `Var[∂L/∂θ] ~ O(1/2^n)`. Deep PQCs become
   untrainable because gradients vanish across the parameter space.

2. **Computational overhead:** Each additional quantum layer multiplies the number of circuit
   evaluations needed per forward and backward pass, making training prohibitively slow under
   NISQ conditions.

**The parallel design solves both problems simultaneously:**

| Design axis | This paper (QC-CNN-Parallel) | Prior sequential hybrids |
|---|---|---|
| Quantum circuit depth | **Shallow (2 variational layers, Circuit 11)** | Deeper PQC stacks |
| Feature extraction | **Width: parallel classical + quantum branches** | Depth: sequential layers |
| Gradient stability | **Preserved — shallow circuits avoid barren plateaus** | Degraded with depth |
| Computational cost | **Lower — fewer circuit evaluations per layer** | Higher with each added layer |
| Inspiration | Inception networks (parallel multi-scale branches) | VGG/ResNet (sequential depth) |

**The classical branch handles:**
- Local spatial feature extraction via a 4×4 convolutional kernel.
- Standard gradient flow through ReLU activations.

**The quantum branch handles:**
- Nonlinear transformations via quantum superposition and entanglement.
- Global nonlinear interactions across the 2×2 patch — complementary to classical local receptive fields.

**Together they provide complementary, multi-scale representations** — the classical branch
captures local texture and edges while the quantum branch models nonlinear patch interactions
across the Hilbert space. This complementarity is why concatenating both branches outperforms
either branch alone.

The paper validates this: QC-CNN-Parallel uses only **depth 3** (2 parallel conv branches +
dense head) vs. **depth 4** for all other compared models (CNN, QC-CNN, HQNN-Quanv, VCNN,
QC-ResNet), yet achieves higher accuracy with fewer convolutional parameters (136 vs. 304–512).
The shallow PQC (Circuit 11, just 2 variational layers) also demonstrates strong robustness
under bit-flip, phase-flip, and depolarizing noise — a direct benefit of hardware-efficient
shallow circuits on NISQ devices (paper Section 5, conclusion).

---

## 3. Code Implementation Walkthrough

### 3.1 Quantum Device Setup

File: [qc-cnn-parallel.py](file:///E:/parallel_quantum-5/parallel_quantum/QC-CNN-Parallel1/qc-cnn-parallel.py) (lines 7-10)

```python
import pennylane as qml

num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)
```

- `default.qubit` = PennyLane pure-state simulator (2^4 = 16 complex amplitudes).
- No physical QPU — all execution is on classical CPU.
- Analytic expectation values (no finite-shot sampling noise).
- For noise experiments: swap to `qml.device("default.mixed", wires=4)`.

---

### 3.2 quantum_circuit — The PQC QNode

File: [qc-cnn-parallel.py](file:///E:/parallel_quantum-5/parallel_quantum/QC-CNN-Parallel1/qc-cnn-parallel.py) (lines 15-66)

```python
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Angle encoding
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
        qml.RY(inputs[i], wires=i)

    # Rotation layer 1
    for i in range(num_qubits):
        qml.RY(weights[i], wires=i)

    # Entangling layer 1 (Circle, q0->q1 start)
    for i in range(num_qubits):
        qml.CRX(weights[4 + i], wires=[i, (i + 1) % num_qubits])

    # Rotation layer 2
    for i in range(num_qubits):
        qml.RY(weights[8 + i], wires=i)

    # Entangling layer 2 (Circle, shifted q1->q2 start)
    for i in range(num_qubits):
        qml.CRX(weights[12 + i], wires=[(i + 1) % num_qubits, (i + 2) % num_qubits])

    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
```

The `interface="torch"` decorator enables PyTorch autograd. Quantum gradients are
computed via the **parameter-shift rule** (paper Eq. 17):

  dE/d(theta_i) = (1/2) * [E(theta + pi/2 * e_i) - E(theta - pi/2 * e_i)]

---

### 3.3 QuantumConvLayer — Sliding Quantum Window

File: [qc-cnn-parallel.py](file:///E:/parallel_quantum-5/parallel_quantum/QC-CNN-Parallel1/qc-cnn-parallel.py) (lines 72-104)

```python
class QuantumConvLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(16))   # 16 trainable PQC params

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        out_h, out_w = h // 2, w // 2
        out = torch.zeros((batch_size, 4, out_h, out_w), ...)

        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    patch = x[b, 0, i*2:(i*2)+2, j*2:(j*2)+2]
                    flattened_patch = patch.flatten() * np.pi    # [0,1] -> [0,pi]
                    q_features = torch.stack(quantum_circuit(flattened_patch, self.weights))
                    out[b, :, i, j] = q_features
        return out
```

**Performance note:** For batch=32 and 28x28 images, this makes **32*14*14 = 6,272
QNode calls per forward pass**. This triple loop is the main computational bottleneck.

---

### 3.4 QCCNNParallel — The Full Model

File: [qc-cnn-parallel.py](file:///E:/parallel_quantum-5/parallel_quantum/QC-CNN-Parallel1/qc-cnn-parallel.py) (lines 110-154)

```python
class QCCNNParallel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.classical_conv = nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1)  # 136 params
        self.quantum_conv   = QuantumConvLayer()                                    # 16 params
        self.fc1 = nn.Linear(12 * 14 * 14, 128)                                   # 301,184
        self.fc2 = nn.Linear(128, 64)                                              # 8,256
        self.fc3 = nn.Linear(64, num_classes)                                      # 650

    def forward(self, x):
        x_class = F.relu(self.classical_conv(x))         # [B, 8, 14, 14]
        x_quant = self.quantum_conv(x)                    # [B, 4, 14, 14]
        x_fused = torch.cat((x_class, x_quant), dim=1)   # [B, 12, 14, 14]
        x_flat  = x_fused.view(x_fused.size(0), -1)      # [B, 2352]
        x_fc    = F.relu(self.fc1(x_flat))               # [B, 128]
        x_fc    = F.relu(self.fc2(x_fc))                 # [B, 64]
        logits  = self.fc3(x_fc)                          # [B, C]
        return logits
```

---

### 3.5 Training and Demo Pipeline

The `__main__` block (lines 160-191) is a **smoke test only** — not a full training run:

```python
model     = QCCNNParallel(num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn   = nn.CrossEntropyLoss()

dummy_images = torch.rand((4, 1, 28, 28))   # batch_size=4
dummy_labels = torch.randint(0, 10, (4,))

logits = model(dummy_images)
loss   = loss_fn(logits, dummy_labels)
optimizer.zero_grad()
loss.backward()    # gradients flow to both PQC + classical weights
optimizer.step()   # one Adam update
```

Replace dummy data with MNIST and loop for 50 epochs for real training.

---

## 4. Parameter Budget

| Component | Trainable Parameters | Notes |
|---|---:|---|
| Classical Conv2d (4x4, 8 filters) | **136** | 128 weights + 8 biases |
| Quantum PQC (Circuit 11) | **16** | 8 rotation + 8 entangling |
| FC1: 2352 -> 128 | 301,184 | 2352*128 + 128 biases |
| FC2: 128 -> 64 | 8,256 | 128*64 + 64 biases |
| FC3: 64 -> 10 | 650 | 64*10 + 10 biases |
| **Total (C=10)** | **310,242** | |

> The paper's Table 4 reports only **convolutional parameters (136)** for comparison,
> since all 7 models compared share the same linear classification head.

Circuit evaluations per forward pass: B * 14 * 14 = 196B QNode calls.
At batch size 32: **6,272 evaluations** per forward pass.

---

## 5. Training Configuration (Paper-Verified)

All settings verified from Table 4 (page 10) and Section 4.2 (page 7):

| Setting | Value | Paper Source |
|---|---|---|
| Framework | PyTorch + PennyLane | Section 4.2.2 |
| Quantum interface | interface="torch" (parameter-shift gradient) | Section 4.2.2 |
| Quantum device (main) | default.qubit | Section 4.2.2 |
| Quantum device (noise) | default.mixed | Section 4.3.3 |
| Qubits | 4 | Table 4 |
| PQC | Circuit 11, 16 parameters | Tables 2-3 |
| Optimizer | Adam | Table 4 |
| Learning rate | 0.01 | Table 4 |
| Batch size (main) | **32** | Table 4 |
| Batch size (noise exp.) | **100** | Section 4.3.3 |
| Epochs | **50** | Table 4 |
| Random seed | 42 | Table 4 |
| Loss function | Cross-entropy | Table 4 / Section 3.5 |
| Gradient method | Parameter-shift rule | Section 3.5, Eq. 17 |

---

## 6. Three Experiments in the Paper

### Experiment 1 — PQC Structure Comparison (Section 4.3.1)

Compares **11 different PQC architectures** using a simplified model (single quantum
conv + classical linear layer) to isolate each PQC's contribution.

Three gate types x three topologies = 9 basic circuits:
- Gate types: RX, RY, RZ
- Topologies: Linear, Circle, All-to-All
- Plus Circuit 10 and Circuit 11 (custom, from Sim et al.)

Each circuit evaluated on 3 metrics:
- **Expressibility (Expr):** KL divergence from Haar random distribution. Lower = better.
- **Entangling (Ent):** Meyer-Wallach measure. Higher = stronger entanglement.
- **Discreteness (Disc):** NEW metric — mean variance of gradients over N random
  initializations. Captures gradient heterogeneity. Circuits with near-zero Disc
  (like RZ gates) suffer from barren plateaus.

Key finding: **RZ circuits** have near-zero Discreteness (2.7e-33 to 3.6e-33) — they
hit barren plateaus. **RY circuits** have high Discreteness but limited expressibility.
**Circuit 11** balances all three, with only 16 parameters.

---

### Experiment 2 — Model Comparison (Section 4.3.2)

Compares QC-CNN-Parallel (with Circuit 11) against 6 other models on 3 datasets:

| Model | Based on | Conv. Params | Key Characteristic |
|---|---|---:|---|
| Classical CNN | LeNet-5 | 464 | Classical baseline |
| QC-CNN | Henderson et al. (2020) | 448 | Fixed (non-trainable) quantum filters |
| HQNN-Quanv | Senokosov et al. (2024) | 448 | Trainable PQC, quanvolutional layer |
| VCNN | Huang et al. (2022) | 456 | Variational CNN |
| QC-ResNet | Shi et al. (2023) | 512 | Quantum ResNet |
| QC-Inception | Wang et al. (2025) | 304 | Quantum Inception |
| **QC-CNN-Parallel** | **This paper** | **136** | **Parallel hybrid, proposed** |

Result: QC-CNN-Parallel achieves highest accuracy on all 3 datasets with fewest parameters.

---

### Experiment 3 — Noise Robustness (Section 4.3.3)

Tests robustness under 4 noise types with error rates p in {0.1, 0.2, 0.3}:

| Noise Type | Mathematical Model |
|---|---|
| Data noise | Gaussian noise added to input images after dimensionality reduction |
| Bit-flip | rho -> (1-p)*rho + p*X*rho*X |
| Phase-flip | rho -> (1-p)*rho + p*Z*rho*Z |
| Depolarizing | rho -> (1-p)*rho + (p/3)*(X*rho*X + Y*rho*Y + Z*rho*Z) |

Uses `default.mixed` simulator (supports noise channels). Batch size = 100 due to
`default.mixed` not supporting batched inputs.

---

## 7. Key Results from the Paper

### Circuit Comparison (Tables 2-3, pages 9-10)

| Circuit | Params | Expr (lower=better) | Ent | Disc | MNIST Acc | FashMNIST Acc |
|---|---:|---:|---:|---:|---:|---:|
| RX-Linear | 4 | 0.1755 | 0.5618 | 0.0280 | 0.6560 | 0.7364 |
| RX-Circle | 4 | 0.1679 | 0.7448 | 0.0060 | 0.6066 | 0.6864 |
| RY-Circle | 4 | 0.3552 | 0.6372 | 0.0607 | 0.7602 | 0.7763 |
| RY-All-to-All | 4 | 0.3454 | 0.4520 | 0.1260 | 0.8006 | 0.7846 |
| RZ-Circle | 4 | 0.1670 | 0.7924 | 3.6e-33 | 0.5481 | 0.6213 |
| Circuit 10 | 28 | 0.0013 | 0.7180 | 0.0208 | 0.8254 | 0.7946 |
| **Circuit 11** | **16** | **0.0071** | **0.5463** | **0.0191** | **0.8057** | **0.7778** |

### Noise Robustness Results (Tables 5-8, pages 12-13, MNIST)

| Noise Type | Model | No noise | p=0.1 | p=0.2 | p=0.3 |
|---|---|---:|---:|---:|---:|
| Data noise | **Proposed** | 0.9005 | 0.8915 | 0.8900 | **0.8425** |
| | HQNN-Quanv | 0.8320 | 0.8344 | 0.7796 | 0.7125 |
| | CNN | 0.8935 | 0.8840 | 0.8530 | 0.7844 |
| Bit-flip | **Proposed** | 0.9005 | 0.8769 | 0.8558 | **0.8405** |
| | HQNN-Quanv | 0.8320 | 0.6775 | 0.6523 | 0.6399 |
| | QNN | 0.8350 | 0.7115 | 0.6124 | 0.4615 |
| Phase-flip | **Proposed** | 0.9005 | 0.8836 | 0.8618 | **0.8602** |
| | HQNN-Quanv | 0.8320 | 0.8279 | 0.8254 | 0.8267 |
| Depolarizing | **Proposed** | 0.9005 | 0.8639 | 0.8664 | **0.8327** |
| | HQNN-Quanv | 0.8320 | 0.7021 | 0.6502 | 0.6059 |
| | QNN | 0.8350 | 0.7552 | 0.6944 | 0.5904 |

**Headline figures (abstract):**
- **+4.89%** average training accuracy over existing hybrid quantum CNNs.
- **+6.24%** average training accuracy over classical CNNs.

---

## 8. How This Implementation Differs from the Paper

> [!IMPORTANT]
> **Bottom line:** The architecture is 100% faithful to the paper; what's missing is
> the full training loop, real datasets, and the noise experiments.
>
> - **Architecture (Circuit 11, classical branch, fusion, dense head):** ✅ Exactly matches paper Sections 3.1–3.5.
> - **Full training loop (50 epochs, real MNIST, batch=32, seed=42):** ❌ Not yet implemented — current code is a smoke test only.
> - **Real datasets (MNIST / Fashion-MNIST / Overhead-MNIST):** ❌ Not loaded — current code uses random dummy tensors.
> - **Noise robustness experiments (default.mixed, bit-flip / phase-flip / depolarizing):** ❌ Not implemented.
> - **Baseline model comparison (6 competing models):** ❌ Not implemented.

### 8.1 Status Table — Paper vs. Implementation

| Aspect | Paper Specification | Current Implementation | Status |
|---|---|---|---|
| Training data | MNIST (10k train / 2k test, balanced subsampled) | Random dummy tensors | NOT DONE |
| Training epochs | 50 | 1 forward+backward step (smoke test) | NOT DONE |
| Batch size | 32 (main) / 100 (noise) | 4 (demo only) | NOT FULL |
| Random seed | 42 | Not set | NOT SET |
| Evaluation | Test accuracy, loss curves over 50 epochs | None | NOT DONE |
| Noise experiments | default.mixed + bit/phase/depol noise | Not implemented | NOT DONE |
| Baseline models | 6 competing models | Not implemented | NOT DONE |
| PQC architecture | Circuit 11 (16 params, 2-layer CRX + RY) | Correct | OK |
| Classical branch | Conv2d(1,8,4x4,stride=2,pad=1) | Correct | OK |
| Feature fusion | Channel-wise concat | Correct | OK |
| Dense head | 2352->128->64->C with ReLU | Correct | OK |
| Optimizer | Adam, lr=0.01 | Adam, lr=0.01 | OK |
| Loss | CrossEntropyLoss | CrossEntropyLoss | OK |
| Parameter-shift rule | Used for PQC gradients | Implicit via interface="torch" | OK |

---

### 8.2 Architecture Differences — NONE

The `QCCNNParallel` class **exactly matches** Figure 1 and Sections 3.1-3.5 of the paper:

- Classical Conv2d: kernel 4x4, stride 2, pad 1, 8 filters -> 136 params (Table 4). OK
- Quantum Conv: 2x2 window, stride 2, 4 qubits, Pauli-Z measurement (Sec 3 / Fig 2). OK
- Angle encoding: H gate + RY(x*pi) per qubit (Eq. 6 / Fig 3). OK
- Circuit 11: 2 RY rotation layers + 2 CRX Circle entangling layers, shifted (Eq. 24). OK
- Feature fusion: channel-wise concatenation -> 12 channels (Sec 3.4). OK
- Dense head: 3 layers, ReLU activations (Sec 3.4, page 6). OK

**Minor implementation note:** The paper states Circuit 11's first entangling layer starts
at q1 and the second at q2. The code uses the loop offset `(i+1)%4 -> (i+2)%4` for the
second entangling layer, which produces the **identical** ring topology starting q1->q2.
These are equivalent implementations.

---

### 8.3 Training Differences

| Difference | Paper | This Code | Impact |
|---|---|---|---|
| Data | Real MNIST/Fashion-MNIST/Overhead-MNIST | Random dummy tensors | No meaningful loss/accuracy |
| Epochs | 50 | 1 step | Cannot reproduce training curves |
| Batch size | 32 | 4 (demo) | Cannot reproduce throughput |
| Subsampling | 1,000 samples/class balanced | N/A | Cannot reproduce paper split |
| Random seed | 42 | Not set | Non-reproducible |
| PennyLane | Required and used | Must be installed | Runs only if PennyLane installed |

---

### 8.4 Evaluation Differences

**Paper reports:**
1. Training accuracy curves for 7 models x 3 datasets over 50 epochs (Figures 6-8).
2. Noise robustness tables: 4 noise types x 3 noise levels (Tables 5-8).
3. Circuit expressibility/entanglement/discreteness comparison (Tables 2-3).

**Current code reports:**
- Parameter counts at startup. OK
- Output shape (logits.shape). OK
- Single loss value for one dummy batch. OK
- No accuracy metrics. MISSING
- No multi-epoch training curves. MISSING
- No noise experiments. MISSING
- No baseline model comparison. MISSING

---

### 8.5 Noise Simulator Differences

| Aspect | Paper (Experiment 3) | Current Code |
|---|---|---|
| Simulator | default.mixed (mixed-state) | default.qubit (pure-state) |
| Noise channels | Bit-flip, Phase-flip, Depolarizing | None |
| Batch size | 100 | 4 (demo) |
| Noise probabilities | p in {0.1, 0.2, 0.3} | Not applicable |

To enable noise experiments:

```python
# Change device
dev = qml.device("default.mixed", wires=4)

# Add noise gates before measurement in quantum_circuit()
for i in range(num_qubits):
    qml.BitFlip(p, wires=i)     # or PhaseFlip / DepolarizingChannel
return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
```

---

### 8.6 What Is Identical to the Paper

1. **PQC architecture (Circuit 11):** Gate sequence, topology, parameter count (16),
   angle encoding (H + RY(x*pi)), Pauli-Z measurement.
2. **Classical branch:** Conv2d spec (4x4, stride 2, pad 1, 8 filters = 136 params).
3. **Tensor flow:** Parallel branches -> channel concat -> flatten -> 3-layer dense head.
4. **Dense head:** 2352 -> 128 -> 64 -> C with ReLU.
5. **Optimizer:** Adam with lr=0.01.
6. **Loss function:** CrossEntropyLoss.
7. **Gradient method:** PennyLane interface="torch" uses parameter-shift rule internally.
8. **Measurement:** 4 Pauli-Z expectations = 4 quantum output channels per patch.
9. **Weight sharing:** Same PQC parameters shared across all 196 patches per image.

---

## 9. Implementation File Map

```
QC-CNN-Parallel1/
|
+-- qc-cnn-parallel.py              <- Main model (architecture smoke test)
|     quantum_circuit()             <- PQC QNode (Circuit 11)
|     QuantumConvLayer              <- 2x2 quantum sliding window
|     QCCNNParallel                 <- Full parallel model
|
+-- implementation/
|   +-- models/
|   |   +-- qc_cnn_parallel.py      <- Extended model (TorchLayer-based)
|   |   +-- quantum_circuit.py      <- Standalone PQC definitions
|   |   +-- scalable_quantum_circuit.py  <- Vectorized/optimized variant
|   |   +-- ablation_models.py      <- Models for ablation experiments
|   +-- run_all.py                  <- Full experiment runner
|   +-- requirements.txt            <- Python dependencies
|   +-- datasets/, training/, utils/, results/   <- Experiment infrastructure
|
+-- ARCHITECTURE.md                 <- Architecture reference (paper-verified)
+-- METHODOLOGY.md                  <- Full math derivation (paper-verified)
+-- EXPERIMENT_SETUP.md             <- Hardware/training setup (paper-verified)
+-- RESULTS.md                      <- Paper results + reproduction template
+-- DATASETS.md                     <- Dataset documentation
+-- cur_imple.md                    <- This file
+-- Quantum Engineering .pdf        <- Source paper (Liu & Lou, 2026)
```

---

## 10. Reproduction Checklist

To fully reproduce paper Experiment 2 results:

- [ ] Install: pip install torch pennylane torchvision numpy
- [ ] Set seeds: torch.manual_seed(42), random.seed(42), np.random.seed(42)
- [ ] Load MNIST via torchvision with ToTensor() transform (normalizes to [0,1])
- [ ] Subsample balanced: 1,000 samples/class training, 200/class testing
- [ ] Batch size = 32
- [ ] Initialize: model = QCCNNParallel(num_classes=10)
- [ ] Optimizer: Adam(model.parameters(), lr=0.01)
- [ ] Train for 50 epochs with per-epoch validation accuracy/loss tracking
- [ ] Report final test accuracy (paper reports 0.9005 on MNIST)
- [ ] Repeat with Fashion-MNIST and Overhead-MNIST
- [ ] For noise experiments: switch to default.mixed, add noise channels, batch=100

> **Performance warning:** The quantum convolution triple loop makes 6,272 QNode calls
> per forward pass at batch=32. Parameter-shift gradients require 2x more evaluations
> (12,544 per backward pass). Training 50 epochs on 10,000 MNIST samples will be slow
> on CPU. Use scalable_quantum_circuit.py for optimized evaluation or reduce batch size.
