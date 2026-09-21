"""
scalable_quantum_circuit.py
============================
Configurable quantum circuit with variable qubit count and depth.

This is a SEPARATE file from quantum_circuit.py. It does NOT modify the
original Circuit 11 implementation. Instead, it provides a generalized
version that allows Experiment 5 to sweep across different configurations.

Experiment 5 configuration
---------------------------
  Part A — Qubit sweep   : n_qubits in [2, 4, 6, 8],  n_layers = 2 (fixed)
  Part B — Depth sweep   : n_qubits = 4 (fixed),        n_layers in [1, 2, 3, 4, 5]

Paper baseline: 4 qubits / depth-2 (Circuit 11, 2 variational layers).

Parameters
----------
n_qubits : int  — number of qubits  (supported: 2, 4, 6, 8)
n_layers : int  — circuit depth / number of variational layers (1–5)

Each variational layer = one RY rotation block + one CRX entangling ring.
Total trainable PQC parameters = n_qubits * 2 * n_layers
  (n_qubits RY angles  +  n_qubits CRX angles, per layer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np


# ---------------------------------------------------------------------------
# 1. Scalable QNode factory
# ---------------------------------------------------------------------------
def make_scalable_circuit(n_qubits: int = 4, n_layers: int = 2):
    """
    Create a QNode with configurable qubit count and depth.

    The circuit follows the Circuit-11 pattern (RY rotations + CRX entangling
    in a ring / circle topology) but generalised to arbitrary qubit counts and
    depths.

    Circuit structure per layer
    ---------------------------
      1. Angle encoding : Hadamard + RY(inputs[i])  for each qubit i
         (encoding is applied only once, before the variational layers)
      2. For each variational layer l in [0, n_layers):
           a. RY(weights[idx])   rotation gate on every qubit
           b. CRX(weights[idx])  entangling ring  ctrl=(i+l) % n_qubits
                                                  target=(i+l+1) % n_qubits

    Parameter count
    ---------------
      n_params = n_qubits * 2 * n_layers
        (n_qubits RY + n_qubits CRX, per layer)

    Parameters
    ----------
    n_qubits : int  — number of qubits (determines patch size & output channels)
    n_layers : int  — number of variational layers (circuit depth)

    Returns
    -------
    qnode    : callable PennyLane QNode (interface="torch")
    n_params : int — total number of trainable PQC parameters
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    # Each layer contributes n_qubits RY params + n_qubits CRX params
    n_params = n_qubits * 2 * n_layers

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        """
        Scalable PQC: angle encoding then n_layers × (RY + CRX ring).

        inputs  : shape (n_qubits,) — patch pixels scaled by π
        weights : shape (n_params,) — trainable parameters
        """
        # --- Angle encoding (paper Eq. 6) ---
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RY(inputs[i], wires=i)

        # --- Variational layers ---
        idx = 0
        for layer in range(n_layers):
            # (a) RY rotation block
            for i in range(n_qubits):
                qml.RY(weights[idx], wires=i)
                idx += 1

            # (b) CRX entangling ring (shift by layer for richer entanglement)
            for i in range(n_qubits):
                ctrl   = (i + layer) % n_qubits
                target = (i + layer + 1) % n_qubits
                qml.CRX(weights[idx], wires=[ctrl, target])
                idx += 1

        # Measure all qubits in the Pauli-Z basis
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit, n_params


# ---------------------------------------------------------------------------
# 2. Scalable QuantumConvLayer
# ---------------------------------------------------------------------------
class ScalableQuantumConvLayer(nn.Module):
    """
    Quantum convolutional layer with configurable qubit count and depth.

    Unlike the original QuantumConvLayer (fixed 4 qubits, 2×2 window),
    this supports arbitrary patch sizes governed by the qubit count:

      Qubits | Patch shape | Stride (h, w) | Output channels
      -------|-------------|---------------|----------------
        2    |  1 × 2      |  (1, 2)       |  2
        4    |  2 × 2      |  (2, 2)       |  4   ← paper default
        6    |  2 × 3      |  (2, 3)       |  6
        8    |  2 × 4      |  (2, 4)       |  8

    The layer applies the circuit to every non-overlapping patch and
    returns a tensor whose channel dimension equals n_qubits (one
    PauliZ expectation value per qubit per patch).
    """

    # Supported patch configurations keyed by qubit count
    PATCH_CONFIG = {
        2: {"patch_h": 1, "patch_w": 2, "stride_h": 1, "stride_w": 2},
        4: {"patch_h": 2, "patch_w": 2, "stride_h": 2, "stride_w": 2},
        6: {"patch_h": 2, "patch_w": 3, "stride_h": 2, "stride_w": 3},
        8: {"patch_h": 2, "patch_w": 4, "stride_h": 2, "stride_w": 4},
    }

    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        """
        Parameters
        ----------
        n_qubits : int — number of qubits; must be in {2, 4, 6, 8}
        n_layers : int — number of variational layers (circuit depth)
        """
        super().__init__()
        if n_qubits not in self.PATCH_CONFIG:
            raise ValueError(
                f"Unsupported n_qubits={n_qubits}. "
                f"Supported values: {sorted(self.PATCH_CONFIG.keys())}"
            )

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Build the scalable QNode and register PQC weights as a Parameter
        self._qnode, n_params = make_scalable_circuit(n_qubits, n_layers)
        self.weights = nn.Parameter(torch.randn(n_params) * 0.1)

        # Spatial patch / stride configuration
        cfg           = self.PATCH_CONFIG[n_qubits]
        self.patch_h  = cfg["patch_h"]
        self.patch_w  = cfg["patch_w"]
        self.stride_h = cfg["stride_h"]
        self.stride_w = cfg["stride_w"]

    def get_output_shape(self, H: int = 28, W: int = 28):
        """
        Compute output spatial dimensions for a given input height and width.

        Returns
        -------
        (out_H, out_W) : tuple[int, int]
        """
        out_H = H // self.stride_h
        out_W = W // self.stride_w
        return out_H, out_W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the scalable quantum conv to every non-overlapping patch.

        Parameters
        ----------
        x : torch.Tensor  shape [B, 1, H, W]  — single-channel input image

        Returns
        -------
        out : torch.Tensor  shape [B, n_qubits, out_H, out_W]
            One PauliZ expectation value per qubit per spatial position.
        """
        B, C, H, W = x.shape
        out_H, out_W = self.get_output_shape(H, W)
        out = torch.zeros(B, self.n_qubits, out_H, out_W,
                          device=x.device, dtype=x.dtype)

        for b in range(B):
            for i in range(out_H):
                for j in range(out_W):
                    # Extract and flatten the spatial patch
                    r0    = i * self.stride_h
                    c0    = j * self.stride_w
                    patch = x[b, 0,
                              r0:r0 + self.patch_h,
                              c0:c0 + self.patch_w].flatten()

                    # Zero-pad if the patch is smaller than n_qubits
                    if patch.shape[0] < self.n_qubits:
                        pad   = torch.zeros(self.n_qubits - patch.shape[0],
                                            device=x.device, dtype=x.dtype)
                        patch = torch.cat([patch, pad])

                    # Angle encoding: scale pixel values by π  (paper Eq. 6)
                    patch  = patch * torch.pi

                    # Run circuit and collect PauliZ expectation values
                    result = self._qnode(patch, self.weights)
                    out[b, :, i, j] = torch.stack(
                        [r if isinstance(r, torch.Tensor) else torch.tensor(r)
                         for r in result]
                    )

        return out


# ---------------------------------------------------------------------------
# 3. Scalable QC-CNN-Parallel model
# ---------------------------------------------------------------------------
class ScalableQCCNNParallel(nn.Module):
    """
    QC-CNN-Parallel with configurable quantum circuit parameters.

    Architecture mirrors the original QCCNNParallel but allows varying:
      - n_qubits : number of qubits (controls quantum output channels)
      - n_layers : circuit depth    (controls expressibility)

    The classical branch (8 filters, 4×4 kernel, stride=2) is kept fixed
    at the paper's spec. When the quantum branch's spatial output differs
    from 14×14, an AdaptiveAvgPool2d is used to align dimensions before
    channel-wise concatenation.

    Experiment 5 sweep summary
    --------------------------
      Part A fixed depth  = 2   ; n_qubits ∈ {2, 4, 6, 8}
      Part B fixed qubits = 4   ; n_layers ∈ {1, 2, 3, 4, 5}

    FC head input size = (8 + n_qubits) × 14 × 14
    """

    def __init__(self, num_classes: int = 10,
                 n_qubits: int = 4, n_layers: int = 2):
        """
        Parameters
        ----------
        num_classes : int — number of output classes (10 for MNIST variants)
        n_qubits    : int — number of qubits; must be in {2, 4, 6, 8}
        n_layers    : int — circuit depth (variational layers)
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # --- Classical branch (same as original QCCNNParallel) ---
        # kernel=4×4, stride=2, padding=1 → [B, 8, 14, 14] for 28×28 input
        # Parameters: 8*(1*4*4)+8 = 136 (Table 4)
        self.classical_conv = nn.Conv2d(
            in_channels=1, out_channels=8,
            kernel_size=4, stride=2, padding=1
        )

        # --- Scalable quantum branch ---
        # Output: [B, n_qubits, out_H, out_W]
        self.quantum_conv = ScalableQuantumConvLayer(n_qubits, n_layers)

        # --- Spatial alignment ---
        # Classical output is always 14×14 for 28×28 input.
        # Quantum output depends on qubit-count-specific stride.
        q_out_H, q_out_W = self.quantum_conv.get_output_shape(28, 28)
        self._need_quantum_resize = (q_out_H != 14 or q_out_W != 14)
        if self._need_quantum_resize:
            self.quantum_pool = nn.AdaptiveAvgPool2d((14, 14))

        # --- FC head (Section 3.4) ---
        # Fused: [B, 8+n_qubits, 14, 14] → flatten → [B, (8+n_qubits)*196]
        fused_channels  = 8 + n_qubits
        fused_features  = fused_channels * 14 * 14   # 14*14 = 196
        self.fc1 = nn.Linear(fused_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape [B, 1, 28, 28], pixels in [0, 1]

        Returns
        -------
        logits : torch.Tensor  shape [B, num_classes]
        """
        # Classical branch
        x_class = F.relu(self.classical_conv(x))       # [B, 8,        14, 14]

        # Quantum branch
        x_quant = self.quantum_conv(x)                  # [B, n_qubits, q_H, q_W]
        if self._need_quantum_resize:
            x_quant = self.quantum_pool(x_quant)        # [B, n_qubits, 14, 14]

        # Channel-wise feature fusion
        x_fused = torch.cat([x_class, x_quant], dim=1) # [B, 8+n_qubits, 14, 14]
        x_flat  = x_fused.view(x_fused.size(0), -1)    # [B, (8+n_qubits)*196]

        # Dense classification head
        h1 = F.relu(self.fc1(x_flat))                  # [B, 128]
        h2 = F.relu(self.fc2(h1))                      # [B, 64]
        return self.fc3(h2)                             # [B, num_classes]

    def count_parameters(self):
        """
        Return a detailed dict of parameter counts per component.

        Keys match the style of QCCNNParallel.count_parameters() and are
        used directly by Experiment 5's result logging:
          - 'classical_conv' : parameters in the classical Conv2d branch
          - 'quantum_pqc'    : trainable PQC angles in the quantum branch
          - 'conv_total'     : classical_conv + quantum_pqc
          - 'fc1/fc2/fc3'    : FC head parameters
          - 'total'          : all parameters
          - 'n_qubits'       : qubit count (for sweep labelling)
          - 'n_layers'       : circuit depth (for sweep labelling)
        """
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            "classical_conv": n(self.classical_conv),
            "quantum_pqc":    n(self.quantum_conv),
            "conv_total":     n(self.classical_conv) + n(self.quantum_conv),
            "fc1":            n(self.fc1),
            "fc2":            n(self.fc2),
            "fc3":            n(self.fc3),
            "total":          n(self),
            "n_qubits":       self.n_qubits,
            "n_layers":       self.n_layers,
        }


# ---------------------------------------------------------------------------
# 4. Gradient variance measurement  (barren plateau detection)
# ---------------------------------------------------------------------------
def measure_gradient_variance(model, dataloader, device, n_batches: int = 5):
    """
    Measure gradient variance across batches to detect barren plateaus.

    A barren plateau manifests as near-zero gradient variance: the loss
    surface is exponentially flat, making gradient-based training fail.

      High variance  → healthy gradients → training will progress
      ~Zero variance → barren plateau   → training will stall

    The function focuses on the quantum branch's PQC weights because
    barren plateaus are a quantum-specific phenomenon.

    .. note::
        The *model* must already be on *device* before calling this
        function. Call ``model.to(device)`` before ``measure_gradient_variance``.

    Parameters
    ----------
    model      : ScalableQCCNNParallel — the model to probe
    dataloader : DataLoader            — training data loader
    device     : torch.device          — computation device (cpu or cuda)
    n_batches  : int                   — number of mini-batches to average over

    Returns
    -------
    dict with keys:
      'mean_grad_norm'  : float — mean L2 norm of gradients over batches
      'grad_variance'   : float — variance of per-batch gradient norms
      'max_grad'        : float — maximum observed gradient norm
      'min_grad'        : float — minimum observed gradient norm
    """
    model.train()
    loss_fn    = nn.CrossEntropyLoss()
    grad_norms = []

    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= n_batches:
            break
        images = images.to(device)
        labels = labels.to(device)

        model.zero_grad()
        logits = model(images)
        loss   = loss_fn(logits, labels)
        loss.backward()

        # Collect the gradient norms of quantum PQC parameters only.
        # If no quantum grad is found, fall back to all-parameter norms.
        quantum_grad_sq = 0.0
        found_quantum   = False
        for name, param in model.named_parameters():
            if param.grad is not None and "quantum_conv" in name:
                quantum_grad_sq += param.grad.data.norm(2).item() ** 2
                found_quantum    = True

        if not found_quantum:
            # Fallback: use all parameters
            for name, param in model.named_parameters():
                if param.grad is not None:
                    quantum_grad_sq += param.grad.data.norm(2).item() ** 2

        grad_norms.append(quantum_grad_sq ** 0.5)

    grad_norms = np.array(grad_norms, dtype=np.float64)
    return {
        "mean_grad_norm": float(grad_norms.mean()),
        "grad_variance":  float(grad_norms.var()),
        "max_grad":       float(grad_norms.max()),
        "min_grad":       float(grad_norms.min()),
    }


__all__ = [
    "make_scalable_circuit",
    "ScalableQuantumConvLayer",
    "ScalableQCCNNParallel",
    "measure_gradient_variance",
]
