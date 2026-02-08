"""
Train optimal feature weights for speaker+passphrase verification.

Scans all WAV files in ../wavs/, groups them by base name (e.g. carl1-4 = group "carl"),
then trains a PyTorch model to learn weights such that:
  - Same-group pairs → score ≈ 1
  - Cross-group pairs → score ≈ 0

Uses GPU if available. Results are printed and can be pasted back into mfcc.py.
"""

import os
import re
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Import feature extraction from mfcc.py
sys.path.insert(0, os.path.dirname(__file__))
from mfcc import build_embeddings, cosine_sim

# ─── Constants ───────────────────────────────────────────────────────────────

WAVS_DIR = os.path.join(os.path.dirname(__file__), "..", "wavs")
EPS = 1e-8

FEATURE_KEYS = [
    "voice_shape", "pitch_avg", "pitch_std", "brightness",
    "delta_mfcc", "delta2_mfcc", "spectral_flux", "onset_rhythm",
    "mfcc_trajectory", "jitter", "shimmer", "hnr",
    "spectral_rolloff", "zcr", "spectral_contrast", "chroma"
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_group(filename):
    """Extract group label from filename.
    carl1.wav → 'carl', comm1_1.wav → 'comm1', test1_3.wav → 'test1'"""
    base = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r'_?\d+$', '', base)


def compute_similarity(v1, v2):
    """Similarity between two feature values (scalar or vector) → [0, 1]."""
    if np.isscalar(v1) or (isinstance(v1, np.ndarray) and v1.ndim == 0):
        diff = abs(float(v1) - float(v2))
        denom = max(abs(float(v1)), abs(float(v2)), 1.0)
        return max(0.0, 1.0 - diff / denom)
    else:
        return cosine_sim(v1, v2)


# ─── Model ───────────────────────────────────────────────────────────────────

class SpeakerVerificationNet(nn.Module):
    """Lean model for speaker verification.
    ~145 params to match the small dataset (66 positive, 880 negative pairs).
    Input: 16 per-feature similarities -> learned weights -> 1 hidden layer -> output."""
    def __init__(self, n_features):
        super().__init__()
        
        # Learnable feature importance (no sigmoid gating — direct soft weights)
        self.feature_weights = nn.Parameter(torch.zeros(n_features))
        
        # Single hidden layer: 16 -> 8 -> 1  (~145 params total)
        self.fc1 = nn.Linear(n_features, 8)
        self.fc2 = nn.Linear(8, 1)
        
        # Dropout for the one hidden layer
        self.dropout = nn.Dropout(0.15)
    
    def forward(self, x):
        # x: (batch, n_features) per-feature similarities
        
        # Soft feature weighting via softmax (sums to 1, no collapse to 0.5)
        w = torch.softmax(self.feature_weights, dim=0)
        x = x * w
        
        # Hidden layer
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        # Output
        x = self.fc2(x)
        return torch.sigmoid(x).squeeze(-1)
    
    def get_feature_importance(self):
        """Extract learned feature weights (softmax-normalized)."""
        return torch.softmax(self.feature_weights, dim=0).detach().cpu().numpy()


# ─── Main training loop ─────────────────────────────────────────────────────

def discover_files():
    """Find all WAV files in wavs/ and group them."""
    pattern = os.path.join(WAVS_DIR, "*.wav")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No WAV files found in {WAVS_DIR}")

    groups = {}
    for f in files:
        g = get_group(f)
        groups.setdefault(g, []).append(f)

    return files, groups


def extract_all_embeddings(files):
    """Build embeddings for every file. Returns dict: filepath → embeddings."""
    embeddings = {}
    for i, f in enumerate(files):
        name = os.path.basename(f)
        print(f"  [{i+1}/{len(files)}] {name}")
        embeddings[f] = build_embeddings(f)
    return embeddings


def build_pairs(files, groups):
    """Build all unique pairs with labels (1 = same group, 0 = different)."""
    file_to_group = {}
    for g, members in groups.items():
        for f in members:
            file_to_group[f] = g

    pairs = []
    labels = []
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            same = file_to_group[files[i]] == file_to_group[files[j]]
            pairs.append((files[i], files[j]))
            labels.append(1.0 if same else 0.0)

    return pairs, labels


def build_similarity_matrix(pairs, embeddings):
    """Compute per-feature similarity for every pair → (N_pairs, N_features) array."""
    X = []
    for a, b in pairs:
        row = []
        for key in FEATURE_KEYS:
            v1, _ = embeddings[a][key]
            v2, _ = embeddings[b][key]
            row.append(compute_similarity(v1, v2))
        X.append(row)
    return np.array(X, dtype=np.float32)


def train(X_base, y_base, n_epochs=3000):
    """Train lean model for speaker verification. Returns trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}", flush=True)

    n_same = int(y_base.sum())
    n_diff = len(y_base) - n_same
    print(f"Pairs: {n_same} same-group, {n_diff} cross-group, {len(y_base)} total", flush=True)

    # Moderate augmentation
    rng = np.random.RandomState(42)
    N_AUG = 50
    X_all = [X_base]
    y_all = [y_base]
    for _ in range(N_AUG):
        noise = rng.normal(0, 0.015, X_base.shape).astype(np.float32)
        X_all.append(np.clip(X_base + noise, 0, 1))
        y_all.append(y_base)

    X_np = np.vstack(X_all)
    y_np = np.concatenate(y_all)
    print(f"Augmented: {len(y_np)} samples ({N_AUG}x)", flush=True)

    # Split train/val on UN-AUGMENTED indices to prevent leakage
    n_base = len(y_base)
    base_idx = rng.permutation(n_base)
    n_val_base = max(int(0.2 * n_base), 1)
    val_base_idx = set(base_idx[:n_val_base])
    
    train_rows = []
    val_rows = []
    for aug_round in range(N_AUG + 1):
        offset = aug_round * n_base
        for i in range(n_base):
            if i in val_base_idx:
                val_rows.append(offset + i)
            else:
                train_rows.append(offset + i)
    
    X_train = torch.from_numpy(X_np[train_rows]).to(device)
    y_train = torch.from_numpy(y_np[train_rows]).to(device)
    X_val = torch.from_numpy(X_np[val_rows]).to(device)
    y_val = torch.from_numpy(y_np[val_rows]).to(device)
    
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)} (split on base pairs, no leakage)", flush=True)

    # Model
    model = SpeakerVerificationNet(len(FEATURE_KEYS)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params}", flush=True)
    
    # MILD class weighting — just 3x, not 13x.
    # 13x made the model predict high for everything to avoid FN penalty.
    pos_weight = torch.tensor(3.0, device=device)
    criterion = nn.BCELoss(reduction='none')
    
    # Margin for hinge loss: cross-group scores must be below this
    MARGIN = 0.35
    HINGE_WEIGHT = 2.0  # how much to penalize FPs above margin
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=2, eta_min=1e-5
    )

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    loss_history = []
    val_loss_history = []

    print(f"\nTraining for up to {n_epochs} epochs (early stop patience=200)...", flush=True)
    print(f"pos_weight={pos_weight.item():.1f}, margin={MARGIN}, hinge_weight={HINGE_WEIGHT}", flush=True)
    print("-" * 72, flush=True)

    for epoch in range(n_epochs):
        # Training
        model.train()
        pred_train = model(X_train)
        
        # BCE loss with mild class weighting
        loss_per_sample = criterion(pred_train, y_train)
        sample_weights = torch.where(y_train == 1.0, pos_weight, torch.ones_like(y_train))
        bce_loss = (loss_per_sample * sample_weights).mean()
        
        # Hinge loss: penalize cross-group predictions ABOVE margin
        # This directly pushes false positives down below the threshold
        neg_mask = (y_train == 0.0)
        neg_preds = pred_train[neg_mask]
        hinge = torch.relu(neg_preds - MARGIN).mean() * HINGE_WEIGHT
        
        # Also penalize same-group predictions BELOW (1 - margin)
        pos_mask = (y_train == 1.0)
        pos_preds = pred_train[pos_mask]
        hinge_pos = torch.relu((1.0 - MARGIN) - pos_preds).mean() * HINGE_WEIGHT
        
        total_loss = bce_loss + hinge + hinge_pos
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step(epoch)
        
        # Validation
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            loss_val_per_sample = criterion(pred_val, y_val)
            sample_weights_val = torch.where(y_val == 1.0, pos_weight, torch.ones_like(y_val))
            bce_val = (loss_val_per_sample * sample_weights_val).mean()
            
            neg_mask_v = (y_val == 0.0)
            hinge_val = torch.relu(pred_val[neg_mask_v] - MARGIN).mean() * HINGE_WEIGHT
            pos_mask_v = (y_val == 1.0)
            hinge_pos_val = torch.relu((1.0 - MARGIN) - pred_val[pos_mask_v]).mean() * HINGE_WEIGHT
            loss_val = bce_val + hinge_val + hinge_pos_val
            
            pred_labels_val = (pred_val >= 0.5).float()
            acc_val = (pred_labels_val == y_val).float().mean().item()
            
            tp = ((pred_labels_val == 1) & (y_val == 1)).sum().item()
            fp = ((pred_labels_val == 1) & (y_val == 0)).sum().item()
            fn = ((pred_labels_val == 0) & (y_val == 1)).sum().item()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        loss_history.append(total_loss.item())
        val_loss_history.append(loss_val.item())
        
        # Early stopping on val loss
        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 100 == 0:
            # Also report mean neg/pos scores for monitoring
            neg_mean = pred_val[neg_mask_v].mean().item() if neg_mask_v.any() else 0
            pos_mean = pred_val[pos_mask_v].mean().item() if pos_mask_v.any() else 0
            print(f"  Epoch {epoch+1:5d}  loss={total_loss.item():.4f}  "
                  f"val={loss_val.item():.4f}  acc={acc_val:.3f}  "
                  f"P={prec:.3f} R={recall:.3f}  "
                  f"neg_avg={neg_mean:.3f} pos_avg={pos_mean:.3f}", flush=True)
        
        if patience_counter >= 200:
            print(f"  Early stopping at epoch {epoch+1}", flush=True)
            break
    
    # Restore best model
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    print(f"\nBest validation loss: {best_val_loss:.8f}")
    
    return model, loss_history, val_loss_history


def evaluate(model, pairs, labels, X_base, groups):
    """Print detailed evaluation of trained model."""
    device = next(model.parameters()).device
    
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    # Get feature importance
    feature_importance = model.get_feature_importance()
    print("\nLearned feature importance (sorted by magnitude):")
    importance_items = [(FEATURE_KEYS[i], feature_importance[i]) for i in range(len(FEATURE_KEYS))]
    for k, v in sorted(importance_items, key=lambda x: -x[1]):
        bar = '█' * int(v * 100)
        print(f"  {k:20s}: {v:.6f}  {bar}")

    # Compute scores with the trained model
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_base).to(device)
        scores = model(X_tensor).cpu().numpy()

    same_scores = scores[np.array(labels) == 1.0]
    diff_scores = scores[np.array(labels) == 0.0]

    # Compute loss metrics
    targets = np.array(labels)
    bce_loss = -np.mean(targets * np.log(scores + 1e-8) + (1 - targets) * np.log(1 - scores + 1e-8))
    same_bce = -np.mean(np.log(same_scores + 1e-8))
    diff_bce = -np.mean(np.log(1 - diff_scores + 1e-8))
    
    print(f"\nLoss Metrics:")
    print(f"  Overall BCE:        {bce_loss:.6f}")
    print(f"  Same-group BCE:     {same_bce:.6f}  (target=1.0)")
    print(f"  Cross-group BCE:    {diff_bce:.6f}  (target=0.0)")

    print(f"\nSame-group scores:   mean={np.mean(same_scores):.4f}  "
          f"min={np.min(same_scores):.4f}  max={np.max(same_scores):.4f}  "
          f"std={np.std(same_scores):.4f}")
    print(f"Cross-group scores:  mean={np.mean(diff_scores):.4f}  "
          f"min={np.min(diff_scores):.4f}  max={np.max(diff_scores):.4f}  "
          f"std={np.std(diff_scores):.4f}")
    print(f"Gap (same - cross):  {np.mean(same_scores) - np.mean(diff_scores):.4f}")

    # Threshold analysis with optimal threshold finding
    print("\nThreshold analysis:")
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in np.linspace(0.1, 0.9, 17):
        tp = np.sum(same_scores >= thresh)
        fn = np.sum(same_scores < thresh)
        tn = np.sum(diff_scores < thresh)
        fp = np.sum(diff_scores >= thresh)
        acc = (tp + tn) / (tp + tn + fp + fn) * 100
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
        
        if thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            print(f"  T={thresh:.1f}  acc={acc:5.1f}%  prec={prec:5.1f}%  "
                  f"rec={recall:5.1f}%  F1={f1:5.1f}%  (TP={tp} FP={fp} FN={fn} TN={tn})")
    
    print(f"\n  Best threshold: {best_thresh:.3f} (F1={best_f1:.1f}%)")

    # ROC-AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(labels, scores)
        print(f"  ROC-AUC: {auc:.4f}")
    except:
        pass

    # Per-group breakdown
    print("\nPer-group average same-pair scores:")
    group_scores = {}
    for (a, b), lab, sc in zip(pairs, labels, scores):
        if lab == 1.0:
            g = get_group(a)
            group_scores.setdefault(g, []).append(sc)
    for g in sorted(group_scores.keys()):
        vals = group_scores[g]
        print(f"  {g:12s}: {np.mean(vals):.4f}  "
              f"(min={np.min(vals):.4f}  max={np.max(vals):.4f}  n={len(vals)})")

    # Worst same-group pairs (should be high but aren't)
    print("\nWorst same-group pairs (lowest scores, potential false negatives):")
    same_pairs_with_scores = [
        (os.path.basename(a), os.path.basename(b), sc)
        for (a, b), lab, sc in zip(pairs, labels, scores)
        if lab == 1.0
    ]
    same_pairs_with_scores.sort(key=lambda x: x[2])
    for name1, name2, sc in same_pairs_with_scores[:5]:
        print(f"  {name1} vs {name2}: {sc:.4f}")

    # Worst cross-group pairs (should be low but aren't == FALSE POSITIVES)
    print("\nWorst cross-group pairs (highest scores, FALSE POSITIVES):")
    diff_pairs_with_scores = [
        (os.path.basename(a), os.path.basename(b), sc)
        for (a, b), lab, sc in zip(pairs, labels, scores)
        if lab == 0.0
    ]
    diff_pairs_with_scores.sort(key=lambda x: -x[2])
    for name1, name2, sc in diff_pairs_with_scores[:10]:
        print(f"  {name1} vs {name2}: {sc:.4f}")


def plot_results(model, loss_history, val_loss_history, pairs, labels, X_base, groups):
    """Visualize feature importance, loss curves, and score distributions."""
    device = next(model.parameters()).device
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_base).to(device)
        scores = model(X_tensor).cpu().numpy()
    
    feature_importance = model.get_feature_importance()
    importance_dict = {FEATURE_KEYS[i]: feature_importance[i] for i in range(len(FEATURE_KEYS))}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Feature importance bar chart
    ax = axes[0, 0]
    sorted_items = sorted(importance_dict.items(), key=lambda x: -x[1])
    names_sorted = [k for k, _ in sorted_items]
    vals_sorted = [v for _, v in sorted_items]
    ax.barh(range(len(names_sorted)), vals_sorted, color='steelblue')
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Learned Feature Importance Weights')
    ax.invert_yaxis()

    # 2. Loss curves
    ax = axes[0, 1]
    ax.plot(loss_history, label='Train Loss', linewidth=1, alpha=0.7)
    ax.plot(val_loss_history, label='Val Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Score histogram
    ax = axes[1, 0]
    same_scores = scores[np.array(labels) == 1.0]
    diff_scores = scores[np.array(labels) == 0.0]
    ax.hist(diff_scores, bins=30, alpha=0.7, label='Cross-group', color='red')
    ax.hist(same_scores, bins=30, alpha=0.7, label='Same-group', color='green')
    ax.axvline(0.5, color='black', linestyle='--', label='Threshold=0.5')
    ax.set_xlabel('Score')
    ax.set_ylabel('Count')
    ax.set_title('Score Distribution')
    ax.legend()

    # 4. Per-group boxplot
    ax = axes[1, 1]
    group_data = {}
    for (a, b), lab, sc in zip(pairs, labels, scores):
        if lab == 1.0:
            g = get_group(a)
            group_data.setdefault(g, []).append(sc)
    group_names = sorted(group_data.keys())
    box_data = [group_data[g] for g in group_names]
    ax.boxplot(box_data, labels=group_names, vert=True)
    ax.set_ylabel('Same-group score')
    ax.set_title('Per-Group Same-Pair Scores')
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)

    plt.suptitle('Speaker Verification Training Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'weight_training.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✓ Plot saved to weight_training.png")


def print_mfcc_update(model):
    """Print feature importance and save model for deployment."""
    feature_importance = model.get_feature_importance()
    importance_dict = {FEATURE_KEYS[i]: feature_importance[i] for i in range(len(FEATURE_KEYS))}
    
    print("\n" + "=" * 72)
    print("DEPLOYMENT OPTIONS:")
    print("=" * 72)
    
    # Option 1: Save full model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'speaker_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_importance': importance_dict
    }, model_path)
    print(f"\n✓ Full model saved to: {model_path}")
    print("  Load with: torch.load('speaker_model.pth')")
    
    # Option 2: Feature importance for mfcc.py
    print("\n✓ COPY-PASTE INTO mfcc.py build_embeddings() return dict:")
    print("-" * 72)
    max_key = max(len(k) for k in FEATURE_KEYS)
    for k in FEATURE_KEYS:
        w = importance_dict[k]
        print(f'        "{k}":{" " * (max_key - len(k))}  (... , {w:.4f}),')
    print("-" * 72)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  FEATURE WEIGHT OPTIMIZER")
    print("  Training on all WAV files in wavs/")
    print("=" * 72)

    # 1. Discover files
    print("\n[1/4] Discovering files...")
    files, groups = discover_files()
    print(f"  Found {len(files)} files in {len(groups)} groups:")
    for g in sorted(groups.keys()):
        members = [os.path.basename(f) for f in groups[g]]
        print(f"    {g:12s}: {', '.join(members)}")

    # 2. Extract embeddings
    print(f"\n[2/4] Extracting embeddings for {len(files)} files...")
    embeddings = extract_all_embeddings(files)

    # 3. Build pairs
    print(f"\n[3/4] Building pairs...")
    pairs, labels = build_pairs(files, groups)

    # Build similarity matrix
    print(f"  Computing per-feature similarities for {len(pairs)} pairs...")
    X_base = build_similarity_matrix(pairs, embeddings)
    y_base = np.array(labels, dtype=np.float32)

    # 4. Train
    print(f"\n[4/4] Training deep neural network...")
    model, loss_history, val_loss_history = train(X_base, y_base)

    # Evaluate
    evaluate(model, pairs, labels, X_base, groups)

    # Print update instructions
    print_mfcc_update(model)

    # Plot
    plot_results(model, loss_history, val_loss_history, pairs, labels, X_base, groups)


if __name__ == "__main__":
    main()
