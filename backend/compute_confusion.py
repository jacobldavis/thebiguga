import itertools
import os
import numpy as np
import matplotlib.pyplot as plt

from speechbrain.inference.speaker import SpeakerRecognition
import huggingface_hub


def label_from_filename(fname):
    # expects names like recording_t1.wav -> label 't'
    base = os.path.basename(fname)
    parts = base.split("_")
    if len(parts) >= 2:
        token = parts[1]
        return token[0]
    return base


def main():
    files = [
        "recording_t1.wav",
        "recording_t2.wav",
        "recording_r1.wav",
        "recording_r2.wav",
        "recording_j1.wav",
        "recording_j2.wav",
    ]

    # download model snapshot (cached by huggingface)
    model_path = huggingface_hub.snapshot_download("speechbrain/spkrec-ecapa-voxceleb")

    verification = SpeakerRecognition.from_hparams(
        source=model_path,
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
    )

    truths = []
    preds = []
    scores = []
    pairs = []

    def to_float(x):
        try:
            return float(x.item())
        except Exception:
            return float(x)

    def to_bool(x):
        try:
            return bool(x.item())
        except Exception:
            return bool(x)

    for a, b in itertools.combinations(files, 2):
        score, prediction = verification.verify_files(a, b)
        lbl_a = label_from_filename(a)
        lbl_b = label_from_filename(b)
        true_same = (lbl_a == lbl_b)

        score_f = to_float(score)
        pred_b = to_bool(prediction)

        truths.append(bool(true_same))
        preds.append(pred_b)
        scores.append(score_f)
        pairs.append((a, b))
        print(f"{a} vs {b}: score={score_f:.4f}, pred={pred_b}, true_same={true_same}")

    # compute confusion counts
    TP = sum(1 for t, p in zip(truths, preds) if t and p)
    FN = sum(1 for t, p in zip(truths, preds) if t and not p)
    FP = sum(1 for t, p in zip(truths, preds) if not t and p)
    TN = sum(1 for t, p in zip(truths, preds) if not t and not p)

    cm = np.array([[TP, FN], [FP, TN]])

    print("\nConfusion matrix (rows=true [same,diff], cols=pred [same,diff]):")
    print(cm)

    # plot
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred_same", "pred_diff"])
    ax.set_yticklabels(["true_same", "true_diff"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    plt.tight_layout()
    outpng = "confusion_matrix.png"
    plt.savefig(outpng)
    print(f"Saved confusion matrix image to {outpng}")


if __name__ == "__main__":
    main()