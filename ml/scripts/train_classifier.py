"""
Train and compare classifiers for the static ISL letters.

Input : data/processed/landmarks.csv   (from extract_landmarks.py)
Output: ml/results/model_comparison.csv   accuracy and F1 for each model
        ml/results/confusion_matrix.png   which letters get confused (best model)
        web/model/letter_classifier.onnx  best model, ready for the browser
        web/model/labels.json             letter names in the model's output order

Run from the project folder:
    python ml/scripts/train_classifier.py
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ASPECT = 640 / 480          # ISL-HS videos are 640 x 480; x and y are stored as fractions of width / height


def normalise(points, aspect=ASPECT):
    """Turn raw landmarks (n, 21, 3) into position- and size-independent features (n, 63).

    1. Correct for the image shape, so x and y use the same unit.
    2. Move the wrist (point 0) to the origin, so the hand position in the image does not matter.
    3. Divide by the palm size (wrist to middle-finger base, point 9), so hand size and
       distance from the camera do not matter.
    """
    p = points.copy()
    p[:, :, 0] *= aspect
    p -= p[:, 0:1, :]
    palm = np.linalg.norm(p[:, 9, :2], axis=1)
    palm[palm == 0] = 1e-6
    return (p / palm[:, None, None]).reshape(len(p), -1).astype(np.float32)


def candidate_models():
    """The four lightweight classifiers to compare."""
    return {
        "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare classifiers for static ISL letters")
    parser.add_argument("--data", default="data/processed/landmarks.csv")
    parser.add_argument("--results", default="ml/results")
    parser.add_argument("--web-model", default="web/model")
    args = parser.parse_args()

    # ---- 1. Load the data (static letters only; J, X and Z are the optional extension) ----
    df = pd.read_csv(args.data)
    df = df[df["is_dynamic"] == 0].reset_index(drop=True)
    coord_cols = [f"{a}{i}" for i in range(21) for a in "xyz"]
    X = normalise(df[coord_cols].to_numpy(dtype=np.float32).reshape(-1, 21, 3))

    letters = sorted(df["letter"].unique())
    letter_to_id = {l: i for i, l in enumerate(letters)}
    y = df["letter"].map(letter_to_id).to_numpy()
    groups = df["person"].to_numpy()
    people = sorted(set(groups))
    print(f"{len(df)} frames, {len(letters)} letters, {len(people)} signers\n")

    # ---- 2. Leave-one-signer-out: train on 5 people, test on the 6th, repeat 6 times ----
    rows, all_preds = [], {}
    for name, model in candidate_models().items():
        accs, f1s, preds = [], [], np.zeros_like(y)
        start = time.time()
        for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
            model.fit(X[train_idx], y[train_idx])
            preds[test_idx] = model.predict(X[test_idx])
            accs.append(accuracy_score(y[test_idx], preds[test_idx]))
            f1s.append(f1_score(y[test_idx], preds[test_idx], average="macro"))

        # prediction time for a single frame, as in the live app
        t = time.perf_counter()
        for _ in range(200):
            model.predict(X[:1])
        ms_per_frame = (time.perf_counter() - t) / 200 * 1000

        all_preds[name] = preds
        rows.append({"model": name,
                     "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs),
                     "macro_f1_mean": np.mean(f1s), "macro_f1_std": np.std(f1s),
                     "ms_per_frame": ms_per_frame, "total_time_s": time.time() - start,
                     **{f"acc_{p}": a for p, a in zip(people, accs)}})
        print(f"{name:20s} accuracy {np.mean(accs):.1%} (+/- {np.std(accs):.1%})   "
              f"macro F1 {np.mean(f1s):.3f}   {ms_per_frame:.2f} ms/frame")

    results = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "model_comparison.csv", index=False)
    best = results.iloc[0]["model"]
    print(f"\nBest model by macro F1: {best}")

    # ---- 3. Confusion matrix for the best model ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = confusion_matrix(y, all_preds[best], normalize="true")
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(letters)), [l.upper() for l in letters])
        ax.set_yticks(range(len(letters)), [l.upper() for l in letters])
        ax.set_xlabel("Predicted letter")
        ax.set_ylabel("True letter")
        ax.set_title(f"{best}: leave-one-signer-out confusion matrix")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(results_dir / "confusion_matrix.png", dpi=150)
        print(f"Saved {results_dir / 'confusion_matrix.png'}")
    except Exception as err:
        print(f"Could not draw the confusion matrix: {err}")

    # most confused pairs, useful for the feedback rules later
    cm_counts = confusion_matrix(y, all_preds[best])
    np.fill_diagonal(cm_counts, 0)
    pairs = sorted(((cm_counts[i, j], letters[i], letters[j]) for i in range(len(letters))
                    for j in range(len(letters)) if cm_counts[i, j] > 0), reverse=True)[:5]
    print("Most common confusions (true -> predicted):")
    for count, a, b in pairs:
        print(f"  {a.upper()} -> {b.upper()}: {count} frames")

    # ---- 4. Train the best model on all six signers and export it for the browser ----
    final = candidate_models()[best]
    if best == "SVM":
        final.set_params(svc__probability=True)          # needed for confidence scores in the app
    final.fit(X, y)

    web_dir = Path(args.web_model)
    web_dir.mkdir(parents=True, exist_ok=True)
    try:
        from skl2onnx import to_onnx
        onnx_model = to_onnx(final, X[:1], options={"zipmap": False}, target_opset=15)
        (web_dir / "letter_classifier.onnx").write_bytes(onnx_model.SerializeToString())
        print(f"\nSaved {web_dir / 'letter_classifier.onnx'}")
    except Exception as err:
        print(f"\nONNX export failed: {err}")
    (web_dir / "labels.json").write_text(json.dumps({
        "model": best,
        "letters": letters,
        "feature": "landmarks scaled by aspect, wrist at origin, divided by wrist-to-point-9 distance",
    }, indent=2))
    print(f"Saved {web_dir / 'labels.json'}")


if __name__ == "__main__":
    main()