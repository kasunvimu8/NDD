import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)
from tqdm import tqdm
import numpy as np

##############################################################################
#   Test Contrastive Siamese Network                                   #
##############################################################################

def test_model_contrastive(model, dataloader, device, threshold=0.5):
    model.eval()
    y_true, y_pred, prob = [], [], []
    if len(dataloader) == 0:
        return None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            if batch is None:
                continue
            emb1 = batch['embeddings1'].to(device)
            emb2 = batch['embeddings2'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(emb1, emb2)
            logits = outputs['logits'].squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # F1 by class if needed
    f1_vals = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Class 0": f1_vals[0] if len(f1_vals) > 0 else 0.0,
        "F1_Class 1": f1_vals[1] if len(f1_vals) > 1 else 0.0,
        "F1 Score (Weighted Avg)": f1_weighted,
    }
    return metrics


##############################################################################
#    Test Triplet Siamese Network                                      #
##############################################################################

def test_model_triplet(model, test_loader, device, threshold=0.5):
    model.eval()
    y_true, y_pred = [], []

    if len(test_loader) == 0:
        return None

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            emb1 = batch["embeddings1"].to(device)
            emb2 = batch["embeddings2"].to(device)
            labels = batch["labels"].cpu().numpy()

            out1 = model.forward_once(emb1)
            out2 = model.forward_once(emb2)

            distances = F.pairwise_distance(out1, out2)
            preds = (distances <= threshold).long().cpu().numpy()

            y_true.extend(labels)
            y_pred.extend(preds)

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_vals = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics_out = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Class 0": f1_vals[0] if len(f1_vals) > 0 else 0.0,
        "F1_Class 1": f1_vals[1] if len(f1_vals) > 1 else 0.0,
        "F1 Score (Weighted Avg)": f1_weighted,
    }
    return metrics_out
