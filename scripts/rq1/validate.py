import torch
import torch.nn.functional as F

##############################################################################
#   Validation Contrastive Siamese Network                                   #
##############################################################################

def validate_model_contrastive(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    if len(dataloader) == 0:
        return 0.0

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            emb1 = batch['embeddings1'].to(device)
            emb2 = batch['embeddings2'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(emb1, emb2, labels)
            total_loss += outputs['loss'].item()

    return total_loss / len(dataloader)


##############################################################################
#    Validation Triplet Siamese Network                                      #
##############################################################################

def validate_model_triplet(model, val_loader, device, threshold=0.5):
    model.eval()
    total_misclassified = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            emb1 = batch["embeddings1"].to(device)
            emb2 = batch["embeddings2"].to(device)
            labels = batch["labels"].to(device).float()

            out1 = model.forward_once(emb1)
            out2 = model.forward_once(emb2)

            distances = F.pairwise_distance(out1, out2)
            preds = (distances <= threshold).float()

            mismatches = (preds != labels).float().sum().item()
            total_misclassified += mismatches
            total_samples += len(labels)

    if total_samples == 0:
        return 0.0

    val_loss = total_misclassified / total_samples
    return val_loss

