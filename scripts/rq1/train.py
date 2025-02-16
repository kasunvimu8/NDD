from tqdm import tqdm
from scripts.utils.networks import TripletLoss


##############################################################################
#   Training Contrastive Siamese Network                                     #
##############################################################################

def train_one_epoch_contrastive(model, dataloader, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    if len(dataloader) == 0:
        return 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True):
        if batch is None:
            continue
        emb1 = batch['embeddings1'].to(device)
        emb2 = batch['embeddings2'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(emb1, emb2, labels)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


##############################################################################
#   Training Triplet Siamese Network                                         #
##############################################################################

def train_one_epoch_triplet(model, dataloader, optimizer, device, epoch, num_epochs, margin=1.0):
    """
    Training loop for triplet-based data.
    """
    model.train()
    criterion = TripletLoss(margin=margin)
    total_loss = 0.0

    if len(dataloader) == 0:
        return 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True):
        if batch is None:
            continue
        anchor = batch["anchor"].to(device)
        positive = batch["positive"].to(device)
        negative = batch["negative"].to(device)

        optimizer.zero_grad()
        anc_out, pos_out, neg_out = model(anchor, positive, negative)
        loss = criterion(anc_out, pos_out, neg_out)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
