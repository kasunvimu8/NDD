import random
from collections import defaultdict, Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE

##############################################################################
#   Pair, SMOTEDPairDataset Dataset Creation and Dataloaders Creation        #
##############################################################################

def pair_collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None

    emb1_list = [b['embeddings1'] for b in batch]
    emb2_list = [b['embeddings2'] for b in batch]
    labels_list = [b['labels'] for b in batch]

    emb1_stack = torch.stack(emb1_list, dim=0)
    emb2_stack = torch.stack(emb2_list, dim=0)
    labels_tensor = torch.stack(labels_list, dim=0)

    return {
        'embeddings1': emb1_stack,
        'embeddings2': emb2_stack,
        'labels': labels_tensor
    }

class PairDataset(Dataset):
    def __init__(self, pairs_list, state_embeddings):
        self.pairs_list = pairs_list
        self.state_embeddings = state_embeddings

    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, idx):
        pair = self.pairs_list[idx]
        appname = pair["appname"]
        s1      = pair["state1"]
        s2      = pair["state2"]
        lbl     = pair["label"]

        emb1 = self.state_embeddings.get((appname, s1))
        emb2 = self.state_embeddings.get((appname, s2))
        if emb1 is None or emb2 is None:
            return None

        # Return a dict that the collate_fn will combine
        return {
            'embeddings1': emb1,
            'embeddings2': emb2,
            'labels': torch.tensor(lbl, dtype=torch.long)
        }

class SMOTEDPairDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        concat_emb = self.X[idx]  # shape [2*embed_dim]
        label      = self.y[idx]

        concat_emb_t = torch.tensor(concat_emb, dtype=torch.float32)
        half_dim      = concat_emb_t.shape[0] // 2

        emb1 = concat_emb_t[:half_dim]
        emb2 = concat_emb_t[half_dim:]

        return {
            'embeddings1': emb1,
            'embeddings2': emb2,
            'labels': torch.tensor(label, dtype=torch.long)
        }

##############################################################################
#   Triplet Training Dataset Creation and Dataloaders Creation              #
##############################################################################

def create_triplets(pairs_list):
    positive_dict = defaultdict(list)  # (appname, anchor) -> list_of_pos
    negative_dict = defaultdict(list)  # (appname, anchor) -> list_of_neg

    # 1) Build dictionaries of positives & negatives
    for p in pairs_list:
        appname = p["appname"]
        s1 = p["state1"]
        s2 = p["state2"]
        lbl = p["label"]  # 1 => near-duplicate, 0 => distinct

        if lbl == 1:
            positive_dict[(appname, s1)].append(s2)
            positive_dict[(appname, s2)].append(s1)
        else:
            negative_dict[(appname, s1)].append(s2)
            negative_dict[(appname, s2)].append(s1)

    # 2) Create triplets: one per pair
    triplets = []
    for p in pairs_list:
        appname = p["appname"]
        s1 = p["state1"]
        s2 = p["state2"]
        lbl = p["label"]

        if lbl == 1:
            # near-duplicate => anchor=s1, positive=s2, pick a negative
            anchor = s1
            neg_candidates = negative_dict.get((appname, anchor), [])
            if not neg_candidates:
                continue
            positive = s2
            negative = random.choice(neg_candidates)
        else:
            # distinct => anchor=s1, negative=s2, pick a positive
            anchor = s1
            pos_candidates = positive_dict.get((appname, anchor), [])
            if not pos_candidates:
                continue
            negative = s2
            positive = random.choice(pos_candidates)

        triplets.append({
            "appname": appname,
            "anchor": anchor,
            "positive": positive,
            "negative": negative
        })

    return triplets

class TripletDataset(Dataset):
    def __init__(self, triplets, state_embeddings):
        self.triplets = triplets
        self.state_embeddings = state_embeddings

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triple = self.triplets[idx]
        appname = triple["appname"]
        anchor_st = triple["anchor"]
        pos_st = triple["positive"]
        neg_st = triple["negative"]

        emb_anchor = self.state_embeddings.get((appname, anchor_st))
        emb_pos = self.state_embeddings.get((appname, pos_st))
        emb_neg = self.state_embeddings.get((appname, neg_st))

        if emb_anchor is None or emb_pos is None or emb_neg is None:
            emb_anchor = torch.zeros_like(list(self.state_embeddings.values())[0])
            emb_pos = torch.zeros_like(emb_anchor)
            emb_neg = torch.zeros_like(emb_anchor)
        else:
            if isinstance(emb_anchor, torch.Tensor):
                emb_anchor = emb_anchor.clone().detach().float()
            else:
                emb_anchor = torch.tensor(emb_anchor, dtype=torch.float32)

            if isinstance(emb_pos, torch.Tensor):
                emb_pos = emb_pos.clone().detach().float()
            else:
                emb_pos = torch.tensor(emb_pos, dtype=torch.float32)

            if isinstance(emb_neg, torch.Tensor):
                emb_neg = emb_neg.clone().detach().float()
            else:
                emb_neg = torch.tensor(emb_neg, dtype=torch.float32)

        return {
            "anchor": emb_anchor,
            "positive": emb_pos,
            "negative": emb_neg
        }

def triplet_collate_fn(batch):
    if not batch:
        return None

    anchors = [item["anchor"] for item in batch]
    positives = [item["positive"] for item in batch]
    negatives = [item["negative"] for item in batch]

    anchors = torch.stack(anchors, dim=0)
    positives = torch.stack(positives, dim=0)
    negatives = torch.stack(negatives, dim=0)

    return {
        "anchor": anchors,
        "positive": positives,
        "negative": negatives
    }

##############################################################################
#   Dataloaders Creation - Across App  Setting                               #
##############################################################################

#For Contrastive Siamese NN
def prepare_datasets_and_loaders_across_app_contrastive(
    pairs_data,
    test_app,
    state_embeddings,
    batch_size=32,
    seed=42
):
    pairs_by_app = defaultdict(list)
    for d in pairs_data:
        pairs_by_app[d["appname"]].append(d)

    apps = list(pairs_by_app.keys())
    if test_app not in apps:
        test_app = apps[-1]

    # Combine all apps except the test_app into training
    train_pairs = []
    for a in apps:
        if a != test_app:
            train_pairs.extend(pairs_by_app[a])

    test_pairs = pairs_by_app[test_app]

    # Shuffle once for test pairs
    random.seed(seed)
    random.shuffle(test_pairs)

    # Shuffle train pairs
    random.seed(seed)
    random.shuffle(train_pairs)

    # Split train pairs => 90% train, 10% validation
    split_idx       = int(len(train_pairs) * 0.9)
    new_train_pairs = train_pairs[:split_idx]
    val_pairs       = train_pairs[split_idx:]

    # 1) Gather embeddings + labels from new_train_pairs
    X_train = []
    y_train = []
    for pair in new_train_pairs:
        appname = pair["appname"]
        s1      = pair["state1"]
        s2      = pair["state2"]
        lbl     = pair["label"]
        emb1    = state_embeddings.get((appname, s1))
        emb2    = state_embeddings.get((appname, s2))

        if emb1 is None or emb2 is None:
            continue

        # Concatenate embeddings along last dimension
        # If emb1 and emb2 are torch Tensors, convert to numpy
        emb1_np = emb1.cpu().numpy() if isinstance(emb1, torch.Tensor) else emb1
        emb2_np = emb2.cpu().numpy() if isinstance(emb2, torch.Tensor) else emb2

        concat_emb = np.concatenate([emb1_np, emb2_np], axis=0)
        X_train.append(concat_emb)
        y_train.append(lbl)

    if len(X_train) == 0:
        print("[Warning] No train samples to oversample. Check data.")
        return None, None, None

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"Original class distribution in training data: {Counter(y_train)}")

    sm = SMOTE(random_state=seed)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)

    train_dataset = SMOTEDPairDataset(X_sm, y_sm)
    print(f"Oversampled class distribution after SMOTE: {Counter(y_sm)}")


    val_dataset  = PairDataset(val_pairs,  state_embeddings)
    test_dataset = PairDataset(test_pairs, state_embeddings)

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=g,
        collate_fn=pair_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pair_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pair_collate_fn
    )

    return train_loader, val_loader, test_loader

#For Triplet Siamese NN
def prepare_datasets_and_loaders_across_app_triplet(
    pairs_data,
    test_app,
    state_embeddings,
    batch_size=32,
    seed=42
):
    pairs_by_app = defaultdict(list)
    for d in pairs_data:
        pairs_by_app[d["appname"]].append(d)

    apps = list(pairs_by_app.keys())
    if test_app not in apps:
        test_app = apps[-1]

    train_pairs = []
    for a in apps:
        if a != test_app:
            train_pairs.extend(pairs_by_app[a])

    test_pairs = pairs_by_app[test_app]
    random.shuffle(test_pairs)

    random.seed(seed)
    random.shuffle(train_pairs)
    split_idx = int(len(train_pairs) * 0.9)
    new_train_pairs = train_pairs[:split_idx]  # 90% for training
    val_pairs = train_pairs[split_idx:]  # 10% for validation

    train_triplets = create_triplets(new_train_pairs)

    train_dataset = TripletDataset(train_triplets, state_embeddings)
    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=g,
        collate_fn=triplet_collate_fn
    )

    val_dataset = PairDataset(val_pairs, state_embeddings)
    test_dataset = PairDataset(test_pairs, state_embeddings)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pair_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pair_collate_fn
    )

    return train_loader, val_loader, test_loader


##############################################################################
#   Dataloaders Creation - With in App  Setting                              #
##############################################################################

def prepare_datasets_and_loaders_within_app_contrastive(
    app_pairs,
    state_embeddings,
    batch_size,
    seed,
    train_ratio=0.8,
    val_ratio=0.1
):
    random.seed(seed)
    random.shuffle(app_pairs)

    total_len = len(app_pairs)
    if total_len == 0:
        return None, None, None

    # Compute split sizes
    train_size = int(train_ratio * total_len)
    val_size   = int(val_ratio * total_len)

    # Create splits
    train_pairs = app_pairs[:train_size]
    val_pairs   = app_pairs[train_size : train_size + val_size]
    test_pairs  = app_pairs[train_size + val_size :]

    # SMOTE for class imbalance
    X_train = []
    y_train = []
    for pair in train_pairs:
        appname = pair["appname"]
        s1      = pair["state1"]
        s2      = pair["state2"]
        lbl     = pair["label"]
        emb1    = state_embeddings.get((appname, s1))
        emb2    = state_embeddings.get((appname, s2))

        if emb1 is None or emb2 is None:
            continue

        # Concatenate embeddings along last dimension
        emb1_np = emb1.cpu().numpy() if isinstance(emb1, torch.Tensor) else emb1
        emb2_np = emb2.cpu().numpy() if isinstance(emb2, torch.Tensor) else emb2

        concat_emb = np.concatenate([emb1_np, emb2_np], axis=0)
        X_train.append(concat_emb)
        y_train.append(lbl)

    if len(X_train) == 0:
        print("[Warning] No train samples to oversample. Check data.")
        return None, None, None

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"Original class distribution in training data: {Counter(y_train)}")

    sm = SMOTE(random_state=seed)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)

    train_dataset = SMOTEDPairDataset(X_sm, y_sm)
    print(f"Oversampled class distribution after SMOTE: {Counter(y_sm)}")

    # Build validation and test datasets without SMOTE
    val_dataset  = PairDataset(val_pairs,  state_embeddings)
    test_dataset = PairDataset(test_pairs, state_embeddings)

    # Setup Dataloaders
    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=g,
        collate_fn=pair_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pair_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pair_collate_fn
    )

    return train_loader, val_loader, test_loader

def prepare_datasets_and_loaders_within_app_triplet(
    app_pairs,
    state_embeddings,
    batch_size,
    seed,
    train_ratio=0.8,
    val_ratio=0.1
):
    random.seed(seed)
    random.shuffle(app_pairs)

    total_len = len(app_pairs)
    if total_len == 0:
        return None, None, None

    # Compute split sizes
    train_size = int(train_ratio * total_len)
    val_size = int(val_ratio * total_len)

    # Create splits
    train_pairs = app_pairs[:train_size]
    val_pairs = app_pairs[train_size: train_size + val_size]
    test_pairs = app_pairs[train_size + val_size:]

    train_triplets = create_triplets(train_pairs)

    train_dataset = TripletDataset(train_triplets, state_embeddings)
    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=g,
        collate_fn=triplet_collate_fn
    )

    val_dataset = PairDataset(val_pairs, state_embeddings)
    test_dataset = PairDataset(test_pairs, state_embeddings)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pair_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pair_collate_fn
    )

    return train_loader, val_loader, test_loader