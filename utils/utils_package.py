import os
import random
import re
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import gensim
from bs4 import BeautifulSoup, Comment, NavigableString
import pandas as pd
from datetime import datetime

from gensim.models import Doc2Vec
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm
from transformers import MarkupLMProcessor, AutoModel, BertTokenizer, BertModel
import sqlite3
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)
from imblearn.over_sampling import SMOTE

####################################################
#            Common functions and classes          #
####################################################

def set_all_seeds(seed):
    """
    Set seeds for reproducibility across random, numpy, torch, and CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cudnn.deterministic = True
    cudnn.benchmark = False

def initialize_weights(model, seed):
    """
    Initializes model weights deterministically using Kaiming (He) initialization for ReLU layers.
    """
    set_all_seeds(seed)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def process_html(content):
    soup = BeautifulSoup(content, 'html.parser')

    # Remove <style> and <script> tags and their content
    for tag in soup(['style', 'script']):
        tag.decompose()

    corpus = []
    retrieve_abstraction_from_html(soup, corpus)
    return corpus

def preprocess_dom_text(dom_content):
    """
    Simple function to preprocess HTML DOM:
    """
    corpus = process_html(dom_content)
    html_cleaned = ' '.join(corpus)

    # remove tags, opening,closing and self-closing tags
    html_cleaned = re.sub(r'<\s*/?\s*\[?([^\s>/\]]+)\]?[^>]*>', r'\1', html_cleaned)
    return html_cleaned

def retrieve_abstraction_from_html(bs, corpus):
    """
    Recursively traverses the BeautifulSoup parse tree `bs`,
    appending tag markers and text tokens to the `corpus` list.
    """
    try:
        # If the current element is a string, tokenize and append its content.
        if isinstance(bs, NavigableString):
            # Skip if it's just whitespace
            text = bs.string.strip() if bs.string else ""
            if text:
                tokens = gensim.utils.simple_preprocess(text)
                corpus.extend(tokens)
            return

        bs_has_name = bs.name is not None
        bs_is_single_tag = False
        if bs_has_name:
            bs_is_single_tag = not bool(list(bs.children))

        if bs_has_name:
            tag_repr = f'<{bs.name}>' if not bs_is_single_tag else f'<{bs.name}/>'
            corpus.append(tag_repr)

        for child in bs.children:
            if isinstance(child, Comment):
                continue
            retrieve_abstraction_from_html(child, corpus)

        if bs_has_name and not bs_is_single_tag:
            corpus.append(f'</{bs.name}>')
    except Exception as e:
        print('HTML structure content error:', e)

def build_xpath(parent_xpath, child_tag, index_in_parent):
    """
    Constructs an xpath-like string for a child node.
    Example:
      parent_xpath = "/html/body/div"
      child_tag = "span"
      index_in_parent = 2
    => "/html/body/div/span[2]"
    """
    return f"{parent_xpath}/{child_tag}[{index_in_parent}]"

def dfs_collect_tokens_xpaths(soup, current_xpath, collected):
    """
    DFS through the DOM. For each Tag, build an xpath. For each NavigableString,
    tokenize its text and store (token, xpath).
    'collected' is a list of (token, xpath) pairs.
    """
    if isinstance(soup, NavigableString):
        text = str(soup).strip()
        if text:
            tokens = gensim.utils.simple_preprocess(text)
            for t in tokens:
                collected.append((t, current_xpath))
        return

    if isinstance(soup, Comment) or soup.name is None:
        return

    child_elements = [c for c in soup.children if c.name or isinstance(c, NavigableString)]
    for idx, child in enumerate(child_elements, start=1):
        child_xpath = build_xpath(current_xpath, child.name if child.name else "text", idx)
        dfs_collect_tokens_xpaths(child, child_xpath, collected)

def parse_html_and_extract_tokens_xpaths(dom_path):
    """
    Reads an HTML file, removes <style> and <script>, then does DFS to collect (token, xpath) pairs.
    """
    from bs4 import BeautifulSoup
    try:
        with open(dom_path, 'r', encoding='utf-8') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'html.parser')
        soup = soup.html
        if not soup:
            return []

        for tag in soup(['style', 'script']):
            tag.decompose()

        collected = []
        dfs_collect_tokens_xpaths(soup, "/html[1]", collected)
        return collected
    except Exception as e:
        print(f"[Error] parse_html_and_extract_tokens_xpaths for {dom_path}: {e}")
        return []

def chunk_tokens_xpaths(token_xpath_list, chunk_size=512, overlap=256):
    """
    Splits the list of (token, xpath) into sub-lists of length chunk_size,
    with a stride of (chunk_size - overlap).
    """
    stride = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(token_xpath_list):
        sublist = token_xpath_list[start : start + chunk_size]
        chunks.append(sublist)
        start += stride
    return chunks

def map_labels(human_classification):
    """
    Transform classification into binary:
       - 2 => 0  (distinct)
       - 0 or 1 => 1  (similar)
       - else => None
    """
    val = int(human_classification)
    if val == 2:
        return 0
    elif val in [0, 1]:
        return 1
    else:
        return None

def load_pairs_from_db(db_path, table_name, selected_apps):
    """
    Reads all state pairs from the given DB + table for apps in `selected_apps`.
    Only keep rows where is_retained=1. Then we transform them into:
        {
          "appname": str,
          "state1": str,
          "state2": str,
          "label": 0 or 1
        }
    Returns a list of pairs (dict).
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    placeholders = ', '.join([f"'{app}'" for app in selected_apps])
    query = f"""
        SELECT
            appname,
            state1,
            state2,
            HUMAN_CLASSIFICATION,
            is_retained
        FROM {table_name}
        WHERE appname IN ({placeholders})
    """

    pairs = []
    app_class_dist = defaultdict(Counter)  # For class distribution per app
    total_class_dist = Counter()  # For total class distribution

    try:
        rows = cursor.execute(query).fetchall()
        for appname, s1, s2, hc, retained_val in rows:
            if retained_val != 1:
                continue
            label = map_labels(hc)
            if label is None:
                continue
            pairs.append({
                "appname": appname,
                "state1":  s1,
                "state2":  s2,
                "label":   label
            })
            # Update class distribution for this app
            app_class_dist[appname][label] += 1

            # Update total class distribution
            total_class_dist[label] += 1
    except Exception as e:
        print("[Error] load_pairs_from_db:", e)
    finally:
        conn.close()

    # Log class distribution by app
    for appname, dist in app_class_dist.items():
        print(f"Class distribution for app {appname}: {dict(dist)}")

    # Log total class distribution
    print(f"Total class distribution: {dict(total_class_dist)}")
    return pairs

def gather_state_chunks_with_xpaths(
    pairs_data,
    dom_root_dir,
    chunk_size=512,
    overlap=256,
    chunk_threshold=5
):
    """
    1) For each (appname, state) found in the pairs, parse DOM => get (token, xpath).
    2) Chunk them => [ (tokens_chunk, xpaths_chunk), ...]
    3) Return a dict: state_chunks[(appname, state)] = [ (tokens_chunk, xpaths_chunk), ... ]
    4) Also track global_max_chunks across all states, but enforce chunk_threshold if given.
    """
    state_chunks = {}
    global_max_chunks = 0

    unique_states = set()
    for d in pairs_data:
        unique_states.add((d["appname"], d["state1"]))
        unique_states.add((d["appname"], d["state2"]))

    # track how many states get truncated
    truncated_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for (appname, state_id) in unique_states:
        total_counts[appname] += 1

        dom_path = os.path.join(dom_root_dir, appname, 'doms', f"{state_id}.html")
        if not os.path.exists(dom_path):
            continue

        token_xpath_list = parse_html_and_extract_tokens_xpaths(dom_path)
        if not token_xpath_list:
            continue

        chunks = chunk_tokens_xpaths(token_xpath_list, chunk_size=chunk_size, overlap=overlap)

        if len(chunks) > chunk_threshold:
            chunks = chunks[:chunk_threshold]
            truncated_counts[appname] += 1

        final_chunks = []
        for c in chunks:
            tokens_chunk = [pair[0] for pair in c]
            xpaths_chunk = [pair[1] for pair in c]
            final_chunks.append((tokens_chunk, xpaths_chunk))

        if len(final_chunks) > global_max_chunks:
            global_max_chunks = len(final_chunks)

        state_chunks[(appname, state_id)] = final_chunks

    # Report truncated states
    print("\n[Truncation/Cropping Report]")
    for app in sorted(truncated_counts.keys()):
        truncated = truncated_counts[app]
        total = total_counts[app]
        print(f"  App: {app}, truncated states: {truncated}/{total}")

    return state_chunks, global_max_chunks

def save_results_to_excel(title, results, results_dir, setting_key, overlap, batch_size, num_epochs, lr, weight_decay, chunk_limit):
    """
    Converts results to a DataFrame, appends a summary row, and saves as an Excel file.
    """
    if not results:
        print("[Info] No results to save. Possibly no data in DB?")
        return
    abs_results_dir = os.path.abspath(results_dir)

    if not os.path.exists(abs_results_dir):
        os.makedirs(abs_results_dir)

    # Convert results list of dicts to a DataFrame
    df = pd.DataFrame(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_xl = os.path.join(
        results_dir,
        f"{title}_results_{setting_key}_overlap_{overlap}_cl_{chunk_limit}_bs_{batch_size}_ep_{num_epochs}_lr_{lr}_wd_{weight_decay}_{timestamp}.xlsx"
    )

    # Create a summary row DataFrame from the mean of each column
    summary_row = {
        "TestApp": "Summary (Average)",
        "Accuracy": df["Accuracy"].mean(),
        "Precision": df["Precision"].mean(),
        "Recall": df["Recall"].mean(),
        "F1_Class 0": df["F1_Class 0"].mean(),
        "F1_Class 1": df["F1_Class 1"].mean(),
        "F1 Score (Weighted Avg)": df["F1 Score (Weighted Avg)"].mean(),
    }
    summary_df = pd.DataFrame([summary_row])

    # Concatenate the summary row to the original DataFrame
    df = pd.concat([df, summary_df], ignore_index=True)

    # Save the DataFrame to an Excel file
    df.to_excel(output_xl, index=False)
    print(f"[Info] All results saved to {output_xl}")

##############################################################################
#                Networks                                                    #
##############################################################################

class SiameseNN(nn.Module):
    """
    Siamese model for binary classification using BCEWithLogitsLoss.
    Incorporates multiple combination methods (concat, abs diff, product, exp(-diff)).
    """
    def __init__(self, input_dim):
        super(SiameseNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),  # Feature dimension => 64

            nn.Linear(64, 32),
            nn.ReLU(),  # final embedding is dimension 32
        )

        # Combine out1, out2, |diff|, product, exp(-|diff|) => 5 * 32 = 160
        self.classifier = nn.Sequential(
            nn.Linear(32 * 5, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(32, 1)  # Single logit
        )

    def forward_once(self, x):
        return self.feature_extractor(x)

    def forward(self, emb1, emb2, labels=None):
        out1 = self.forward_once(emb1)  # shape [B, 32]
        out2 = self.forward_once(emb2)  # shape [B, 32]

        abs_diff = torch.abs(out1 - out2)
        prod     = out1 * out2
        sim      = torch.exp(-abs_diff)

        combined = torch.cat([out1, out2, abs_diff, prod, sim], dim=1)  # shape [B, 160]
        logits   = self.classifier(combined)                            # shape [B, 1]

        outputs = {'logits': logits}
        if labels is not None:
            labels = labels.float().unsqueeze(1)  # shape => [B, 1]
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            outputs['loss'] = loss
        return outputs

class TripletSiameseNN(nn.Module):
    """
    Similar architecture to your SiameseNN but returns embeddings (not logits).
    We'll do the distance + margin logic externally in TripletLoss.
    """
    def __init__(self, input_dim):
        super(TripletSiameseNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU()  # Final embedding size=32
        )

    def forward_once(self, x):
        return self.feature_extractor(x)

    def forward(self, anchor, positive, negative):
        anchor_out = self.forward_once(anchor)
        pos_out = self.forward_once(positive)
        neg_out = self.forward_once(negative)
        return anchor_out, pos_out, neg_out

##############################################################################
#                Loss functions                                              #
##############################################################################

class TripletLoss(nn.Module):
    """
    Margin-based triplet loss:
      L = mean( max(0, d(anchor, pos)^2 - d(anchor, neg)^2 + margin ) ).
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_out, pos_out, neg_out):
        dist_pos = F.pairwise_distance(anchor_out, pos_out)
        dist_neg = F.pairwise_distance(anchor_out, neg_out)

        loss = dist_pos.pow(2) - dist_neg.pow(2) + self.margin
        loss = torch.clamp(loss, min=0.0).mean()
        return loss

##############################################################################
#                 Pair setting (BCE)                                         #
##############################################################################

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

def prepare_datasets_and_loaders_bce(
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

    random.seed(seed)
    random.shuffle(train_pairs)

    print(f" trained pairs length: {len(train_pairs)}")

    train_dataset = PairDataset(new_train_pairs, state_embeddings)
    val_dataset   = PairDataset(val_pairs, state_embeddings)
    test_dataset  = PairDataset(test_pairs, state_embeddings)

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

def prepare_datasets_and_loaders_bce_balanced(
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

    # --- SMOTE CHANGES START ---
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

        # Filter out None or mismatch
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

    # 2) Apply SMOTE
    sm = SMOTE(random_state=seed)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)

    # 3) Build a custom dataset from SMOTE outputs
    train_dataset = SMOTEDPairDataset(X_sm, y_sm)
    print(f"Oversampled class distribution after SMOTE: {Counter(y_sm)}")

    # --- SMOTE CHANGES END ---

    # Build validation and test datasets in the old way, no SMOTE
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

def train_one_epoch_bce(model, dataloader, optimizer, device, epoch, num_epochs):
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

def validate_model_bce(model, dataloader, device):
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

def test_model_bce(model, dataloader, device, threshold=0.5):
    """
    Evaluate model => return a dict of metrics.
    """
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
            logits = outputs['logits'].squeeze(1)     # [B]
            probs = torch.sigmoid(logits)            # [B]
            preds = (probs > threshold).float()      # [B]

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
#                 Triplet setting                                           #
##############################################################################

def create_triplets(pairs_list):
    """
    Creates exactly one triplet per labeled pair, if possible.
      - For label=1 pairs (near-duplicate): anchor=s1, positive=s2, sample negative
      - For label=0 pairs (distinct): anchor=s1, negative=s2, sample positive
    Generates up to len(pairs_list) triplets in total.
    """

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
    """
    Triplet dataset: each item => anchor, positive, negative
    Uses your existing 'state_embeddings' dict => {(appname, state_id): embedding}.
    """
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
    """
    Collate function => stacks anchor, positive, negative embeddings
    for the entire batch into 3 big tensors: [B, input_dim]
    """
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

def prepare_datasets_and_loaders_triplets(
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

def train_one_epoch_triplets(model, dataloader, optimizer, device, epoch, num_epochs, margin=1.0):
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


def validate_model_triplets(model, val_loader, device, threshold=0.5):
    """
    A simple pair-based validation.
    We'll compute a 'val_loss' as the average classification error
    (i.e., how often we disagree with the true label).

    If distance <= threshold => predicted_label=1, else 0.
    val_loss = proportion of misclassified samples.
    """
    model.eval()
    total_misclassified = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            emb1 = batch["embeddings1"].to(device)
            emb2 = batch["embeddings2"].to(device)
            labels = batch["labels"].to(device).float()  # 0 or 1

            out1 = model.forward_once(emb1)  # [B, emb_dim]
            out2 = model.forward_once(emb2)  # [B, emb_dim]

            distances = F.pairwise_distance(out1, out2)  # [B]
            preds = (distances <= threshold).float()

            mismatches = (preds != labels).float().sum().item()
            total_misclassified += mismatches
            total_samples += len(labels)

    if total_samples == 0:
        return 0.0

    val_loss = total_misclassified / total_samples
    return val_loss


def test_model_triplets(model, test_loader, device, threshold=0.5):
    """
    Pair-based testing => compute distance, then compare to threshold to get
    near-duplicate(1)/distinct(0). Return a dict with accuracy, precision, recall, f1, etc.
    """
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
            labels = batch["labels"].cpu().numpy()  # keep on CPU for metrics

            out1 = model.forward_once(emb1)  # [B, emb_dim]
            out2 = model.forward_once(emb2)  # [B, emb_dim]

            distances = F.pairwise_distance(out1, out2)  # [B]
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


##############################################################################
#                  Embedding - MarkupLM                                     #
##############################################################################

def embed_states_fixed_length_markuplm(
    state_chunks,
    global_max_chunks,
    processor,
    markup_model,
    device
):
    """
    For each state, produce a fixed-length embedding (global_max_chunks * hidden_dim).
    Zero-pad if fewer than global_max_chunks chunks.
    """
    state_embeddings = {}
    hidden_dim = markup_model.config.hidden_size

    for (key, chunks) in tqdm(state_chunks.items(), desc="Embedding states"):
        chunk_embs = []
        for (tokens_chunk, xpaths_chunk) in chunks:
            encoding = processor(
                nodes=tokens_chunk,
                xpaths=xpaths_chunk,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            token_type_ids = encoding.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            with torch.no_grad():
                outputs = markup_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            emb = outputs.last_hidden_state[:, 0, :]  # [batch_size=1, hidden_dim]
            chunk_embs.append(emb.squeeze(0))         # shape [hidden_dim]

        # Zero pad
        num_chunks = len(chunk_embs)
        if num_chunks < global_max_chunks:
            for _ in range(global_max_chunks - num_chunks):
                chunk_embs.append(torch.zeros(hidden_dim, device=device))

        final_emb = torch.cat(chunk_embs, dim=0)  # shape [global_max_chunks*hidden_dim]
        state_embeddings[key] = final_emb.cpu()

    return state_embeddings, global_max_chunks * hidden_dim

def run_embedding_pipeline_markuplm(
    pairs_data,
    dom_root_dir,
    chunk_size,
    overlap,
    device,
    markup_model_name="microsoft/markuplm-base",
    chunk_threshold=5
):
    """
    1) Gathers DOM data for states in pairs_data
    2) Loads MarkupLM
    3) Embeds states to a fixed dimension
    4) Returns (state_embeddings, final_input_dim)
    """
    # 1) gather
    state_chunks, global_max_chunks = gather_state_chunks_with_xpaths(
        pairs_data, dom_root_dir, chunk_size, overlap, chunk_threshold
    )
    if global_max_chunks == 0:
        return None, 0

    # 2) load model
    processor = MarkupLMProcessor.from_pretrained(markup_model_name)
    processor.parse_html = False
    markup_model = AutoModel.from_pretrained(markup_model_name)
    markup_model.eval()
    markup_model.to(device)

    # 3) embed
    state_embeddings, final_input_dim = embed_states_fixed_length_markuplm(
        state_chunks,
        global_max_chunks,
        processor,
        markup_model,
        device
    )
    return state_embeddings, final_input_dim


##############################################################################
#                  Embedding - Doc2Vec                                      #
##############################################################################

def run_doc2vec_embedding_pipeline(pairs_data, dom_root_dir, doc2vec_model_path):
    # Load the doc2vec model
    d2v_model = Doc2Vec.load(doc2vec_model_path)
    vector_size = d2v_model.vector_size

    unique_states = set()
    for d in pairs_data:
        unique_states.add((d["appname"], d["state1"]))
        unique_states.add((d["appname"], d["state2"]))

    state_embeddings = {}
    for (appname, state_id) in unique_states:
        if state_id not in state_embeddings:
            # Load the DOM file
            dom_path = os.path.join(dom_root_dir, appname, 'doms', f"{state_id}.html")
            if not os.path.isfile(dom_path):
                # If missing, skip or handle gracefully
                state_embeddings[state_id] = np.zeros(vector_size, dtype=np.float32)
                continue

            with open(dom_path, "r", encoding="utf-8", errors="ignore") as f:
                dom_content = f.read()

            # Preprocess text => remove angle brackets, etc.
            cleaned_text = preprocess_dom_text(dom_content)

            # Convert to list of tokens (Doc2Vec usually uses tokenized input)
            tokens = cleaned_text.split()

            # Inference with doc2vec
            embedding = d2v_model.infer_vector(tokens)
            state_embeddings[(appname, state_id)] = torch.tensor(embedding, dtype=torch.float)

    return state_embeddings, vector_size

##############################################################################
#                  Embedding - Bert-Large-Uncased                            #
##############################################################################

def gather_state_chunks_bert( pairs_data, dom_root_dir, chunk_size, overlap, chunk_threshold):
    # A set of unique states
    unique_states = set()
    for d in pairs_data:
        unique_states.add((d["appname"], d["state1"]))
        unique_states.add((d["appname"], d["state2"]))

    # Data structures
    state_chunks = {}
    global_max_chunks = 0

    # track how many states get truncated
    truncated_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for (appname, state_id) in unique_states:
        total_counts[appname] += 1

        # Path to the DOM file
        dom_path = os.path.join(dom_root_dir, appname, 'doms', f"{state_id}.html")
        if not os.path.isfile(dom_path):
            # If missing, store an empty chunk list
            state_chunks[(appname, state_id)] = []
            continue

        with open(dom_path, "r", encoding="utf-8", errors="ignore") as f:
            dom_content = f.read()

        cleaned_text = preprocess_dom_text(dom_content)
        tokens = cleaned_text.split()

        chunk_step = chunk_size - overlap if (chunk_size > overlap) else chunk_size
        chunk_list = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_list.append(" ".join(chunk_tokens))
            start += chunk_step

        # Truncate if #chunks > chunk_threshold
        if len(chunk_list) > chunk_threshold:
            chunk_list = chunk_list[:chunk_threshold]
            truncated_counts[appname] += 1

        # Update global max
        if len(chunk_list) > global_max_chunks:
            global_max_chunks = len(chunk_list)

        # Store in dictionary
        state_chunks[(appname, state_id)] = chunk_list

    # Report truncated states
    print("\n[Truncation/Cropping Report]")
    for app in sorted(truncated_counts.keys()):
        truncated = truncated_counts[app]
        total = total_counts[app]
        print(f"  App: {app}, truncated states: {truncated}/{total}")

    return state_chunks, global_max_chunks

def embed_states_fixed_length_bert(state_chunks, tokenizer, bert_model, device):
    bert_model.eval()
    hidden_size = bert_model.config.hidden_size
    state_embeddings = {}

    with torch.no_grad():
        for ((appname, state_id), chunk_list) in tqdm(state_chunks.items(), desc="Embedding states"):
            if len(chunk_list) == 0:
                # No content => zero vector
                final_emb = torch.zeros(hidden_size, dtype=torch.float)
                state_embeddings[(appname, state_id)] = final_emb
                continue

            chunk_embs = []
            for chunk_text in chunk_list:
                inputs = tokenizer(
                    chunk_text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                )
                # Move to device
                for k in inputs.keys():
                    inputs[k] = inputs[k].to(device)

                outputs = bert_model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]  # (1, hidden_size)
                chunk_embs.append(cls_emb)

            chunk_embs = torch.stack(chunk_embs, dim=0)  # shape (N, 1, hidden_size)
            state_emb = torch.mean(chunk_embs, dim=0)  # shape (1, hidden_size)
            state_emb = state_emb.squeeze(0).cpu()

            state_embeddings[(appname, state_id)] = state_emb

    return state_embeddings, hidden_size

def run_embedding_pipeline_bert(
        bert_model,
        tokenizer,
        pairs_data,
        dom_root_dir,
        chunk_size,
        overlap,
        device,
        chunk_threshold=5
):

    state_chunks, global_max_chunks = gather_state_chunks_bert(
        pairs_data,
        dom_root_dir,
        chunk_size,
        overlap,
        chunk_threshold
    )
    if global_max_chunks == 0:
        print("[Warning] global_max_chunks = 0. No data to embed.")
        return None, 0

    # 2) Load BERT model & tokenizer
    bert_model.to(device)

    # 3) Embed
    state_embeddings, final_input_dim = embed_states_fixed_length_bert(
        state_chunks,
        tokenizer,
        bert_model,
        device
    )

    return state_embeddings, final_input_dim

##############################################################################
#                  Across App Classification                                 #
##############################################################################

def load_single_app_pairs_from_db(db_path, table_name, appname):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f"""
            SELECT
                appname,
                state1,
                state2,
                HUMAN_CLASSIFICATION,
                is_retained
            FROM {table_name}
            WHERE appname = ?
            """

    pairs = []
    try:
        # Use parameterized query to safely pass the appname
        rows = cursor.execute(query, (appname,)).fetchall()
        print(len(rows))
        for appname_val, s1, s2, hc, retained_val in rows:
            # Skip rows that are not retained
            if retained_val != 1:
                continue

            label = map_labels(hc)
            # Skip rows with unrecognized labels
            if label is None:
                continue

            pairs.append({
                "appname": appname_val,
                "state1": s1,
                "state2": s2,
                "label": label
            })
    except Exception as e:
        print("[Error] load_pairs_from_db:", e)
    finally:
        conn.close()

    return pairs

def prepare_datasets_and_loaders_within_app_bce(
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

    # Create datasets
    train_dataset = PairDataset(train_pairs, state_embeddings)
    val_dataset   = PairDataset(val_pairs, state_embeddings)
    test_dataset  = PairDataset(test_pairs, state_embeddings)

    # Create DataLoaders
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

def prepare_datasets_and_loaders_within_app_bce_balanced(
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

    # SMOTE for class imbalance ---
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

    # 2) Apply SMOTE
    sm = SMOTE(random_state=seed)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)

    # 3) Build a oversampled training data
    train_dataset = SMOTEDPairDataset(X_sm, y_sm)
    print(f"Oversampled class distribution after SMOTE: {Counter(y_sm)}")
    # --- SMOTE CHANGES END ---

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