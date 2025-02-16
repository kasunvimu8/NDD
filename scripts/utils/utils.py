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
import sqlite3
from torch.backends import mps


####################################################
#            Common functions                      #
####################################################

def initialize_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("[Info] Using device:", device)
    return device

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

    if "TrainingTime" in df.columns:
        # Convert training times to numeric (ignoring non-numeric values like "N/A")
        numeric_times = pd.to_numeric(df["TrainingTime"], errors="coerce")
        if numeric_times.count() > 0:
            summary_row["TrainingTime"] = numeric_times.mean()
        else:
            summary_row["TrainingTime"] = "N/A"
    summary_df = pd.DataFrame([summary_row])

    # Concatenate the summary row to the original DataFrame
    df = pd.concat([df, summary_df], ignore_index=True)

    # Save the DataFrame to an Excel file
    df.to_excel(output_xl, index=False)
    print(f"[Info] All results saved to {output_xl}")

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
        rows = cursor.execute(query, (appname,)).fetchall()
        for appname_val, s1, s2, hc, retained_val in rows:
            # Skip rows that are not retained
            if retained_val != 1:
                continue

            label = map_labels(hc)
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