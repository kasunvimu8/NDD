import os
from collections import defaultdict
import numpy as np
import torch
from gensim.models import Doc2Vec
from tqdm import tqdm
from transformers import MarkupLMProcessor, AutoModel
from scripts.utils import preprocess_dom_text, chunk_tokens_xpaths, parse_html_and_extract_tokens_xpaths


##############################################################################
#                  Embedding - Doc2Vec                                      #
##############################################################################

def run_embedding_pipeline_doc2vec(pairs_data, dom_root_dir, doc2vec_model_path):
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
#                  Embedding - Bert                                          #
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
#                  Embedding - MarkupLM                                     #
##############################################################################

def gather_state_chunks_markuplm(
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
    markup_model_name,
    chunk_threshold=5
):
    """
    1) Gathers DOM data for states in pairs_data
    2) Loads MarkupLM
    3) Embeds states to a fixed dimension
    4) Returns (state_embeddings, final_input_dim)
    """
    state_chunks, global_max_chunks = gather_state_chunks_markuplm(
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
