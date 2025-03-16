import json
import os
import sys
from bs4 import BeautifulSoup
from gensim.models import Doc2Vec
from transformers import AutoTokenizer, AutoModel, MarkupLMProcessor
from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append("/Users/kasun/Documents/uni/semester-4/thesis/NDD")
from scripts.utils.utils import fix_json_crawling, get_model, initialize_device, preprocess_dom_text, dfs_collect_tokens_xpaths, chunk_tokens_xpaths

base_path        = "/Users/kasun/Documents/uni/semester-4/thesis/NDD"
doc2vec_path     = f"/{base_path}/resources/embedding-models/content_tags_model_train_setsize300epoch50.doc2vec.model"
no_of_inferences = 0

# Configurations
configurations = [
    {
        'model_name' : "bert-base-uncased",
        'title' : "withinapp_bert",
        'embedding_type' : "bert",
        'setting' : "contrastive",
        'chunk_size' : 512,
        'overlap' : 0,
        'chunk_limit' : 5,
        'doc2vec_path' : None,
        'lr' : 5e-05,
        'epochs' : 50,
    },
    {
        'model_name': "bert-base-uncased",
        'title': "withinapp_bert",
        'embedding_type': "bert",
        'setting': "triplet",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 5,
        'doc2vec_path' : None,
        'lr' : 0.0005,
        'epochs' : 50,
    },
    {
        'model_name': None,
        'title': "withinapp_doc2vec",
        'embedding_type': "doc2vec",
        'setting': "contrastive",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 5,
        'doc2vec_path': doc2vec_path,
        'lr' : 5e-05,
        'epochs' : 50,
    },
    {
        'model_name': None,
        'title': "withinapp_doc2vec",
        'embedding_type': "doc2vec",
        'setting': "triplet",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 5,
        'doc2vec_path': doc2vec_path,
        'lr' : 0.001,
        'epochs': 50,
    },
    {
        'model_name': "microsoft/markuplm-base",
        'title': "withinapp_markuplm",
        'embedding_type': "markuplm",
        'setting': "contrastive",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 5,
        'doc2vec_path': None,
        'lr' : 5e-05,
        'epochs': 50,
    },
    {
        'model_name': "microsoft/markuplm-base",
        'title': "withinapp_markuplm",
        'embedding_type': "markuplm",
        'setting': "triplet",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 5,
        'doc2vec_path': None,
        'lr' : 0.0005,
        'epochs': 50,
    },
    {
        'model_name' : "bert-base-uncased",
        'title' : "acrossapp_bert",
        'embedding_type' : "bert",
        'setting' : "contrastive",
        'chunk_size' : 512,
        'overlap' : 0,
        'chunk_limit' : 2,
        'doc2vec_path' : None,
        'lr' : 2e-05,
        'epochs' : 10,
    },
    {
        'model_name': "bert-base-uncased",
        'title': "acrossapp_bert",
        'embedding_type': "bert",
        'setting': "triplet",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 2,
        'doc2vec_path' : None,
        'lr' : 2e-05,
        'epochs' : 15,
    },
    {
        'model_name': None,
        'title': "acrossapp_doc2vec",
        'embedding_type': "doc2vec",
        'setting': "contrastive",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 2,
        'doc2vec_path': doc2vec_path,
        'lr' : 2e-05,
        'epochs' : 10,
    },
    {
        'model_name': None,
        'title': "acrossapp_doc2vec",
        'embedding_type': "doc2vec",
        'setting': "triplet",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 2,
        'doc2vec_path': doc2vec_path,
        'lr' : 0.0001,
        'epochs': 7,
    },
    {
        'model_name': "microsoft/markuplm-base",
        'title': "acrossapp_markuplm",
        'embedding_type': "markuplm",
        'setting': "contrastive",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 1,
        'doc2vec_path': None,
        'lr' : 2e-05,
        'epochs': 15,
    },
    {
        'model_name': "microsoft/markuplm-base",
        'title': "acrossapp_markuplm",
        'embedding_type': "markuplm",
        'setting': "triplet",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 1,
        'doc2vec_path': None,
        'lr' : 2e-05,
        'epochs': 12,
    },
    {
        'model_name': "Rocketknight1/falcon-rw-1b",
        'title': "acrossapp_refinedweb",
        'embedding_type': "refinedweb",
        'setting': "contrastive",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 2,
        'doc2vec_path': None,
        'lr': 2e-05,
        'epochs': 10,
    },
    {
        'model_name': "Rocketknight1/falcon-rw-1b",
        'title': "acrossapp_refinedweb",
        'embedding_type': "refinedweb",
        'setting': "triplet",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 2,
        'doc2vec_path': None,
        'lr': 2e-05,
        'epochs': 15,
    },
    {
        'model_name': "Rocketknight1/falcon-rw-1b",
        'title': "withinapp_refinedweb",
        'embedding_type': "refinedweb",
        'setting': "contrastive",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 5,
        'doc2vec_path': None,
        'lr': 5e-05,
        'epochs': 50,
    },
    {
        'model_name': "Rocketknight1/falcon-rw-1b",
        'title': "withinapp_refinedweb",
        'embedding_type': "refinedweb",
        'setting': "triplet",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 5,
        'doc2vec_path': None,
        'lr': 0.0005,
        'epochs': 50,
    }
]

title            = "withinapp_doc2vec"
appname          = "pagekit" # appname is treated as within app -> target app and across app -> test app
setting          = "triplet" # contrastive or triplet
dimensions       = 300

current_configs   = [config for config in configurations if config['title'] == title]
current_config    = current_configs[0] if (current_configs[0]['setting'] == 'contrastive' and setting == 'contrastive') else current_configs[1]

model_name       = current_config['model_name']
setting          = current_config['setting']
embedding_type   = current_config['embedding_type']
chunk_size       = current_config['chunk_size']
chunk_limit      = current_config['chunk_limit']
overlap          = current_config['overlap']
trained_epochs   = current_config['epochs']
lr               = current_config['lr']

def increase_no_of_inferences():
    global no_of_inferences
    no_of_inferences += 1
    if no_of_inferences % 10 == 0:
        print(f"Number of inferences: {no_of_inferences}")

def load_model_and_tokenizer():
    model_path = f"{base_path}/models/{title}_{setting}_{appname}_cl_{chunk_limit}_bs_128_ep_{trained_epochs}_lr_{lr}_wd_0.01.pt"

    if not os.path.exists(model_path):
        print(f"[Warning] Model file not found at {model_path}. Skipping.")
        sys.exit(1)


    classification_model = get_model(model_path, setting, device, dimensions)
    classification_model.to(device)

    model_state = torch.load(model_path, map_location=device, weights_only=True)
    classification_model.load_state_dict(model_state, strict=True)
    classification_model.eval()

    return classification_model, dimensions

def chunk_text_crawling(text, chunk_size, overlap):
    tokens = text.split()
    step = max(1, chunk_size - overlap)
    chunks = []
    i = 0
    while i < len(tokens):
        chunks.append(" ".join(tokens[i : i + chunk_size]))
        i += step
    return chunks

def parse_dom_into_nodes_xpaths(dom_str):
    soup = BeautifulSoup(dom_str, "html.parser")
    soup = soup.html
    if not soup:
        return []

    for tag in soup(['style', 'script']):
        tag.decompose()

    collected = []
    dfs_collect_tokens_xpaths(soup, "/html[1]", collected)

    return collected

def embed_dom_bert_crawling(dom_str, tokenizer, embedding_model, device, chunk_size, dimension, overlap):
    clean_str = preprocess_dom_text(dom_str)
    chunk_list = chunk_text_crawling(clean_str, chunk_size, overlap)

    if not chunk_list:
        return torch.zeros(dimension, device=device)

    embedding_model.eval()
    chunk_embs = []

    with torch.no_grad():
        for chunk_text in chunk_list:
            inputs = tokenizer(chunk_text, return_tensors="pt", max_length=512, truncation=True)
            for k in inputs:
                inputs[k] = inputs[k].to(device)

            outputs = embedding_model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # shape: [1, 768]
            chunk_embs.append(cls_emb)

    # Stack into shape [num_chunks, 1, 768], then take the mean along dim=0 => [1, 768]
    chunk_embs = torch.stack(chunk_embs, dim=0)  # [num_chunks, 1, 768]
    mean_emb = torch.mean(chunk_embs, dim=0)     # [1, 768]

    return mean_emb.to(device)

def embed_dom_doc2vec_crawling(dom_str, doc2vec_model, device):
    clean_str = preprocess_dom_text(dom_str)
    tokens = clean_str.split()

    # shape [300]
    doc_vec = doc2vec_model.infer_vector(tokens)

    # Convert to torch and reshape to [1, 300]
    emb = torch.tensor(doc_vec, dtype=torch.float, device=device).unsqueeze(0)
    return emb

def embed_dom_markuplm_crawling(dom_str, markup_model, processor, device, chunk_size, dimension, overlap):
    markup_model.eval()
    hidden_dim = markup_model.config.hidden_size

    # global_max_chunks = dimension // hidden_dim
    global_max_chunks = dimension // hidden_dim if hidden_dim else 1
    if global_max_chunks < 1:
        global_max_chunks = 1

    token_xpath_list = parse_dom_into_nodes_xpaths(dom_str)
    if not token_xpath_list:
        print(f"[Warning] No tokens and xpaths found in {dom_str}. Skipping.")
        return torch.zeros(1, dimension, device=device)

    chunks = chunk_tokens_xpaths(token_xpath_list, chunk_size=chunk_size, overlap=overlap)

    chunk_embs = []
    with torch.no_grad():
        for c in chunks:
            tokens_chunk = [pair[0] for pair in c]
            xpaths_chunk = [pair[1] for pair in c]

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

            outputs = markup_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            # We take the CLS embedding => shape [batch_size=1, hidden_dim]
            cls_emb = outputs.last_hidden_state[:, 0, :]
            chunk_embs.append(cls_emb.squeeze(0))

    num_chunks = len(chunk_embs)

    if num_chunks > global_max_chunks:
        chunk_embs = chunk_embs[:global_max_chunks]
        num_chunks = global_max_chunks

    if num_chunks < global_max_chunks:
        for _ in range(global_max_chunks - num_chunks):
            chunk_embs.append(torch.zeros(hidden_dim, device=device))

    # Concatenate => shape [global_max_chunks*hidden_dim], then unsqueeze(0) => [1, global_max_chunks*hidden_dim]
    final_emb = torch.cat(chunk_embs, dim=0)  # [global_max_chunks * hidden_dim]
    final_emb = final_emb.unsqueeze(0)        # => [1, dimension]

    return final_emb  # shape [1, dimension]

def get_embedding(dom_str, model_name, embedding_type, chunk_size, dimension, overlap):
    if embedding_type in ('bert', 'refinedweb'):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        bert_model = AutoModel.from_pretrained(model_name)
        bert_model.to(device)

        state_embedding = embed_dom_bert_crawling(
            dom_str=dom_str,
            tokenizer=tokenizer,
            embedding_model=bert_model,
            device=device,
            chunk_size=chunk_size,
            dimension=dimension,
            overlap=overlap,
        )
        return state_embedding
    elif embedding_type == 'markuplm':
        processor = MarkupLMProcessor.from_pretrained(model_name)
        processor.parse_html = False
        markup_model = AutoModel.from_pretrained(model_name)
        markup_model.to(device)

        state_embedding = embed_dom_markuplm_crawling(
            dom_str=dom_str,
            markup_model=markup_model,
            processor=processor,
            device=device,
            chunk_size=chunk_size,
            dimension=dimension,
            overlap=overlap
        )
        print(f"shape {state_embedding.shape}")
        return state_embedding
    elif embedding_type == 'doc2vec':
        doc2vec_model = Doc2Vec.load(doc2vec_path)
        doc2vec_model.random.seed(42)  # fix seed if needed

        # produce [1, 300]
        state_embedding = embed_dom_doc2vec_crawling(dom_str, doc2vec_model, device)
        return state_embedding
    else:
        raise ValueError(f"Unknown embedding_type: {embedding_type}")

def saf_equals(
        dom1,
        dom2,
        classification_model,
        model_name,
        embedding_type,
        setting,
        chunk_size=512,
        dimension=768,
        overlap=0,
        threshold=0.5
):
    """
    Returns 1 if dom1 and dom2 are considered duplicates, else 0.

    setting="contrastive":  uses classification_model(emb1, emb2) -> outputs['logits']
                 probability => compare with threshold
    setting="triplet": uses classification_model.forward_once(...) and
                    distance => compare with threshold
    """
    emb1 = get_embedding(dom1, model_name, embedding_type, chunk_size, dimension, overlap)
    emb2 = get_embedding(dom2, model_name, embedding_type, chunk_size, dimension, overlap)

    if setting == "contrastive":
        # Contrastive(BCE) approach
        outputs = classification_model(emb1, emb2)
        logits = outputs["logits"].squeeze(1)  # shape [1]
        probs = torch.sigmoid(logits)
        print(f"[Info] BCE probability : {probs.item()}")
        preds = (probs > threshold).float()
        return int(preds.item())

    elif setting == "triplet":
        # Triplet-based approach => compute distance
        out1 = classification_model.forward_once(emb1)
        out2 = classification_model.forward_once(emb2)

        distance = F.pairwise_distance(out1, out2)  # shape [1]
        print(f"[Info] Triplet distance : {distance.item()}")
        # If distance <= threshold => duplicates
        pred = 1 if distance.item() <= threshold else 0
        return pred

    else:
        raise ValueError(f"Unknown mode: {setting}")

device = initialize_device()
classification_model, dimension = load_model_and_tokenizer()
app = Flask(__name__)

@app.route('/equals', methods=('GET', 'POST'))
def equals_route():
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json' or content_type == 'application/json; utf-8':
        fixed_json = fix_json_crawling(request.data.decode('utf-8'))
        if fixed_json == "Error decoding JSON":
            print("Exiting due to JSON error")
            exit(1)
        data = json.loads(fixed_json)
    else:
        return 'Content-Type not supported!'

    parametersJava = data

    dom1 = parametersJava['dom1']
    dom2 = parametersJava['dom2']
    url1 = parametersJava['url1']
    url2 = parametersJava['url2']

    # compute equality of DOM objects
    result = saf_equals(dom1, dom2, classification_model, model_name, embedding_type, setting, chunk_size, dimension, overlap, threshold=0.5)
    result = "true" if result == 1 else "false"
    increase_no_of_inferences()

    print(f"[Info] url1 : {url1}, url2 : {url2}, results -> {result}")
    return result

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "OK", "message": "Service is up and running. Call /equals for SAF service"})

if __name__ == "__main__":
    print(f"******* We are using the model: {title} - {setting}  *******")
    app.run(debug=False)
