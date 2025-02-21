import json
import os
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flask import Flask, request, jsonify
import torch
import torch.nn as nn

sys.path.append("/Users/kasun/Documents/uni/semester-4/thesis/NDD")
from scripts.utils.utils import fix_json_crawling, get_model, initialize_device, preprocess_dom_text

base_path        = "/Users/kasun/Documents/uni/semester-4/thesis/NDD"

no_of_inferences = 0
title            = "withinapp_bert"
model_name       = "bert-base-uncased"
setting          = "contrastive"
embedding_type   = "bert"
appname         = "addressbook"
chunk_size      = 512
chunk_limit     = 5
overlap         = 0
trained_epochs  = 50
lr              = 5e-5

# counter for number of inferences
def increase_no_of_inferences():
    global no_of_inferences
    no_of_inferences += 1
    if no_of_inferences % 100 == 0:
        print(f"Number of inferences: {no_of_inferences}")

def get_dimensions(model_path):
    device = initialize_device()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    input_dim = None
    for key in state_dict.keys():
        if key.startswith("feature_extractor.0.weight"):
            # The weight shape is (output_features, input_dim).
            input_dim = state_dict[key].shape[1]
            print(f"[Info] Dimension : {input_dim}")
            break

    if input_dim is None :
        print(f"[Warning] Dimension : {input_dim} must be a multiple of 768 or Could not deduce the input dimension")
        sys.exit(1)

    return input_dim

def load_model_and_tokenizer():
    model_path = f"{base_path}/models/{title}_{setting}_{appname}_cl_{chunk_limit}_bs_128_ep_{trained_epochs}_lr_{lr}_wd_0.01.pt"

    if not os.path.exists(model_path):
        print(f"[Warning] Model file not found at {model_path}. Skipping.")
        sys.exit(1)

    dimensions = get_dimensions(model_path)
    device = initialize_device()

    model = get_model(model_path, setting, device, dimensions)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()

    return model, tokenizer, dimensions

def chunk_text_crawling(text, chunk_size=512, overlap=0):
    """
    Splits the text into overlapping segments.
    """
    tokens = text.split()
    step = max(1, chunk_size - overlap)
    chunks = []
    i = 0
    while i < len(tokens):
        chunks.append(" ".join(tokens[i : i + chunk_size]))
        i += step
    return chunks

def embed_dom_crawling(dom_str, tokenizer, model, device, chunk_size=512, dimension=5, overlap=0):
    clean_str = preprocess_dom_text(dom_str)
    chunks = chunk_text_crawling(clean_str, chunk_size, overlap)
    num_chunks_needed = dimension // 768

    if len(chunks) > num_chunks_needed:
        chunks = chunks[:num_chunks_needed]

    # Edge case: if no text or empty chunk list, return zero-vector
    if not chunks:
        return torch.zeros(model.config.hidden_size)

    model.eval()
    chunk_embs = []

    with torch.no_grad():
        for ch in chunks:
            inputs = tokenizer(ch, return_tensors="pt", max_length=512, truncation=True).to(device)
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            chunk_embs.append(cls_emb)

    while len(chunk_embs) < num_chunks_needed:
        chunk_embs.append(torch.zeros(768, device=device))

    final_emb = torch.cat(chunk_embs, dim=0)
    return final_emb.cpu()

def saf_equals(dom1, dom2, model, device, tokenizer, chunk_size, dimensions, overlap, threshold=0.5):
    model.eval()

    with torch.no_grad():
        emb1 = embed_dom_crawling(dom1, tokenizer, model, device, chunk_size, dimensions, overlap)
        emb2 = embed_dom_crawling(dom2, tokenizer, model, device, chunk_size, dimensions, overlap)

        emb1 = emb1.unsqueeze(0).to(device)
        emb2 = emb2.unsqueeze(0).to(device)

        # Forward pass through the contrastive model
        outputs = model(emb1, emb2)
        logits = outputs['logits'].squeeze(1)

        probs = torch.sigmoid(logits)

        preds = (probs > threshold).float()

    return int(preds.item())

model, tokenizer, dimensions = load_model_and_tokenizer()

app = Flask(__name__)

@app.route('/equals', methods=('GET', 'POST'))
def equals_route():
    print('Router called')
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json' or content_type == 'application/json; utf-8':
        fixed_json = fix_json_crawling(request.data.decode('utf-8'))
        if fixed_json == "Error decoding JSON":
            print("Exiting due to JSON error")
            exit(1)
        data = json.loads(fixed_json)
    else:
        return 'Content-Type not supported!'

    # get params sent by java
    parametersJava = data

    obj1 = parametersJava['dom1']
    obj2 = parametersJava['dom2']

    # compute equality of DOM objects
    result = saf_equals(obj1, obj2, model, tokenizer, chunk_size, dimensions, overlap, threshold=0.5)

    result = "true" if result == 1 else "false"

    increase_no_of_inferences()
    # return true if the two objects are clones/near-duplicates => comment was here before
    return result

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "OK", "message": "Service is up and running. Call /equals for SAF service"})

if __name__ == "__main__":
    print(f"******* We are using the model: {title}  *******")
    app.run(debug=False)
