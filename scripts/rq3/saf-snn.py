import json
import os
import sys
from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from gensim.models import Doc2Vec
from transformers import AutoTokenizer, AutoModel, MarkupLMProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append("/Users/kasun/Documents/uni/semester-4/thesis/NDD")
from scripts.utils.utils import fix_json_crawling, get_model, initialize_device, saf_equals, set_all_seeds

base_path        = "/Users/kasun/Documents/uni/semester-4/thesis/NDD"
doc2vec_path     = f"/{base_path}/resources/embedding-models/content_tags_model_train_setsize300epoch50.doc2vec.model"
no_of_inferences = 0

app_to_dim = {
    'withinapp_markuplm' : {
        'addressbook' : 2304,
        'claroline' : 768,
        'ppma' : 768,
        'mrbs' : 2304,
        'mantisbt' : 1536,
        'dimeshift' : 1536,
        'pagekit' : 768,
        'phoenix' : 768,
        'petclinic' : 768,
    },
    'acrossapp_markuplm' : {
        'addressbook' : 768,
        'claroline' : 768,
        'ppma' : 768,
        'mrbs' : 768,
        'mantisbt' : 768,
        'dimeshift' : 768,
        'pagekit' : 768,
        'phoenix' : 768,
        'petclinic' : 768,
    },
    
}

# Configurations
configurations = [
    {
        'model_name': None,
        'title': "withinapp_doc2vec",
        'embedding_type': "doc2vec",
        'setting': "contrastive",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 5,
        'doc2vec_path': doc2vec_path,
        'lr' : 1e-04,
        'epochs' : 15,
        'wd' : 0.05,
        'bs' : 32,
    },
    {
        'model_name': "bert-base-uncased",
        'title': "withinapp_bert",
        'embedding_type': "bert",
        'setting': "contrastive",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 5,
        'doc2vec_path': None,
        'lr': 1e-04,
        'epochs': 15,
        'wd' : 0.05,
        'bs' : 32,
    },
    {
        'model_name': "answerdotai/ModernBERT-base",
        'title': "withinapp_modernbert",
        'embedding_type': "bert",
        'setting': "contrastive",
        'chunk_size': 8192,
        'overlap': 0,
        'chunk_limit': 5,
        'doc2vec_path': None,
        'lr': 1e-04,
        'epochs': 15,
        'wd': 0.05,
        'bs': 32,
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
        'lr': 1e-04,
        'epochs': 15,
        'wd' : 0.05,
        'bs' : 32,
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
        'lr': 1e-03,
        'epochs': 15,
        'wd': 0.05,
        'bs': 32,
    },
    {
        'model_name': "bert-base-uncased",
        'title': "withinapp_bert",
        'embedding_type': "bert",
        'setting': "triplet",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 5,
        'doc2vec_path': None,
        'lr': 1e-03,
        'epochs': 15,
        'wd': 0.05,
        'bs': 32,
    },
    {
        'model_name': "answerdotai/ModernBERT-base",
        'title': "withinapp_modernbert",
        'embedding_type': "bert",
        'setting': "triplet",
        'chunk_size': 8192,
        'overlap': 0,
        'chunk_limit': 5,
        'doc2vec_path': None,
        'lr': 1e-03,
        'epochs': 15,
        'wd': 0.05,
        'bs': 32,
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
        'lr': 1e-03,
        'epochs': 15,
        'wd': 0.05,
        'bs': 32,
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
        'lr': 2e-05,
        'epochs': 10,
        'wd': 0.01,
        'bs': 128,
    },
    {
        'model_name': "bert-base-uncased",
        'title': "acrossapp_bert",
        'embedding_type': "bert",
        'setting': "contrastive",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 2,
        'doc2vec_path': None,
        'lr': 2e-05,
        'epochs': 10,
        'wd': 0.01,
        'bs': 128,
    },
    {
        'model_name': "answerdotai/ModernBERT-base",
        'title': "acrossapp_modernbert",
        'embedding_type': "bert",
        'setting': "contrastive",
        'chunk_size': 8192,
        'overlap': 0,
        'chunk_limit': 2,
        'doc2vec_path': None,
        'lr': 0,
        'epochs': 10,
        'wd': 0.01,
        'bs': 128,
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
        'lr': 2e-05,
        'epochs': 15,
        'wd': 0.01,
        'bs': 128,
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
        'lr': 0.0001,
        'epochs': 7,
        'wd': 0.01,
        'bs': 128,
    },
    {
        'model_name': "bert-base-uncased",
        'title': "acrossapp_bert",
        'embedding_type': "bert",
        'setting': "triplet",
        'chunk_size': 512,
        'overlap': 0,
        'chunk_limit': 2,
        'doc2vec_path': None,
        'lr': 2e-05,
        'epochs': 15,
        'wd': 0.01,
        'bs': 128,
    },
    {
        'model_name': "answerdotai/ModernBERT-base",
        'title': "acrossapp_modernbert",
        'embedding_type': "bert",
        'setting': "triplet",
        'chunk_size': 8192,
        'overlap': 0,
        'chunk_limit': 2,
        'doc2vec_path': None,
        'lr': 2e-05,
        'epochs': 15,
        'wd': 0.01,
        'bs': 128,
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
        'lr': 2e-05,
        'epochs': 12,
        'wd': 0.01,
        'bs': 128,
    },
]
title            = "acrossapp_modernbert"
appname          = "petclinic" # appname is treated as within app -> target app and across app -> test app
setting          = "triplet" # contrastive or triplet

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
bs               = current_config['bs']
wd               = current_config['wd']


if embedding_type == 'markuplm':
    dimensions = app_to_dim[title][appname]
elif embedding_type == 'bert':
    dimensions = 768
elif embedding_type == 'doc2vec':
    dimensions = 300

def increase_no_of_inferences():
    global no_of_inferences
    no_of_inferences += 1
    if no_of_inferences % 10 == 0:
        print(f"Number of inferences: {no_of_inferences}")

def load_model_and_tokenizer(embedding_type, model_name):
    embedding_model = None
    tokenizer = None
    processor = None
    model_path = f"{base_path}/models/{title}_{setting}_{appname}_cl_{chunk_limit}_bs_{bs}_ep_{trained_epochs}_lr_{lr}_wd_{wd}.pt"

    if not os.path.exists(model_path):
        print(f"[Warning] Model file not found at {model_path}. Skipping.")
        sys.exit(1)

    if embedding_type == 'doc2vec':
        embedding_model = Doc2Vec.load(doc2vec_path)
        embedding_model.random.seed(42)  # fix seed if needed

    elif embedding_type == 'bert':
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        embedding_model = AutoModel.from_pretrained(model_name)
        embedding_model.to(device)

    elif embedding_type == 'markuplm':
        processor = MarkupLMProcessor.from_pretrained(model_name)
        processor.parse_html = False
        embedding_model = AutoModel.from_pretrained(model_name)
        embedding_model.to(device)
    else:
        print(f"[Error] Unknown embedding type {embedding_type}. Skipping.")
        sys.exit(1)


    classification_model = get_model(model_path, setting, device, dimensions)
    classification_model.to(device)

    model_state = torch.load(model_path, map_location=device, weights_only=True)
    classification_model.load_state_dict(model_state, strict=True)
    classification_model.eval()

    return classification_model, embedding_model, tokenizer, processor

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
    result = saf_equals(
        dom1=dom1,
        dom2=dom2,
        classification_model=classification_model,
        embedding_model=embedding_model,
        processor=processor,
        tokenizer=tokenizer,
        embedding_type=embedding_type,
        setting=setting,
        device=device,
        chunk_size=chunk_size,
        dimension=dimensions,
        overlap=overlap,
        threshold=0.5
       )
    result = "true" if result == 1 else "false"
    increase_no_of_inferences()

    print(f"[Info] url1 : {url1}, url2 : {url2}, results -> {result}")
    return result

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "OK", "message": "Service is up and running. Call /equals for SAF service"})

if __name__ == "__main__":
    seed = 42
    set_all_seeds(seed)
    device = initialize_device()
    classification_model, embedding_model, tokenizer, processor = load_model_and_tokenizer(embedding_type, model_name)
    print(f"******* We are using the model: {appname} - {title} - {setting} *******")
    app.run(debug=False)
