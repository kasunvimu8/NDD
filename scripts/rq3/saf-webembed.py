import os
import json
import pickle
import sys

import numpy as np
from flask import Flask, request, jsonify
from gensim.models import Doc2Vec
from torch import cosine_similarity
sys.path.append("/Users/kasun/Documents/uni/semester-4/thesis/NDD")

from scripts.utils.utils import embed_dom_doc2vec_crawling, initialize_device, fix_json_crawling

# -----------------------------------------------------------------
#  Global settings and baseline model info
# -----------------------------------------------------------------
no_of_inferences = 0
BASE_PATH = "/Users/kasun/Documents/uni/semester-4/thesis/NDD"
MODEL_DIR = os.path.join(BASE_PATH, "resources/baseline-trained-classifiers")
SELECTED_APPS = [
    'addressbook', 'claroline', 'ppma', 'mrbs',
    'mantisbt', 'dimeshift', 'pagekit', 'phoenix', 'petclinic'
]

baseline_model_info = {
    "webembed": {
        "withinapps": "within-apps-{app}-svm-rbf-doc2vec-distance-content-tags.sav",
        "acrossapp": "across-apps-{app}-svm-rbf-doc2vec-distance-content-tags.sav",
    }
}

# -----------------------------------------------------------------
#  Load the doc2vec model (used for the "webembed" feature extraction)
# -----------------------------------------------------------------
DOC2VEC_PATH = os.path.join(BASE_PATH, "resources/embedding-models", "content_tags_model_train_setsize300epoch50.doc2vec.model")
doc2vec_model = Doc2Vec.load(DOC2VEC_PATH)

# -----------------------------------------------------------------
#  Utility functions
# -----------------------------------------------------------------
def increase_no_of_inferences():
    global no_of_inferences
    no_of_inferences += 1
    if no_of_inferences % 10 == 0:
        print(f"[Info] Number of inferences: {no_of_inferences}")

def load_baseline_model(appname, setting):
    if appname not in SELECTED_APPS:
        raise ValueError(f"Unknown appname: {appname}")
    if 'webembed' not in baseline_model_info:
        raise ValueError(f"Unknown baseline method: webembed")
    if setting not in baseline_model_info["webembed"]:
        raise ValueError(f"Unknown baseline setting for webembed: {setting}")

    filename = baseline_model_info["webembed"][setting].format(app=appname)
    model_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    return clf

def saf_equals_baseline(dom1, dom2, classifier, doc2vec_model, device):
    # Get the doc2vec embeddings for both DOMs (returns tensors)
    emb1 = embed_dom_doc2vec_crawling(dom1, doc2vec_model, device)
    emb2 = embed_dom_doc2vec_crawling(dom2, doc2vec_model, device)

    # Compute cosine similarity directly on the tensors
    sim = cosine_similarity(emb1, emb2, dim=1)
    sim_value = sim.item()

    dist = np.array([[sim_value]])

    # Classification with the SVM
    pred = classifier.predict(dist)
    return int(pred[0])
# -----------------------------------------------------------------
#  Build the Flask app
# -----------------------------------------------------------------
app = Flask(__name__)

# Configure the baseline parameters for the webembed method.
appname = "petclinic"        # one of SELECTED_APPS
setting = "withinapps"       # valid options: 'withinapps' or 'acrossapp'
classifier_baseline = load_baseline_model(appname, setting)

device = initialize_device()


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "OK",
        "message": "Baseline Service is up. Call /equals to compare two states."
    })

@app.route('/equals', methods=['POST'])
def equals_route():
    content_type = request.headers.get('Content-Type')
    if content_type not in ('application/json', 'application/json; utf-8'):
        return 'Content-Type not supported!', 400

    fixed_json = fix_json_crawling(request.data.decode('utf-8'))
    if fixed_json == "Error decoding JSON":
        print("Exiting due to JSON error")
        return "Error decoding JSON", 400

    data = json.loads(fixed_json)

    # Expect the request to include 'dom1' and 'dom2' (and optionally url1, url2)
    dom1 = data['dom1']
    dom2 = data['dom2']
    url1 = data.get('url1', 'N/A')
    url2 = data.get('url2', 'N/A')

    nd_label = saf_equals_baseline(dom1=dom1,
                                   dom2=dom2,
                                   classifier=classifier_baseline,
                                   doc2vec_model=doc2vec_model,
                                   device=device)
    result = "true" if nd_label == 1 else "false"

    increase_no_of_inferences()

    print(f"[Info] url1: {url1}, url2: {url2}, result -> {result}")
    return result

if __name__ == "__main__":
    print(f"******* Starting Baseline SAF: {appname}  - {setting} *******")
    app.run(debug=False)
