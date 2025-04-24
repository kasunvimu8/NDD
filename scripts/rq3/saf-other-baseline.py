import os
import sys
import json
import pickle
from flask import Flask, request, jsonify
sys.path.append("/Users/kasun/Documents/uni/semester-4/thesis/NDD")

from scripts.utils.utils import fix_json_crawling

# -----------------------------------------------------------------
#  Global settings and baseline model info
# -----------------------------------------------------------------
no_of_inferences = 0
BASE_PATH = "/Users/kasun/Documents/uni/semester-4/thesis/NDD"
MODEL_DIR = f"{BASE_PATH}/resources/baseline-trained-classifiers"
SELECTED_APPS = [
    'addressbook', 'claroline', 'ppma', 'mrbs',
    'mantisbt', 'dimeshift', 'pagekit', 'phoenix', 'petclinic'
]

baseline_model_info = {
    "webembed": {
        "withinapps": "within-apps-{app}-svm-rbf-doc2vec-distance-content-tags.sav",
        "acrossapp": "across-apps-{app}-svm-rbf-doc2vec-distance-content-tags.sav",
    },
    "DOM_RTED": {
        "withinapps": "within-apps-{app}-svm-rbf-dom-rted.sav",
        "acrossapp": "across-apps-{app}-svm-rbf-dom-rted.sav",
    },
    "VISUAL_PDiff": {
        "withinapps": "within-apps-{app}-svm-rbf-visual-pdiff.sav",
        "acrossapp": "across-apps-{app}-svm-rbf-visual-pdiff.sav",
    }
}


# -----------------------------------------------------------------
#  Utility functions
# -----------------------------------------------------------------
def increase_no_of_inferences():
    global no_of_inferences
    no_of_inferences += 1
    if no_of_inferences % 10 == 0:
        print(f"[Info] Number of inferences: {no_of_inferences}")


def load_baseline_model(appname, method, setting):
    """
    appname: e.g. 'mantisbt'
    method:  e.g. 'webembed', 'DOM_RTED', or 'VISUAL_PDiff'
    setting: 'withinapps' or 'acrossapp'
    Returns a loaded SVM model.
    """
    if appname not in SELECTED_APPS:
        raise ValueError(f"Unknown appname: {appname}")
    if method not in baseline_model_info:
        raise ValueError(f"Unknown baseline method: {method}")
    if setting not in baseline_model_info[method]:
        raise ValueError(f"Unknown baseline setting for {method}: {setting}")

    filename = baseline_model_info[method][setting].format(app=appname)
    model_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    return clf


# -----------------------------------------------------------------
#  SAF equals for baseline methods using provided distance
# -----------------------------------------------------------------
def saf_equals_baseline_distance(distance, classifier):
    """
    Given a distance (float) directly from the request,
    use the classifier to predict if the two states are near-duplicates.

    Returns 1 if predicted near-duplicate (true), 0 if distinct (false).
    """
    pred = classifier.predict([[distance]])[0]
    return int(pred)


# -----------------------------------------------------------------
#  Build the Flask app
# -----------------------------------------------------------------
app = Flask(__name__)

appname = "phoenix"  # one of SELECTED_APPS
method = "DOM_RTED"  # one of:"DOM_RTED", "VISUAL_PDiff"
setting = "acrossapp"  # either 'withinapps' or 'acrossapp'
classifier_baseline = load_baseline_model(appname, method, setting)

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
    print(data)

    distance = data['distance']

    nd_label = saf_equals_baseline_distance(distance, classifier_baseline)
    result = "true" if nd_label == 1 else "false"

    increase_no_of_inferences()
    print(f"[Info] Distance: {distance}, result -> {result}")
    return result


if __name__ == "__main__":
    print(f"******* Starting Baseline SAF: {appname} - {method} - {setting} *******")
    app.run(debug=False)
