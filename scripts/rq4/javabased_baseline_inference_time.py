import os
import sys
import time
import subprocess
import pickle
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append("/Users/kasun/Documents/uni/semester-4/thesis/NDD")

from scripts.utils.utils import (
    set_all_seeds,
    load_pairs_from_db,
)

if __name__ == "__main__":
    # Set random seed for reproducibility
    seed = 42
    set_all_seeds(seed)

    # List of applications to process
    selected_apps = [
        'addressbook', 'claroline', 'ppma', 'mrbs',
        'mantisbt', 'dimeshift', 'pagekit', 'phoenix', 'petclinic'
    ]

    # Base paths and directories
    base_path       = "/Users/kasun/Documents/uni/semester-4/thesis/NDD"
    db_path         = f"{base_path}/dataset/SS_refined.db"
    table_name      = "nearduplicates"
    results_dir     = f"{base_path}/results"
    dom_root_dir    = f"{base_path}/resources/doms"
    jar_path        = f"{base_path}/resources/baseline-runner/BaseLineRunner-1.0-SNAPSHOT.jar"
    classifier_path = f"/Users/kasun/Documents/uni/semester-4/thesis/Baseline-NDD/trained_classifiers"

    output_file = f"{results_dir}/rq4/rted_pdiff_inference_times.xlsx"

    # Sample size for pairs
    sample_size = 1000

    # Load pairs from DB and sample a subset
    all_pairs = load_pairs_from_db(db_path, table_name, selected_apps)
    all_pairs_df = pd.DataFrame(all_pairs)
    SS_sampled = all_pairs_df.sample(n=sample_size, random_state=seed)

    print("\n[Info] Class distribution of the sample:")
    print(SS_sampled['appname'].value_counts())

    results = []

    # Methods to test via the jar: rted, pdiff, fraggen
    methods = ["rted", "pdiff", "fraggen"]

    for setting in ['within-apps', 'across-apps']:
        for method in methods:
            app_time = {app: 0.0 for app in selected_apps}

            for app in selected_apps:
                app_dataset = SS_sampled[SS_sampled['appname'] == app]
                if app_dataset.empty:
                    print(f"[Warning] No pairs found for app {app}. Skipping.")
                    continue

                total_time = 0.0
                count = 0

                print(f"\n[Info - {methods} - {setting} ] Processing App: {app} for method: {method.upper()}")

                for idx, row in app_dataset.iterrows():
                    state_1 = row['state1']
                    state_2 = row['state2']

                    if method in ["rted", "fraggen"]:
                        file_path_1 = os.path.join(dom_root_dir, app, "doms", f"{state_1}.html")
                        file_path_2 = os.path.join(dom_root_dir, app, "doms", f"{state_2}.html")
                    elif method == "pdiff":
                        file_path_1 = os.path.join(dom_root_dir, app, "screenshots", f"{state_1}.png")
                        file_path_2 = os.path.join(dom_root_dir, app, "screenshots", f"{state_2}.png")
                    else:
                        continue

                    if not os.path.isfile(file_path_1):
                        print(f"[Warning] File not found: {file_path_1}")
                        continue
                    if not os.path.isfile(file_path_2):
                        print(f"[Warning] File not found: {file_path_2}")
                        continue

                    # The jar is expected to output "distance,elapsedSeconds" on stdout.
                    cmd = ["java", "-jar", jar_path, method, file_path_1, file_path_2]

                    try:
                        # Call the jar and get its output
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        output = result.stdout.strip()  # Expected output: "distance,elapsedSeconds"

                        parts = output.split(',')
                        if len(parts) != 2:
                            print(f"[Error] Unexpected output format for pair {idx}: {output}")
                            continue
                        distance_val = float(parts[0])
                        jar_elapsed_time = float(parts[1])

                        if method == "rted" or method == "pdiff":
                            model_key = "dom-rted" if method == "rted" else "visual-pdiff"
                            classifier = f"{classifier_path}/{setting}-{app}-svm-rbf-{model_key}.sav"

                            try:
                                model = pickle.load(open(classifier, 'rb'))
                                start_time = time.perf_counter()
                                prediction = model.predict(np.array(distance_val).reshape(1, -1))
                                end_time = time.perf_counter()

                                # distance time from jar and classification time svm
                                total_time = total_time +  jar_elapsed_time + (end_time - start_time)
                                count += 1
                            except FileNotFoundError:
                                print("Cannot find classifier %s" % classifier)
                                exit()
                            except pickle.UnpicklingError:
                                print(classifier)
                                exit()
                        else:
                            total_time += jar_elapsed_time
                            count += 1

                        print(f"[Info] Pair {idx} - Jar Output: {output}")
                    except subprocess.CalledProcessError as e:
                        print(f"[Error] Command failed for pair {idx}: {cmd}")
                        print(e)
                        continue

                avg_time = total_time / count if count > 0 else 0.0
                app_time[app] = avg_time

            # Append results for this method
            for app in selected_apps:
                results.append({
                    "App": app,
                    "Title": method.upper(),
                    "Setting": setting,
                    "Inference Time (s)": app_time[app]
                })

    # Save the aggregated results to an Excel file
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"\n[Info] Baseline WebEmbed inference times saved")