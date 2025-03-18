import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append("/Users/kasun/Documents/uni/semester-4/thesis/NDD")

from scripts.utils.utils import (
    set_all_seeds,
    load_pairs_from_db, initialize_device, embed_dom_doc2vec_crawling,
)

if __name__ == "__main__":
    seed = 42
    set_all_seeds(seed)
    
    # List of applications to process
    selected_apps = [
        'addressbook', 'claroline', 'ppma', 'mrbs',
        'mantisbt', 'dimeshift', 'pagekit', 'phoenix', 'petclinic'
    ]
    
    # Paths
    base_path   = "/Users/kasun/Documents/uni/semester-4/thesis/NDD"
    db_path     = f"{base_path}/dataset/SS_refined.db"
    table_name  = "nearduplicates"
    results_dir = f"{base_path}/results"
    dom_root_dir = f"{base_path}/resources/doms"

    doc2vec_path    = f"{base_path}/resources/embedding-models/content_tags_model_train_setsize300epoch50.doc2vec.model"
    classifier_path = f"/Users/kasun/Documents/uni/semester-4/thesis/Baseline-NDD/trained_classifiers"
    output_file     = f"{results_dir}/rq4/inference_times_baseline_webembed.xlsx"
    
    # Sample size
    sample_size = 1000
    
    # Load the doc2vec model
    doc2vec_model = Doc2Vec.load(doc2vec_path)
    device = initialize_device()

    # -------------------------------------------------------------------------
    # Load pairs and sampling
    # -------------------------------------------------------------------------
    all_pairs    = load_pairs_from_db(db_path, table_name, selected_apps)
    all_pairs_df = pd.DataFrame(all_pairs)
    SS_sampled   = all_pairs_df.sample(n=sample_size, random_state=seed)
    
    print("\n[Info] Class distribution of the sample:")
    print(SS_sampled['appname'].value_counts())
    
    # -------------------------------------------------------------------------
    # Inference Timing
    # -------------------------------------------------------------------------
    results = []

    for setting in ['within-apps', 'across-apps']:

        app_time = {app: 0 for app in selected_apps}
        for app in selected_apps:
            # baseline classifier path
            classifier = f"{classifier_path}/{setting}-{app}-svm-rbf-doc2vec-distance-content-tags.sav"

            if not os.path.exists(classifier_path):
                print(f"[Warning] Classifier not found for {app} at {classifier_path}. Skipping.")
                continue

            print(f"\n[Info] Processing App: {app}")

            model = pickle.load(open(classifier, 'rb'))
            # Subset the sample for the current app
            app_dataset = SS_sampled[SS_sampled['appname'] == app]
            total_time = 0.0

            # Inference on each pair (no batching)
            for idx, row in app_dataset.iterrows():
                state_1 = row['state1']
                state_2 = row['state2']

                dom_path_1 = os.path.join(dom_root_dir, app, 'doms', f"{state_1}.html")
                dom_path_2 = os.path.join(dom_root_dir, app, 'doms', f"{state_2}.html")

                if not os.path.isfile(dom_path_1):
                    print('not found')
                    continue
                if not os.path.isfile(dom_path_2):
                    print('not found')
                    continue

                with open(dom_path_1, "r", encoding="utf-8", errors="ignore") as f:
                    dom_content_1 = f.read()
                with open(dom_path_2, "r", encoding="utf-8", errors="ignore") as f:
                    dom_content_2 = f.read()

                # this should be the place where the saf inference time measuring starts
                start_time = time.perf_counter()

                # Getting the doc2vec embedding
                emb1 = embed_dom_doc2vec_crawling(dom_content_1, doc2vec_model, device)
                emb2 = embed_dom_doc2vec_crawling(dom_content_2, doc2vec_model, device)

                emb1_cpu = emb1.cpu().detach().numpy()
                emb2_cpu = emb2.cpu().detach().numpy()

                # Compute similarity -> distance feature
                sim = cosine_similarity(emb1_cpu, emb2_cpu)
                dist = np.array([sim[0, 0]])  # Just shape (1,)
                dist = dist.reshape(1, -1)   # Make sure it's (1,1) for predict

                # Classification with the SVM
                pred = model.predict(dist)
                print('prediction:', pred)

                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                total_time += elapsed_time

            # Average inference time for the pairs of this app
            avg_time = total_time / len(app_dataset) if len(app_dataset) > 0 else 0
            app_time[app] = avg_time

        for app in selected_apps:
            results.append({
                'App': app,
                'Title': 'Webembed',
                'Setting': setting,
                'Inference Time (s)': app_time[app]
            })

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_excel(f"{results_dir}/rq4/webembed_inference_times.xlsx", index=False)
    print(f"\n[Info] Baseline WebEmbed inference times saved")
