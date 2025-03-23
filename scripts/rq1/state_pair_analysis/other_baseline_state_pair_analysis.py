import os
import pickle
import sqlite3
import pandas as pd

BASE_PATH   = "/Users/kasun/Documents/uni/semester-4/thesis/NDD"
DB_PATH     = f"{BASE_PATH}/dataset/SS_refined.db"
TABLE_NAME  = "nearduplicates"
MODEL_DIR   = f"{BASE_PATH}/resources/baseline-trained-classifiers"
CSV_PATH    = f"{BASE_PATH}/resources/baseline-dataset/SS_threshold_set.csv"
RESULTS_DIR = f"{BASE_PATH}/results"

SELECTED_APPS = [
    'addressbook', 'claroline', 'ppma', 'mrbs',
    'mantisbt', 'dimeshift', 'pagekit', 'phoenix', 'petclinic'
]

# -------------------------------------------------------
#  reading only retained records
# -------------------------------------------------------
def get_db_retained(sqlite_db_path, table_name):
    conn = sqlite3.connect(sqlite_db_path)
    query = f"""
        SELECT
            appname,
            state1,
            state2,
            HUMAN_CLASSIFICATION,
            is_retained,
            nd_type
        FROM {table_name}
        WHERE is_retained = 1
    """
    df_db_retained = pd.read_sql_query(query, conn)
    conn.close()
    return df_db_retained

# -------------------------------------------------------
#   ND analysis function
# -------------------------------------------------------
def compute_nd_analysis(y_true, y_pred, nd_types):
    """
    - y_true: list of ground-truth labels (0=distinct, 1=near-duplicate).
    - y_pred: list of predicted labels (0=distinct, 1=near-duplicate).
    - nd_types: list of ND categories in {0=clone, 2=ND2, 3=ND3} for near-duplicate pairs.
    """
    analysis_dict = {
        'total_clones': 0, 'correctly_pred_clones': 0,
        'total_nd2': 0, 'correctly_pred_nd2': 0,
        'total_nd3': 0, 'correctly_pred_nd3': 0,
        'total_distinct': 0, 'correctly_pred_distinct': 0
    }

    for i in range(len(y_true)):
        gt_label = y_true[i]  # 0 or 1
        pred_label = y_pred[i]  # 0 or 1
        nd_type_val = nd_types[i]  # 0, 2, or 3

        if gt_label == 0:
            # Distinct
            analysis_dict['total_distinct'] += 1
            if pred_label == 0:
                analysis_dict['correctly_pred_distinct'] += 1
        elif gt_label == 1:
            # Near-duplicate
            if nd_type_val == 0:
                # Clone
                analysis_dict['total_clones'] += 1
                if pred_label == 1:
                    analysis_dict['correctly_pred_clones'] += 1
            elif nd_type_val == 2:
                # ND2
                analysis_dict['total_nd2'] += 1
                if pred_label == 1:
                    analysis_dict['correctly_pred_nd2'] += 1
            elif nd_type_val == 3:
                # ND3
                analysis_dict['total_nd3'] += 1
                if pred_label == 1:
                    analysis_dict['correctly_pred_nd3'] += 1

    return analysis_dict


def main():
    # -------------------------------------------------------
    #   Load the main CSV + retained data from DB
    # -------------------------------------------------------
    df_csv = pd.read_csv(CSV_PATH, quotechar='"', escapechar='\\', on_bad_lines='warn')
    df_db_retained = get_db_retained(DB_PATH, TABLE_NAME)

    merge_cols = ["appname", "state1", "state2"]
    df_merged = pd.merge(
        df_db_retained,
        df_csv,
        on=merge_cols,
        how='left',
        suffixes=('_db', '')
    )

    df_merged['label'] = df_merged['HUMAN_CLASSIFICATION'].replace({0: 1, 1: 1, 2: 0})
    columns_to_keep = [
        "appname",
        "state1",
        "state2",
        "label",
        "nd_type",
        "doc2vec_distance_content_tags",
        "DOM_RTED",
        "VISUAL_PDiff"
    ]
    df_merged = df_merged[columns_to_keep]

    baseline_models_info = [
        {
            "name": "content_tags",
            "setting" : "withinapps",
            "filename": "within-apps-{app}-svm-rbf-doc2vec-distance-content-tags.sav",
            "embedding": "doc2vec_distance_content_tags"
        },
        {
            "name": "content_tags",
            "setting": "acrossapps",
            "filename": "across-apps-{app}-svm-rbf-doc2vec-distance-content-tags.sav",
            "embedding": "doc2vec_distance_content_tags"
        },
        {
            "name": "DOM_RTED",
            "setting": "withinapps",
            "filename": "within-apps-{app}-svm-rbf-dom-rted.sav",
            "embedding": "DOM_RTED"
        },
        {
            "name": "DOM_RTED",
            "setting": "acrossapps",
            "filename": "across-apps-{app}-svm-rbf-dom-rted.sav",
            "embedding": "DOM_RTED"
        },
        {
            "name": "VISUAL_PDiff",
            "setting": "withinapps",
            "filename": "within-apps-{app}-svm-rbf-visual-pdiff.sav",
            "embedding": "VISUAL_PDiff"
        },
        {
            "name": "VISUAL_PDiff",
            "setting": "acrossapps",
            "filename": "across-apps-{app}-svm-rbf-visual-pdiff.sav",
            "embedding": "VISUAL_PDiff"
        }
    ]

    results = []

    # -------------------------------------------------------
    #  Main loop: for each app & baseline classifier
    # -------------------------------------------------------
    for baseline in baseline_models_info:

        clf_name = baseline['name']
        setting  = baseline['setting']
        feature_col = baseline['embedding']

        for app in SELECTED_APPS:
            model_file = baseline['filename'].format(app=app)

            model_path = os.path.join(MODEL_DIR, model_file)
            if not os.path.exists(model_path):
                print(f"[Warning] Model file not found: {model_path}, {clf_name} for {app}.")
                exit(1)

            with open(model_path, 'rb') as f:
                clf = pickle.load(f)

            # Filter data for the current app
            df_app = df_merged[df_merged['appname'] == app].copy()
            if df_app.empty:
                print(f"[Warning] No data for app={app}. Skipping.")
                continue

            df_app['X'] = df_app[feature_col]
            X_app = df_app[['X']].values  # shape (N, 1)

            # True labels, ND type
            y_app = df_app['label'].values  # shape (N,)
            nd_app = df_app['nd_type'].values  # shape (N,)

            y_pred = clf.predict(X_app)
            analysis_counts = compute_nd_analysis(y_app, y_pred, nd_app)

            # Store for final results
            results.append({
                'App': app,
                'Method': clf_name,
                'Setting': setting,
                'Clones GT': analysis_counts['total_clones'],
                'Clones Pred': analysis_counts['correctly_pred_clones'],
                'ND2 GT': analysis_counts['total_nd2'],
                'ND2 Pred': analysis_counts['correctly_pred_nd2'],
                'ND3 GT': analysis_counts['total_nd3'],
                'ND3 Pred': analysis_counts['correctly_pred_nd3'],
                'Distinct GT': analysis_counts['total_distinct'],
                'Distinct Pred': analysis_counts['correctly_pred_distinct'],
            })

    # -------------------------------------------------------
    #  Save final results
    # -------------------------------------------------------
    df_results = pd.DataFrame(results)
    out_file = os.path.join(RESULTS_DIR, "rq1", "pair-analysis", "other_baseline_pair_analysis.xlsx")
    df_results.to_excel(out_file, index=False)
    print(f"\n[Info] Baseline dummy analysis complete. Results saved to: {out_file}")


if __name__ == "__main__":
    main()
