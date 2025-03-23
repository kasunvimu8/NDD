import os
import json
import sqlite3
import pandas as pd
import numpy as np

BASE_PATH   = "/Users/kasun/Documents/uni/semester-4/thesis/NDD"
DB_PATH     = f"{BASE_PATH}/dataset/SS_refined.db"
TABLE_NAME  = "nearduplicates"
JSON_PATH   = f"{BASE_PATH}/resources/baseline-dataset/combinedEntries.json"
RESULTS_DIR = f"{BASE_PATH}/results"

SELECTED_APPS = [
    'addressbook', 'claroline', 'ppma', 'mrbs',
    'mantisbt', 'dimeshift', 'pagekit', 'phoenix', 'petclinic'
]


# --------------------------------------------------------------------------
#  1) Load only retained records from DB
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
#  2) The ND analysis function (binary label + nd_type)
# --------------------------------------------------------------------------
def compute_nd_analysis(y_true, y_pred, nd_types):
    """
    - y_true: list of ground-truth labels (0=distinct, 1=near-duplicate)
    - y_pred: list of predicted labels (0=distinct, 1=near-duplicate)
    - nd_types: list of ND categories in {0=clone, 2=ND2, 3=ND3} for near-duplicate pairs
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
        nd_type_val = nd_types[i]  # 0,2,3

        if gt_label == 0:
            # Distinct
            analysis_dict['total_distinct'] += 1
            if pred_label == 0:
                analysis_dict['correctly_pred_distinct'] += 1
        elif gt_label == 1:
            # Near-duplicate
            if nd_type_val == 0:
                # clone
                analysis_dict['total_clones'] += 1
                if pred_label == 1:
                    analysis_dict['correctly_pred_clones'] += 1
            elif nd_type_val == 2:
                analysis_dict['total_nd2'] += 1
                if pred_label == 1:
                    analysis_dict['correctly_pred_nd2'] += 1
            elif nd_type_val == 3:
                analysis_dict['total_nd3'] += 1
                if pred_label == 1:
                    analysis_dict['correctly_pred_nd3'] += 1

    return analysis_dict

# --------------------------------------------------------------------------
#  3) Main logic
# --------------------------------------------------------------------------
def main():
    # -------------------- A) Read DB + JSON --------------------------------
    df_db_retained = get_db_retained(DB_PATH, TABLE_NAME)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        all_entries = json.load(f)

    columns = [
        "appname",  # idx=0
        "col1",  # idx=1
        "state1",  # idx=2
        "state2",  # idx=3
        "col4", "col5", "col6", "col7", "col8", "col9", "col10", "col11", "col12", "col13",
        "y_actual",  # idx=14
        "col15",
        "frag_pred",  # idx=16 (the “fragged” classification)
        "col17"
    ]
    df_json = pd.DataFrame(all_entries, columns=columns)

    # -------------------- B) Merge DB + JSON ------------------
    # We only keep is_retained=1 pairs from the DB
    merge_cols = ["appname", "state1", "state2"]
    df_merged = pd.merge(df_db_retained, df_json, on=merge_cols, how='left',suffixes=('_db', ''))

    if df_merged.empty:
        print("No matching retained entries found. Exiting.")
        return

    # Dropping mrbs rows as it is not evaluated with fraggen
    df_merged = df_merged[df_merged['appname'] != 'mrbs']

    def map_db_label(x):
        return 1 if x in [0, 1] else 0  # 0 or 1 => near-dupe(1), 2 => distinct(0)

    df_merged['label'] = df_merged['HUMAN_CLASSIFICATION'].apply(map_db_label)

    def map_frag_label(x):
        # 0 => clone => near-dupe(1),
        # 1 => ND => near-dupe(1),
        # 2 => distinct => 0
        if x in [0, 1]:
            return 1
        else:
            return 0

    df_merged['fraggen_label'] = df_merged['frag_pred'].apply(map_frag_label)

    # -------------------- C) Per-app ND Analysis ---------------------------
    results = []

    for app in SELECTED_APPS:
        df_app = df_merged[df_merged['appname'] == app].copy()
        if df_app.empty:
            print(f"[Warning] No data for app={app}. Skipping.")
            continue

        y_true = df_app['label'].values  # 0 or 1
        y_pred = df_app['fraggen_label'].values
        nd_app = df_app['nd_type'].values  # 0,2,3

        analysis_counts = compute_nd_analysis(y_true, y_pred, nd_app)
        results.append({
            'App': app,
            'Method': 'Fraggen',
            'Setting': 'withinapps',
            'Clones GT': analysis_counts['total_clones'],
            'Clones Pred': analysis_counts['correctly_pred_clones'],
            'ND2 GT': analysis_counts['total_nd2'],
            'ND2 Pred': analysis_counts['correctly_pred_nd2'],
            'ND3 GT': analysis_counts['total_nd3'],
            'ND3 Pred': analysis_counts['correctly_pred_nd3'],
            'Distinct GT': analysis_counts['total_distinct'],
            'Distinct Pred': analysis_counts['correctly_pred_distinct']
        })

    if not results:
        print("No per-app results found. Exiting.")
        return

    # -------------------- D) Save Results to Excel -------------------------

    df_results = pd.DataFrame(results)
    out_file = os.path.join(RESULTS_DIR, "rq1", "pair-analysis", "fraggen_baseline_pair_analysis.xlsx")
    df_results.to_excel(out_file, index=False)
    print(f"\n[Info] Baseline analysis complete. Results saved to: {out_file}")

if __name__ == "__main__":
    main()
