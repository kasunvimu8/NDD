import os
import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
sys.path.append("/Users/kasun/Documents/uni/semester-4/thesis/NDD")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from scripts.utils.embedding import run_embedding_pipeline_doc2vec, run_embedding_pipeline_bert,run_embedding_pipeline_markuplm
from scripts.utils.utils import (
    get_model,
    set_all_seeds,
    load_pairs_from_db,
    initialize_device,
)

class PairAnalysisDataset(Dataset):
    def __init__(self, pairs_data, state_embeddings):
        self.samples = []
        for record in pairs_data:
            app = record['appname']
            s1  = record['state1']
            s2  = record['state2']
            lbl = record['label']      # 0 or 1
            ndt = record['nd_type']    # 0 => clone, 2 => ND2, 3 => ND3

            key1 = (app, s1)
            key2 = (app, s2)

            if key1 not in state_embeddings or key2 not in state_embeddings:
                continue

            emb1 = state_embeddings[key1]
            emb2 = state_embeddings[key2]

            if isinstance(emb1, torch.Tensor):
                emb1 = emb1.clone().detach().float()
            else:
                emb1 = torch.tensor(emb1, dtype=torch.float32)

            if isinstance(emb2, torch.Tensor):
                emb2 = emb2.clone().detach().float()
            else:
                emb2 = torch.tensor(emb2, dtype=torch.float32)

            self.samples.append({
                'embeddings1': emb1,
                'embeddings2': emb2,
                'labels':      torch.tensor(lbl, dtype=torch.long),
                'nd_type':     ndt
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def prepare_pair_analysis_loader(pairs_data, state_embeddings, batch_size):
    dataset = PairAnalysisDataset(pairs_data, state_embeddings)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader

def get_embedding_and_dimension(
    embedding_type,
    app_dataset,
    app,
    title,
    chunk_size,
    overlap,
    chunk_limit,
    device,
    model_name,
    doc2vec_path,
    dom_root_dir,
    emb_dir
):
    cache_path = os.path.join(emb_dir, f"{title}_cache_{app}.pkl")

    if embedding_type == 'doc2vec':
        state_embeddings, final_input_dim = run_embedding_pipeline_doc2vec(
            pairs_data=app_dataset,
            dom_root_dir=dom_root_dir,
            doc2vec_model_path=doc2vec_path,
            cache_path=cache_path
        )

    elif embedding_type == 'bert' or embedding_type == 'modernbert':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModel.from_pretrained(model_name)
        state_embeddings, final_input_dim = run_embedding_pipeline_bert(
            tokenizer=tokenizer,
            bert_model=bert_model,
            pairs_data=app_dataset,
            dom_root_dir=dom_root_dir,
            chunk_size=chunk_size,
            overlap=overlap,
            device=device,
            chunk_threshold=chunk_limit,
            cache_path=cache_path
        )

    elif embedding_type == 'markuplm':
        state_embeddings, final_input_dim = run_embedding_pipeline_markuplm(
            pairs_data=app_dataset,
            dom_root_dir=dom_root_dir,
            chunk_size=chunk_size,
            overlap=overlap,
            device=device,
            markup_model_name=model_name,
            chunk_threshold=chunk_limit,
            cache_path=cache_path
        )
    else:
        raise ValueError(f"Embedding type '{embedding_type}' is not supported.")

    return state_embeddings, final_input_dim

def compute_nd_analysis(y_true, y_pred, nd_types):
    analysis_dict = {
        'total_clones': 0, 'correctly_pred_clones': 0,
        'total_nd2':    0, 'correctly_pred_nd2':    0,
        'total_nd3':    0, 'correctly_pred_nd3':    0,
        'total_distinct': 0, 'correctly_pred_distinct': 0
    }

    for i in range(len(y_true)):
        gt_label = y_true[i]
        pred_label = y_pred[i]
        nd_type_val = nd_types[i]

        if gt_label == 0:
            # Distinct
            analysis_dict['total_distinct'] += 1
            if pred_label == 0:
                analysis_dict['correctly_pred_distinct'] += 1

        elif gt_label == 1:
            # Near-duplicate
            if nd_type_val == 0:
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
            else:
                print(f"[Error] Unknown ND type: {nd_type_val}")
        else:
            print(f"[Error] Unknown label: {gt_label}")

    return analysis_dict

def run_test_and_collect_predictions(model, data_loader, device, setting, threshold=0.5):
    y_true_list = []
    y_pred_list = []
    nd_type_list = []

    with torch.no_grad():
        for batch in data_loader:
            emb1 = batch['embeddings1'].to(device)
            emb2 = batch['embeddings2'].to(device)
            labels = batch['labels'].to(device)  # 0 or 1
            batch_nd_type = batch['nd_type']

            if setting == 'contrastive':
                outputs = model(emb1, emb2)
                logits = outputs['logits'].squeeze(1)
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).long()
            else:
                # triplet
                out1 = model.forward_once(emb1)
                out2 = model.forward_once(emb2)
                distances = torch.nn.functional.pairwise_distance(out1, out2)
                preds = (distances <= threshold).long()

            y_true_list.extend(labels.cpu().tolist())
            y_pred_list.extend(preds.cpu().tolist())
            nd_type_list.extend(batch_nd_type)

    return y_true_list, y_pred_list, nd_type_list

if __name__ == "__main__":

    seed = 42
    set_all_seeds(seed)
    device = initialize_device()

    # Folders / Paths
    base_path = "/Users/kasun/Documents/uni/semester-4/thesis/NDD"
    table_name = "nearduplicates"
    db_path = f"{base_path}/dataset/SS_refined.db"
    dom_root_dir = f"{base_path}/resources/doms"
    results_dir = f"{base_path}/results"
    model_dir = f"{base_path}/models"
    emb_dir = f"{base_path}/embeddings"
    doc2vec_path = f"{base_path}/resources/embedding-models/content_tags_model_train_setsize300epoch50.doc2vec.model"

    # Global Config
    selected_apps = [
        'addressbook', 'claroline', 'ppma', 'mrbs',
        'mantisbt', 'dimeshift', 'pagekit', 'phoenix', 'petclinic'
    ]
    save_results = True

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
            'lr': 1e-04,
            'epochs': 15,
            'wd': 0.05,
            'bs': 32,
            'dimension': 300,

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
            'wd': 0.05,
            'bs': 32,
            'dimension': 768,
        },
        {
            'model_name': "answerdotai/ModernBERT-base",
            'title': "withinapp_modernbert",
            'embedding_type': "modernbert",
            'setting': "contrastive",
            'chunk_size': 8192,
            'overlap': 0,
            'chunk_limit': 5,
            'doc2vec_path': None,
            'lr': 1e-04,
            'epochs': 15,
            'wd': 0.05,
            'bs': 32,
            'dimension': 768,
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
            'wd': 0.05,
            'bs': 32,
            'dimension': 2304,
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
            'dimension': 300,
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
            'dimension': 768,
        },
        {
            'model_name': "answerdotai/ModernBERT-base",
            'title': "withinapp_modernbert",
            'embedding_type': "modernbert",
            'setting': "triplet",
            'chunk_size': 8192,
            'overlap': 0,
            'chunk_limit': 5,
            'doc2vec_path': None,
            'lr': 1e-03,
            'epochs': 15,
            'wd': 0.05,
            'bs': 32,
            'dimension': 768,
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
            'dimension': 2304,
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
            'dimension': 300,
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
            'dimension': 768,
        },
        {
            'model_name': "answerdotai/ModernBERT-base",
            'title': "acrossapp_modernbert",
            'embedding_type': "modernbert",
            'setting': "contrastive",
            'chunk_size': 8192,
            'overlap': 0,
            'chunk_limit': 2,
            'doc2vec_path': None,
            'lr': 2e-05,
            'epochs': 10,
            'wd': 0.01,
            'bs': 128,
            'dimension': 768,
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
            'dimension': 768,
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
            'dimension': 300,
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
            'dimension': 768,
        },
        {
            'model_name': "answerdotai/ModernBERT-base",
            'title': "acrossapp_modernbert",
            'embedding_type': "modernbert",
            'setting': "triplet",
            'chunk_size': 8192,
            'overlap': 0,
            'chunk_limit': 2,
            'doc2vec_path': None,
            'lr': 2e-05,
            'epochs': 15,
            'wd': 0.01,
            'bs': 128,
            'dimension': 768,
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
            'dimension': 768,
        },
    ]

    all_pairs = load_pairs_from_db(db_path, table_name, selected_apps)
    all_pairs_df = pd.DataFrame(all_pairs)

    results = []

    # ---------------------------
    #   Main Loop
    # ---------------------------
    for cfg in configurations:
        model_name   = cfg['model_name']
        setting      = cfg['setting']
        embedding_type = cfg['embedding_type']
        chunk_size   = cfg['chunk_size']
        chunk_limit  = cfg['chunk_limit']
        overlap      = cfg['overlap']
        trained_epochs = cfg['epochs']
        lr           = cfg['lr']
        title        = cfg['title']
        dimension    = cfg['dimension']
        weight_decay = cfg['wd']
        batch_size   = cfg['bs']

        # For each app
        for app in selected_apps:
            print(f"\n[Info] Processing App: {app} | Config: {title} ({setting}, {embedding_type})")

            # Subset the DF to get pairs for the current app
            app_dataset = all_pairs_df[all_pairs_df['appname'] == app].to_dict('records')
            if not app_dataset:
                print(f"[Warning] No data for app={app}. Skipping.")
                continue

            # ---------------------------
            #   1) Get embeddings
            # ---------------------------
            state_embeddings, final_input_dim = get_embedding_and_dimension(
                embedding_type=embedding_type,
                app_dataset=app_dataset,
                app=app,
                title=title,
                chunk_size=chunk_size,
                overlap=overlap,
                chunk_limit=chunk_limit,
                device=device,
                model_name=model_name,
                doc2vec_path=doc2vec_path,
                dom_root_dir=dom_root_dir,
                emb_dir=emb_dir
            )

            # ---------------------------
            #   2) Prepare DataLoader
            # ---------------------------
            data_loader = prepare_pair_analysis_loader(
                pairs_data=app_dataset,
                state_embeddings=state_embeddings,
                batch_size=batch_size
            )

            # ---------------------------
            #   3) Load SNN Model
            # ---------------------------
            model_path = os.path.join(
                model_dir,
                f"{title}_{setting}_{app}_cl_{chunk_limit}_bs_{batch_size}_ep_{trained_epochs}_lr_{lr}_wd_{weight_decay}.pt"
            )
            if not os.path.exists(model_path):
                print(f"[Warning] Model file not found at {model_path}. Skipping.")
                continue

            classification_model = get_model(
                model_path=model_path,
                setting=setting,
                device=device,
                dimension=final_input_dim
            )

            # ---------------------------
            #   4) Evaluate in Batches
            # ---------------------------
            y_true_list, y_pred_list, nd_type_list = run_test_and_collect_predictions(
                classification_model,
                data_loader,
                device,
                setting
            )
            analysis_counts = compute_nd_analysis(y_true_list, y_pred_list, nd_type_list)

            # Accumulate results
            results.append({
                'App': app,
                'Method': embedding_type+"_"+setting,
                'Setting': title.split("_")[0],
                'Embedding': embedding_type,
                'Clones GT': analysis_counts['total_clones'],
                'Clones Pred': analysis_counts['correctly_pred_clones'],
                'ND2 GT': analysis_counts['total_nd2'],
                'ND2 Pred': analysis_counts['correctly_pred_nd2'],
                'ND3 GT': analysis_counts['total_nd3'],
                'ND3 Pred': analysis_counts['correctly_pred_nd3'],
                'Distinct GT': analysis_counts['total_distinct'],
                'Distinct Pred': analysis_counts['correctly_pred_distinct'],
            })

    # ---------------------------
    #   Save Results
    # ---------------------------
    df = pd.DataFrame(results)
    output_file = os.path.join(results_dir, "rq1", "pair-analysis", "snn_pair_analysis_new.xlsx")
    df.to_excel(output_file, index=False)
    print(f"[Info] SNN pair analysis successfully completed. Results saved to: {output_file}")


