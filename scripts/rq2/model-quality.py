import csv
import json
import os
import sys
import pandas as pd
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModel
sys.path.append("/Users/kasun/Documents/uni/semester-4/thesis/NDD")

from scripts.utils.embedding import run_embedding_pipeline_bert, run_embedding_pipeline_doc2vec, run_embedding_pipeline_markuplm
from scripts.utils.networks import ContrastiveSiameseNN, TripletSiameseNN
from scripts.utils.utils import load_single_app_pairs_from_db, initialize_device

def is_duplicate_contrastive(model, state_embeddings, app, state1, state2, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        emb_vec1 = state_embeddings[(app, state1)]
        emb_vec2 = state_embeddings[(app, state2)]

        emb1 = emb_vec1.clone().detach().unsqueeze(0).to(device)
        emb2 = emb_vec2.clone().detach().unsqueeze(0).to(device)


        # Forward pass through the Siamese model
        outputs = model(emb1, emb2)
        logits  = outputs["logits"].squeeze(1)
        probs   = torch.sigmoid(logits)
        preds   = (probs > threshold).float()

        return bool(preds.item() == 1.0)

def is_duplicate_triplet(model, state_embeddings, app, state1, state2, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        emb_vec1 = state_embeddings[(app, state1)]
        emb_vec2 = state_embeddings[(app, state2)]

        emb1 = emb_vec1.clone().detach().unsqueeze(0).to(device)
        emb2 = emb_vec2.clone().detach().unsqueeze(0).to(device)

        out1 = model.forward_once(emb1)
        out2 = model.forward_once(emb2)

        distances = F.pairwise_distance(out1, out2)
        preds = (distances <= threshold).long().cpu().numpy()

        return bool(preds.item() == 1.0)

def get_model(model_path, setting, device, dimension):
    if setting == "contrastive":
        model = ContrastiveSiameseNN(input_dim=dimension)
    else:
        model = TripletSiameseNN(input_dim=dimension)

    model_state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()

    return model

def get_embedding(embedding_type, model_name, app_pairs, dom_root_dir, chunk_size, overlap, chunk_limit, emb_dir, app, device, title, doc2vec_path=None):
    if embedding_type == 'bert' or embedding_type == 'refinedweb':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModel.from_pretrained(model_name)
        state_embeddings, final_input_dim = run_embedding_pipeline_bert(
            tokenizer=tokenizer,
            bert_model=bert_model,
            pairs_data=app_pairs,
            dom_root_dir=dom_root_dir,
            chunk_size=chunk_size,
            overlap=overlap,
            device=device,
            chunk_threshold=chunk_limit,
            cache_path=os.path.join(emb_dir, f"{title}_cache_{app}.pkl")
        )
    elif embedding_type == 'markuplm':
        state_embeddings, final_input_dim = run_embedding_pipeline_markuplm(
            pairs_data=app_pairs,
            dom_root_dir=dom_root_dir,
            chunk_size=chunk_size,
            overlap=overlap,
            device=device,
            markup_model_name=model_name,
            chunk_threshold=chunk_limit,
            cache_path=os.path.join(emb_dir, f"{title}_cache_{app}.pkl")
        )
    elif embedding_type == 'doc2vec':
        if doc2vec_path is None:
            raise ValueError("For doc2vec, you must provide a doc2vec_model_path via the 'doc2vec_path' argument.")

        state_embeddings, final_input_dim = run_embedding_pipeline_doc2vec(
            pairs_data=app_pairs,
            dom_root_dir=dom_root_dir,
            doc2vec_model_path=doc2vec_path,
            cache_path=os.path.join(emb_dir, f"{title}_cache_{app}.pkl")
        )
    else:
        raise ValueError(f"Unknown embedding_type: {embedding_type}")

    return state_embeddings, final_input_dim


base_path = '/Users/kasun/Documents/uni/semester-4/thesis/NDD'
doc2vec_path = "/Users/kasun/Documents/uni/semester-4/thesis/NDD/resources/embedding-models/content_tags_model_train_setsize300epoch50.doc2vec.model"

APPS = [
    'addressbook', 'claroline', 'ppma', 'mrbs',
    'mantisbt', 'dimeshift', 'pagekit', 'phoenix', 'petclinic'
]

# All configuration for within app model quality assessment
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



OUTPUT_CSV = True
table_name   = "nearduplicates"
db_path      = f"{base_path}/dataset/SS_refined.db"
dom_root_dir = f"{base_path}/resources/doms"
emb_dir      = f"{base_path}/embeddings"


# -----------------------------------------------------------------------------
# RQ2 - Model Quality
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    device = initialize_device()

    for config in configurations:
        model_name, title, embedding_type, chunk_size, overlap, chunk_limit, doc2vec_path, setting, lr, epochs = (
            config[k] for k in ["model_name", "title", "embedding_type", "chunk_size", "overlap", "chunk_limit", "doc2vec_path", "setting", "lr", "epochs"]
        )

        print(f'\n======== Setting {setting}  Embedding : {embedding_type} ========')
        filename = f'{base_path}/results/rq2/{title}_{setting}_test_model_quality.csv'

        if OUTPUT_CSV and not os.path.exists(filename):
            header = ['Setting', 'App', 'Method', 'F1', 'Precision', 'Recall']
            with open(filename, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(header)

        for app in APPS:
                print(f'\n=== Coverage for App={app}  ===')

                model_path = f"{base_path}/models/{title}_{setting}_{app}_cl_{chunk_limit}_bs_128_ep_{epochs}_lr_{lr}_wd_0.01.pt"
                if not os.path.exists(model_path):
                    print(f"[Warning] Model file not found for {app} at {model_path}. Skipping.")
                    continue

                app_pairs = load_single_app_pairs_from_db(db_path, table_name, app)
                state_embeddings, final_input_dim = get_embedding(
                    embedding_type=embedding_type,
                    model_name=model_name,
                    app_pairs=app_pairs,
                    dom_root_dir=dom_root_dir,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    chunk_limit=chunk_limit,
                    emb_dir=emb_dir,
                    app=app,
                    device=device,
                    title=title,
                    doc2vec_path=doc2vec_path,
                )
                model = get_model(model_path, setting, device, final_input_dim)

                cluster_file_name = f'{base_path}/outputs/{app}.json'
                if not os.path.exists(cluster_file_name):
                    print(f"[Warning] No cluster file for {app} => {cluster_file_name}. Skipping coverage.")
                    continue

                with open(cluster_file_name, 'r') as f:
                    data = json.load(f)

                model_states = []  # states that are included in the "model"
                covered_bins = []
                number_of_bins = 0
                total_number_of_states = 0
                not_detected_near_duplicate_pairs = []
                all_comparison_pairs = []

                for bin_name, bin_states in data.items():
                    bin_index = 0
                    number_of_bins += 1
                    bin_covered = False

                    for state in bin_states:
                        total_number_of_states += 1

                        if len(model_states) == 0:
                            # If model empty, add the first state
                            model_states.append(state)
                            bin_covered = True
                        else:
                            is_distinct = True
                            if bin_index == 0:
                                # First state in the bin => compare with all existing states in the model
                                for ms in model_states:

                                    if setting == "contrastive":
                                        near_dup = is_duplicate_contrastive(model, state_embeddings, app, ms, state,device=device, threshold=0.5)
                                    else:
                                        near_dup = is_duplicate_triplet(model, state_embeddings, app, ms, state, device=device, threshold=0.5)

                                    if near_dup:  # near-duplicate
                                        is_distinct = False
                                        break

                                if is_distinct:
                                    model_states.append(state)
                                    bin_covered = True
                                bin_index += 1
                            else:
                                # For subsequent states, compare with the previous state in the bin
                                prev_state = bin_states[bin_index - 1]
                                if setting == "contrastive":
                                    near_dup = is_duplicate_contrastive(model, state_embeddings, app, prev_state, state, device=device, threshold=0.5)
                                else:
                                    near_dup = is_duplicate_triplet(model, state_embeddings, app, prev_state, state, device=device, threshold=0.5)

                                all_comparison_pairs.append((prev_state, state))

                                if not near_dup:
                                    # If not near-duplicate => distinct => add to model
                                    model_states.append(state)
                                    not_detected_near_duplicate_pairs.append((prev_state, state))
                                bin_index += 1

                    if bin_covered:
                        covered_bins.append(bin_name)

                # Compute Precision, Recall, F1
                unique_states_in_model = len(covered_bins)
                precision = unique_states_in_model / len(model_states) if len(model_states) > 0 else 0
                recall = len(covered_bins) / number_of_bins if number_of_bins > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                # Print results
                print(f"App: {app}")
                print(f"  Covered bins: {len(covered_bins)} / {number_of_bins}")
                print(f"  Number of states in model: {len(model_states)}")
                print(f"  Unique states in model (= covered bins): {unique_states_in_model}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall:    {recall:.4f}")
                print(f"  F1 Score:  {f1_score:.4f}")
                print(f"  # Not-detected near-duplicate pairs: {len(not_detected_near_duplicate_pairs)}")

                if OUTPUT_CSV:
                    with open(filename, 'a', encoding='UTF8') as f:
                        writer = csv.writer(f)
                        writer.writerow([title, app, 'SiameseNN', f1_score, precision, recall])

