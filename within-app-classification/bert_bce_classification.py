import torch
import torch.optim as optim
from torch.backends import mps
import sys
from transformers import AutoTokenizer, AutoModel
sys.path.append("/Users/kasun/Documents/uni/semester-4/thesis/NDD")

from scripts.datasets import prepare_datasets_and_loaders_within_app_contrastive
from scripts.embedding import run_embedding_pipeline_bert
from scripts.test import test_model_contrastive
from scripts.train import train_one_epoch_contrastive
from scripts.validate import validate_model_contrastive
from scripts.networks import ContrastiveSiameseNN
from scripts.utils import (
    set_all_seeds,
    initialize_weights,
    save_results_to_excel,
    load_single_app_pairs_from_db
)

##############################################################################
#     Main Script: BERT BCE Within-App Classification                     #
##############################################################################

if __name__ == "__main__":
    seed = 42
    set_all_seeds(seed)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("[Info] Using device:", device)

    selected_apps = [
        'addressbook', 'claroline', 'ppma', 'mrbs',
        'mantisbt', 'dimeshift', 'pagekit', 'phoenix','petclinic'
    ]

    db_path      = "/Users/kasun/Documents/uni/semester-4/thesis/NDD/dataset/SS_refined.db"
    table_name   = "nearduplicates"
    dom_root_dir = "/Users/kasun/Documents/uni/semester-4/thesis/NDD/resources/doms"
    results_dir  = "/Users/kasun/Documents/uni/semester-4/thesis/NDD/results"
    title        = "bert-base_withinapp"
    setting_key  = "bce-balanced"
    model_name   = "bert-base-uncased"

    chunk_size   = 512
    batch_size   = 128
    num_epochs   = 30
    lr           = 5e-5
    weight_decay = 0.01
    chunk_limit  = 5
    overlap      = 0

    results = []

    for app in selected_apps:
        print("\n=============================================")
        print(f"[Info] Starting within-app classification for: {app}")
        print("=============================================")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModel.from_pretrained(model_name)

        app_pairs = load_single_app_pairs_from_db(db_path, table_name, app)
        if not app_pairs:
            print(f"[Warning] No data found for app={app}. Skipping.")
            break
        print(f"[Info] Total pairs in DB (retained=1) for {app}: {len(app_pairs)}")

        state_embeddings, final_input_dim = run_embedding_pipeline_bert(
            tokenizer=tokenizer,
            bert_model=bert_model,
            pairs_data=app_pairs,
            dom_root_dir=dom_root_dir,
            chunk_size=chunk_size,
            overlap=overlap,
            device=device,
            chunk_threshold=chunk_limit
        )

        if not state_embeddings or (final_input_dim == 0):
            print("[Warning] No embeddings found. Skipping.")
            continue

        train_loader, val_loader, test_loader = prepare_datasets_and_loaders_within_app_contrastive(
            app_pairs=app_pairs,
            state_embeddings=state_embeddings,
            batch_size=batch_size,
            seed=seed,
            train_ratio=0.8,
            val_ratio=0.1
        )
        if not train_loader or not val_loader or not test_loader:
            print("[Warning] Data split invalid or empty. Skipping.")
            continue

        model = ContrastiveSiameseNN(input_dim=final_input_dim)
        initialize_weights(model, seed)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            train_loss = train_one_epoch_contrastive(model, train_loader, optimizer, device, epoch, num_epochs)
            val_loss   = validate_model_contrastive(model, val_loader, device)
            print(f"  Epoch {epoch+1}/{num_epochs} => Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        metrics_dict = test_model_contrastive(model, test_loader, device, threshold=0.5)
        print(f"[Test Results] for app={app}: {metrics_dict}")

        row = {
            "TestApp": app,
            "Accuracy": metrics_dict["Accuracy"],
            "Precision": metrics_dict["Precision"],
            "Recall": metrics_dict["Recall"],
            "F1_Class 0": metrics_dict["F1_Class 0"],
            "F1_Class 1": metrics_dict["F1_Class 1"],
            "F1 Score (Weighted Avg)": metrics_dict["F1 Score (Weighted Avg)"]
        }
        results.append(row)

    save_results_to_excel(
        title=title,
        results=results,
        results_dir=results_dir,
        setting_key=setting_key,
        overlap=overlap,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        chunk_limit=chunk_limit
    )
