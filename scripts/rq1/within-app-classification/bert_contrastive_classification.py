import os
import torch
import torch.optim as optim
import sys
import time
from transformers import AutoTokenizer, AutoModel
sys.path.append("/Users/kasun/Documents/uni/semester-4/thesis/NDD")

from scripts.rq1.datasets import prepare_datasets_and_loaders_within_app_contrastive
from scripts.utils.embedding import run_embedding_pipeline_bert
from scripts.rq1.test import test_model_contrastive
from scripts.rq1.train import train_one_epoch_contrastive
from scripts.rq1.validate import validate_model_contrastive
from scripts.utils.networks import ContrastiveSiameseNN
from scripts.utils.utils import (
    set_all_seeds,
    initialize_weights,
    save_results_to_excel,
    load_single_app_pairs_from_db,
    initialize_device
)

##############################################################################
#     Main Script: BERT Contrastive Within-App Classification                     #
##############################################################################

if __name__ == "__main__":
    seed = 42
    set_all_seeds(seed)
    device = initialize_device()

    selected_apps = [
        'addressbook', 'claroline', 'ppma', 'mrbs',
        'mantisbt', 'dimeshift', 'pagekit', 'phoenix','petclinic'
    ]

    base_path    = "/Users/kasun/Documents/uni/semester-4/thesis/NDD"
    table_name   = "nearduplicates"
    db_path      = f"{base_path}/dataset/SS_refined.db"
    dom_root_dir = f"{base_path}/resources/doms"
    results_dir  = f"{base_path}/results"
    model_dir    = f"{base_path}/models"
    emb_dir      = f"{base_path}/embeddings"
    setting_key  = "contrastive"
    save_results = True

    batch_size   = 32
    num_epochs   = 15
    lr           = 1e-4
    weight_decay = 0.05
    chunk_limit  = 5
    overlap      = 0

    #for bert-base uncomment below
    #model_name    = "bert-base-uncased"
    #chunk_size    = 512
    #title         = "withinapp_bert"

    #for modern-bert-base uncomment below
    title           = "withinapp_modernbert"
    model_name      = "answerdotai/ModernBERT-base"
    chunk_size      = 8192

    results = []

    for app in selected_apps:
        print("\n=============================================")
        print(f"[Info] Starting within-app classification for: {app}")
        print("=============================================")
        preprocess_start_time = time.time()

        model_filename = f"{title}_{setting_key}_{app}_cl_{chunk_limit}_bs_{batch_size}_ep_{num_epochs}_lr_{lr}_wd_{weight_decay}.pt"
        model_file = os.path.join(model_dir, model_filename)

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
            chunk_threshold=chunk_limit,
            cache_path=os.path.join(emb_dir ,f"{title}_cache_{app}.pkl")
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
        preprocess_time = time.time() - preprocess_start_time

        if os.path.exists(model_file):
            print(f"[Info] Found saved model for {app} at {model_file}. Loading model and skipping training.")
            model.load_state_dict(torch.load(model_file, weights_only=True))
            training_time = "N/A"
        else:
            print(f"[Info] No saved model for {app}. Training will start.")
            start_time = time.time()
            for epoch in range(num_epochs):
                train_loss = train_one_epoch_contrastive(model, train_loader, optimizer, device, epoch, num_epochs)
                val_loss   = validate_model_contrastive(model, val_loader, device)
                print(f"  Epoch {epoch+1}/{num_epochs} => Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            training_time = time.time() - start_time
            torch.save(model.state_dict(), model_file)
            print(f"[Info] Model saved to {model_file}")

        metrics_dict = test_model_contrastive(model, test_loader, device, threshold=0.5)
        print(f"[Test Results] for app={app}: {metrics_dict}")

        row = {
            "TestApp": app,
            "Accuracy": metrics_dict["Accuracy"],
            "Precision": metrics_dict["Precision"],
            "Recall": metrics_dict["Recall"],
            "F1_Class 0": metrics_dict["F1_Class 0"],
            "F1_Class 1": metrics_dict["F1_Class 1"],
            "F1 Score (Weighted Avg)": metrics_dict["F1 Score (Weighted Avg)"],
            "TrainingTime": training_time,
            "PreprocessingTime": preprocess_time,
        }
        results.append(row)

    if save_results :
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
