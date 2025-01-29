import torch
from torch.backends import mps
import torch.optim as optim
import sys
sys.path.append("/Users/kasun/Documents/uni/semester-4/thesis/NDD")

from utils.utils_package  import (
    set_all_seeds,
    initialize_weights,
    save_results_to_excel,
    load_pairs_from_db,
    run_embedding_pipeline_markuplm,
    TripletSiameseNN,
    prepare_datasets_and_loaders_triplets,
    train_one_epoch_triplets,
    validate_model_triplets,
    test_model_triplets
)


##############################################################################
#      Main Functions  MarkupLM Triplet AcrossApp Classification             #
##############################################################################


if __name__ == "__main__":
    seed = 42
    set_all_seeds(seed)

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

    db_path       = "/Users/kasun/Documents/uni/semester-4/thesis/NDD/dataset/SS_refined.db"
    table_name    = "nearduplicates"
    dom_root_dir  = "/Users/kasun/Documents/uni/semester-4/thesis/NDD/resources/doms"
    results_dir   = "/Users/kasun/Documents/uni/semester-4/thesis/NDD/results"
    title         = "markuplm_acrossapp"
    setting_key   = "triplet"

    chunk_size    = 512
    batch_size    = 128
    num_epochs    = 7
    lr            = 2e-5
    weight_decay  = 0.01
    chunk_limit   = 1
    overlap       = 0
    margin        = 1

    results = []

    for test_app in selected_apps:

        print("\n=============================================")
        print(f"[Info] Starting across-app iteration: test_app = {test_app}")
        print("=============================================")

        all_pairs = load_pairs_from_db(db_path, table_name, selected_apps)
        if not all_pairs:
            print("[Warning] No data found in DB with is_retained=1. Skipping.")
            continue
        print(f"[Info] Total pairs: {len(all_pairs)}")

        state_embeddings, final_input_dim = run_embedding_pipeline_markuplm(
            pairs_data=all_pairs,
            dom_root_dir=dom_root_dir,
            chunk_size=chunk_size,
            overlap=overlap,
            device=device,
            markup_model_name="microsoft/markuplm-base",
            chunk_threshold=chunk_limit
        )
        if not state_embeddings or (final_input_dim == 0):
            print("[Warning] No embeddings found. Skipping.")
            continue

        train_loader, val_loader, test_loader = prepare_datasets_and_loaders_triplets(
            all_pairs,
            test_app=test_app,
            state_embeddings=state_embeddings,
            batch_size=batch_size,
            seed=seed
        )

        model = TripletSiameseNN(input_dim=final_input_dim)
        initialize_weights(model, seed)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            train_loss = train_one_epoch_triplets(
                model,
                train_loader,
                optimizer,
                device,
                epoch,
                num_epochs,
                margin=margin
            )

            val_loss = validate_model_triplets(model, val_loader, device, threshold=0.5)
            print(f"  Epoch {epoch+1}/{num_epochs} => Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        metrics_dict = test_model_triplets(model, test_loader, device, threshold=0.5)
        print(f"[Test Results] for test_app={test_app}: {metrics_dict}")

        row = {
            "TestApp": test_app,
            "Accuracy": metrics_dict["Accuracy"],
            "Precision": metrics_dict["Precision"],
            "Recall": metrics_dict["Recall"],
            "F1 Score (Weighted Avg)": metrics_dict["F1 Score (Weighted Avg)"],
            "F1_Class 0": metrics_dict["F1_Class 0"],
            "F1_Class 1": metrics_dict["F1_Class 1"]
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
