import os
import sys
import time

import pandas as pd
import torch
from tqdm import tqdm
sys.path.append(r"D:\\Acadamic\\4-Semester-2425-Winter\\thesis\\codes\\NDD")
from scripts.utils.utils import (
    get_model,
    saf_equals,
    set_all_seeds,
    load_pairs_from_db,
    initialize_device
)

if __name__ == "__main__":
    seed = 42
    set_all_seeds(seed)
    device = initialize_device()

    selected_apps = [
        'addressbook', 'claroline', 'ppma', 'mrbs',
        'mantisbt', 'dimeshift', 'pagekit', 'phoenix','petclinic'
    ]

    base_path       = r"D:\\Acadamic\\4-Semester-2425-Winter\\thesis\\codes\\NDD"
    table_name      = "nearduplicates"
    db_path         = f"{base_path}\\dataset\\SS_refined.db"
    dom_root_dir    = f"{base_path}\\resources\\doms"
    results_dir     = f"{base_path}\\results"
    model_dir       = f"{base_path}\\models"
    emb_dir         = f"{base_path}\\embeddings"
    doc2vec_path    = f"{base_path}\\resources\\embedding-models\\content_tags_model_train_setsize300epoch50.doc2vec.model"
    save_results    = True

    batch_size    = 128
    weight_decay  = 0.01
    
    # Configurations
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
            'dimension' : 768,
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
            'dimension' : 768,
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
            'dimension' : 300,
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
            'dimension' : 300,
        },
        # {
        #     'model_name': "microsoft/markuplm-base",
        #     'title': "withinapp_markuplm",
        #     'embedding_type': "markuplm",
        #     'setting': "contrastive",
        #     'chunk_size': 512,
        #     'overlap': 0,
        #     'chunk_limit': 5,
        #     'doc2vec_path': None,
        #     'lr' : 5e-05,
        #     'epochs': 50,
        #     'dimension' : 2304,
        # },
        # {
        #     'model_name': "microsoft/markuplm-base",
        #     'title': "withinapp_markuplm",
        #     'embedding_type': "markuplm",
        #     'setting': "triplet",
        #     'chunk_size': 512,
        #     'overlap': 0,
        #     'chunk_limit': 5,
        #     'doc2vec_path': None,
        #     'lr' : 0.0005,
        #     'epochs': 50,
        #     'dimension' : 768,
        # },
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
            'dimension' : 768,
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
            'dimension' : 768,
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
            'dimension' : 300,
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
            'dimension' : 300,
        },
        # {
        #     'model_name': "microsoft/markuplm-base",
        #     'title': "acrossapp_markuplm",
        #     'embedding_type': "markuplm",
        #     'setting': "contrastive",
        #     'chunk_size': 512,
        #     'overlap': 0,
        #     'chunk_limit': 1,
        #     'doc2vec_path': None,
        #     'lr' : 2e-05,
        #     'epochs': 15,
        #     'dimension' : 768,
        # },
        # {
        #     'model_name': "microsoft/markuplm-base",
        #     'title': "acrossapp_markuplm",
        #     'embedding_type': "markuplm",
        #     'setting': "triplet",
        #     'chunk_size': 512,
        #     'overlap': 0,
        #     'chunk_limit': 1,
        #     'doc2vec_path': None,
        #     'lr' : 2e-05,
        #     'epochs': 12,
        #     'dimension' : 768,
        # },
    ]

    results = []
    all_pairs = load_pairs_from_db(db_path, table_name, selected_apps)
    all_pairs_df = pd.DataFrame(all_pairs)
    
    SS_sampled = all_pairs_df.sample(n=10000, random_state=42)

    # class distribution of the sampled datset
    print("\nClass distribution of the sample :")
    print(SS_sampled['appname'].value_counts())
    
    results = []
    
    for current_config in configurations:
        model_name       = current_config['model_name']
        setting          = current_config['setting']
        embedding_type   = current_config['embedding_type']
        chunk_size       = current_config['chunk_size']
        chunk_limit      = current_config['chunk_limit']
        overlap          = current_config['overlap']
        trained_epochs   = current_config['epochs']
        lr               = current_config['lr']
        title            = current_config['title']
        dimension        = current_config['dimension']
        
        app_time = {app : 0 for app in selected_apps} # store the avg time taken for each app
        
        def load_model_and_tokenizer(dimension, app):
            model_path = f"{base_path}/models/{title}_{setting}_{app}_cl_{chunk_limit}_bs_128_ep_{trained_epochs}_lr_{lr}_wd_0.01.pt"

            if not os.path.exists(model_path):
                print(f"[Warning] Model file not found at {model_path}. Skipping.")
                sys.exit(1)


            classification_model = get_model(model_path, setting, device, dimension)
            classification_model.to(device)

            model_state = torch.load(model_path, map_location=device, weights_only=True)
            classification_model.load_state_dict(model_state, strict=True)
            classification_model.eval()

            return classification_model

        for app in tqdm(selected_apps, desc=f"{title} ({setting}-{embedding_type})", leave=False):
            print(f"[Info] Processing App: {app} | Config: {title} ({setting}, {embedding_type})")
            
            total_time = 0
            classification_model = load_model_and_tokenizer(dimension, app)
            
            app_dataset = SS_sampled[SS_sampled['appname'] == app]
            
            state_1 = app_dataset['state1'] 
            state_2 = app_dataset['state2'] 
        
            dom_path_1 = os.path.join(dom_root_dir, app, 'doms', f"{state_1}.html")
            dom_path_2 = os.path.join(dom_root_dir, app, 'doms', f"{state_2}.html")
            
            print(dom_path_1)
            
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
                
            start_time = time.perf_counter()
            result = saf_equals(dom_content_1, dom_content_2, classification_model, model_name, embedding_type, setting, device, doc2vec_path, chunk_size, dimensions, overlap, threshold=0.5)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
        app_time[app] = total_time / len(app_dataset)
        
        for app in selected_apps:
            results.append({
                'App': app,
                'Title': title,
                'Setting': setting,
                'Embedding': embedding_type,
                'Inference Time (s)': app_time[app]
            })

            
    output_file = f"{results_dir}\\rq4\\inference_times.xlsx"
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"\nResults saved to {output_file}")
        
        
