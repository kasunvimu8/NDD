import os
import sys
import time

import pandas as pd
import torch
from gensim.models import Doc2Vec
from transformers import AutoTokenizer, AutoModel, MarkupLMProcessor
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append("/Users/kasun/Documents/uni/semester-4/thesis/NDD")
from scripts.utils.utils import (
    get_model,
    set_all_seeds,
    load_pairs_from_db,
    initialize_device,
    saf_equals
)

if __name__ == "__main__":
    seed = 42
    set_all_seeds(seed)
    device = initialize_device()

    selected_apps = [
        'addressbook', 'claroline', 'ppma', 'mrbs',
        'mantisbt', 'dimeshift', 'pagekit', 'phoenix','petclinic'
    ]

    app_to_dim = {
        'withinapp_markuplm' : {
            'addressbook' : 2304,
            'claroline' : 768,
            'ppma' : 768,
            'mrbs' : 2304,
            'mantisbt' : 1536,
            'dimeshift' : 1536,
            'pagekit' : 768,
            'phoenix' : 768,
            'petclinic' : 768,
        },
        'acrossapp_markuplm' : {
            'addressbook' : 768,
            'claroline' : 768,
            'ppma' : 768,
            'mrbs' : 768,
            'mantisbt' : 768,
            'dimeshift' : 768,
            'pagekit' : 768,
            'phoenix' : 768,
            'petclinic' : 768,
        },
    }

    base_path       = "/Users/kasun/Documents/uni/semester-4/thesis/NDD"
    table_name      = "nearduplicates"
    db_path         = f"{base_path}/dataset/SS_refined.db"
    dom_root_dir    = f"{base_path}/resources/doms"
    results_dir     = f"{base_path}/results"
    model_dir       = f"{base_path}/models"
    emb_dir         = f"{base_path}/embeddings"
    doc2vec_path    = f"{base_path}/resources/embedding-models/content_tags_model_train_setsize300epoch50.doc2vec.model"
    save_results    = True

    weight_decay  = 0.01
    sample_size   = 1000

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
            'embedding_type': "bert",
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
            'embedding_type': "bert",
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
            'embedding_type': "bert",
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
            'embedding_type': "bert",
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

    results = []
    all_pairs = load_pairs_from_db(db_path, table_name, selected_apps)
    all_pairs_df = pd.DataFrame(all_pairs)
    
    SS_sampled = all_pairs_df.sample(n=sample_size, random_state=42)

    # class distribution of the sampled datset
    print("\nClass distribution of the sample :")
    print(SS_sampled['appname'].value_counts())
    
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
        bs               = current_config['bs']
        wd               = current_config['wd']

        # Preparing required model
        embedding_model = None
        tokenizer = None
        processor = None

        if embedding_type == 'doc2vec':
            embedding_model = Doc2Vec.load(doc2vec_path)
            embedding_model.random.seed(42)  # fix seed if needed

        elif embedding_type == 'bert':
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            embedding_model = AutoModel.from_pretrained(model_name)
            embedding_model.to(device)

        elif embedding_type == 'markuplm':
            processor = MarkupLMProcessor.from_pretrained(model_name)
            processor.parse_html = False
            embedding_model = AutoModel.from_pretrained(model_name)
            embedding_model.to(device)
        else:
            print(f"[Error] Unknown embedding type {embedding_type}. Skipping.")
            sys.exit(1)

        app_time = {app : 0 for app in selected_apps}
        
        def load_model_and_tokenizer(dimension, app):
            model_path = f"{base_path}/models/{title}_{setting}_{app}_cl_{chunk_limit}_bs_{bs}_ep_{trained_epochs}_lr_{lr}_wd_{wd}.pt"

            if not os.path.exists(model_path):
                print(f"[Warning] Model file not found at {model_path}. Skipping.")
                sys.exit(1)


            classification_model = get_model(model_path, setting, device, dimension)
            classification_model.to(device)

            model_state = torch.load(model_path, map_location=device, weights_only=True)
            classification_model.load_state_dict(model_state, strict=True)
            classification_model.eval()

            return classification_model

        for app in selected_apps:
            print(f"[Info] Processing App: {app} | Config: {title} ({setting}, {embedding_type})")

            if embedding_type == 'markuplm':
                dimension = app_to_dim[title][app]

            total_time = 0
            classification_model = load_model_and_tokenizer(dimension, app)
            
            app_dataset = SS_sampled[SS_sampled['appname'] == app]
            
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
                    
                start_time = time.perf_counter()
                result = saf_equals(
                    dom1=dom_content_1,
                    dom2=dom_content_2,
                    classification_model=classification_model,
                    embedding_model=embedding_model,
                    processor=processor,
                    tokenizer=tokenizer,
                    embedding_type=embedding_type,
                    setting=setting,
                    device=device,
                    chunk_size=chunk_size,
                    dimension=dimension,
                    overlap= overlap,
                    threshold=0.5
                )
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                total_time += elapsed_time
            app_time[app] = total_time / len(app_dataset) if len(app_dataset) > 0 else 0
        
        for app in selected_apps:
            results.append({
                'App': app,
                'Title': title,
                'Setting': setting,
                'Embedding': embedding_type,
                'Inference Time (s)': app_time[app]
            })

            
    df = pd.DataFrame(results)
    df.to_excel(f"{results_dir}/rq4/snn_inference_times_new.xlsx", index=False)
    print(f"\nSNN inference times saved")
        
        
