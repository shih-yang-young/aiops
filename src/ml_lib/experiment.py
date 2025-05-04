from dataset import CustomDataset
from train_eval import train_epoch, eval_model
from hybrid_model import HybridLstmClassifier, HybridCnnClassifier, HybridBilstmClassifier
from cnn_model import HybridCnn1Filter234Drop3Classifier, HybridCnn1Filter234Drop5Classifier, HybridCnn1Filter345Drop3Classifier, HybridCnn1Filter345Drop5Classifier, HybridCnn2Filter234Drop3Classifier, HybridCnn2Filter234Drop5Classifier, HybridCnn2Filter345Drop3Classifier, HybridCnn2Filter345Drop5Classifier

from lstm_model import Lstm1Layer256Hidden3Dropout, Lstm1Layer256Hidden5Dropout, Lstm1Layer512Hidden3Dropout, Lstm1Layer512Hidden5Dropout, Lstm2Layer256Hidden3Dropout, Lstm2Layer256Hidden5Dropout, Lstm2Layer512Hidden3Dropout, Lstm2Layer512Hidden5Dropout, Lstm3Layer256Hidden3Dropout, Lstm3Layer256Hidden5Dropout, Lstm3Layer512Hidden3Dropout, Lstm3Layer512Hidden5Dropout

from bilstm_model import Bilstm1Layer256Hidden3Dropout, Bilstm1Layer256Hidden5Dropout, Bilstm1Layer512Hidden3Dropout, Bilstm1Layer512Hidden5Dropout, Bilstm2Layer256Hidden3Dropout, Bilstm2Layer256Hidden5Dropout, Bilstm2Layer512Hidden3Dropout, Bilstm2Layer512Hidden5Dropout, Bilstm3Layer256Hidden3Dropout, Bilstm3Layer256Hidden5Dropout, Bilstm3Layer512Hidden3Dropout, Bilstm3Layer512Hidden5Dropout

from resampling import apply_resampling
from device import get_device_info
from transformers import BertTokenizer, RobertaTokenizer, DebertaTokenizer, AutoModelForSequenceClassification
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import torch
import time

def get_tokenizer(model_name):
    if "bert-base-uncased" in model_name.lower():
        print("tokenizer is bert-base-uncased")
        return BertTokenizer.from_pretrained(model_name)
    elif "roberta-base" in model_name.lower():
        print("tokenizer is roberta-base")
        return RobertaTokenizer.from_pretrained(model_name)
    elif "microsoft/deberta-base" in model_name.lower():
        print("tokenizer is microsoft/deberta-base")
        return DebertaTokenizer.from_pretrained(model_name)
    else:
        raise ValueError("Unsupported model tokenizer.")

def get_model(model_name, hybrid=None, num_labels=None):
    if hybrid is None:
        print("model is", model_name)
        return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    elif hybrid == "lstm":
        print("model is", model_name, hybrid)
        return HybridLstmClassifier(model_name, num_labels)
    elif hybrid == "bilstm":
        print("model is", model_name, hybrid)
        return HybridBilstmClassifier(model_name, num_labels)
    elif hybrid == "cnn":
        print("model is", model_name, hybrid)
        return HybridCnnClassifier(model_name, num_labels)
    elif hybrid == "cnn1filter234drop3":
        print("model is", model_name, hybrid)
        return HybridCnn1Filter234Drop3Classifier(model_name, num_labels)
    elif hybrid == "cnn1filter234drop5":
        print("model is", model_name, hybrid)
        return HybridCnn1Filter234Drop5Classifier(model_name, num_labels)
    elif hybrid == "cnn1filter345drop3":
        print("model is", model_name, hybrid)
        return HybridCnn1Filter345Drop3Classifier(model_name, num_labels)
    elif hybrid == "cnn1filter345drop5":
        print("model is", model_name, hybrid)
        return HybridCnn1Filter345Drop5Classifier(model_name, num_labels)
    elif hybrid == "cnn2filter234drop3":
        print("model is", model_name, hybrid)
        return HybridCnn2Filter234Drop3Classifier(model_name, num_labels)
    elif hybrid == "cnn2filter234drop5":
        print("model is", model_name, hybrid)
        return HybridCnn2Filter234Drop5Classifier(model_name, num_labels)
    elif hybrid == "cnn2filter345drop3":
        print("model is", model_name, hybrid)
        return HybridCnn2Filter345Drop3Classifier(model_name, num_labels)
    elif hybrid == "cnn2filter345drop5":
        print("model is", model_name, hybrid)
        return HybridCnn2Filter345Drop5Classifier(model_name, num_labels)
    elif hybrid == "Lstm1Layer256Hidden3Dropout":
        print("model is", model_name, hybrid)
        return Lstm1Layer256Hidden3Dropout(model_name, num_labels)
    elif hybrid == "Lstm1Layer256Hidden5Dropout":
        print("model is", model_name, hybrid)
        return Lstm1Layer256Hidden5Dropout(model_name, num_labels)
    elif hybrid == "Lstm1Layer512Hidden3Dropout":
        print("model is", model_name, hybrid)
        return Lstm1Layer512Hidden3Dropout(model_name, num_labels)
    elif hybrid == "Lstm1Layer512Hidden5Dropout":
        print("model is", model_name, hybrid)
        return Lstm1Layer512Hidden5Dropout(model_name, num_labels)
    elif hybrid == "Lstm2Layer256Hidden3Dropout":
        print("model is", model_name, hybrid)
        return Lstm2Layer256Hidden3Dropout(model_name, num_labels)
    elif hybrid == "Lstm2Layer256Hidden5Dropout":
        print("model is", model_name, hybrid)
        return Lstm2Layer256Hidden5Dropout(model_name, num_labels)
    elif hybrid == "Lstm2Layer512Hidden3Dropout":
        print("model is", model_name, hybrid)
        return Lstm2Layer512Hidden3Dropout(model_name, num_labels)
    elif hybrid == "Lstm2Layer512Hidden5Dropout":
        print("model is", model_name, hybrid)
        return Lstm2Layer512Hidden5Dropout(model_name, num_labels)
    elif hybrid == "Lstm3Layer256Hidden3Dropout":
        print("model is", model_name, hybrid)
        return Lstm3Layer256Hidden3Dropout(model_name, num_labels)
    elif hybrid == "Lstm3Layer256Hidden5Dropout":
        print("model is", model_name, hybrid)
        return Lstm3Layer256Hidden5Dropout(model_name, num_labels)
    elif hybrid == "Lstm3Layer512Hidden3Dropout":
        print("model is", model_name, hybrid)
        return Lstm3Layer512Hidden3Dropout(model_name, num_labels)
    elif hybrid == "Lstm3Layer512Hidden5Dropout":
        print("model is", model_name, hybrid)
        return Lstm3Layer512Hidden5Dropout(model_name, num_labels)
    elif hybrid == "Bilstm1Layer256Hidden3Dropout":
        print("model is", model_name, hybrid)
        return Bilstm1Layer256Hidden3Dropout(model_name, num_labels)
    elif hybrid == "Bilstm1Layer256Hidden5Dropout":
        print("model is", model_name, hybrid)
        return Bilstm1Layer256Hidden5Dropout(model_name, num_labels)
    elif hybrid == "Bilstm1Layer512Hidden3Dropout":
        print("model is", model_name, hybrid)
        return Bilstm1Layer512Hidden3Dropout(model_name, num_labels)
    elif hybrid == "Bilstm1Layer512Hidden5Dropout":
        print("model is", model_name, hybrid)
        return Bilstm1Layer512Hidden5Dropout(model_name, num_labels)
    elif hybrid == "Bilstm2Layer256Hidden3Dropout":
        print("model is", model_name, hybrid)
        return Bilstm2Layer256Hidden3Dropout(model_name, num_labels)
    elif hybrid == "Bilstm2Layer256Hidden5Dropout":
        print("model is", model_name, hybrid)
        return Bilstm2Layer256Hidden5Dropout(model_name, num_labels)
    elif hybrid == "Bilstm2Layer512Hidden3Dropout":
        print("model is", model_name, hybrid)
        return Bilstm2Layer512Hidden3Dropout(model_name, num_labels)
    elif hybrid == "Bilstm2Layer512Hidden5Dropout":
        print("model is", model_name, hybrid)
        return Bilstm2Layer512Hidden5Dropout(model_name, num_labels)
    elif hybrid == "Bilstm3Layer256Hidden3Dropout":
        print("model is", model_name, hybrid)
        return Bilstm3Layer256Hidden3Dropout(model_name, num_labels)
    elif hybrid == "Bilstm3Layer256Hidden5Dropout":
        print("model is", model_name, hybrid)
        return Bilstm3Layer256Hidden5Dropout(model_name, num_labels)
    elif hybrid == "Bilstm3Layer512Hidden3Dropout":
        print("model is", model_name, hybrid)
        return Bilstm3Layer512Hidden3Dropout(model_name, num_labels)
    elif hybrid == "Bilstm3Layer512Hidden5Dropout":
        print("model is", model_name, hybrid)
        return Bilstm3Layer512Hidden5Dropout(model_name, num_labels)
    else:
        raise ValueError("Unsupported hybrid type.")

def run_kfold_experiment(
    X, y, model_name, hybrid_type, resample_method,
    kfold, seed, epochs, patience, max_length, batch_size, lr, weight_decay, 
    upper_cap, lower_cap, use_weighted_loss=False
):
    print(f"▶ Running: {model_name} + {hybrid_type or 'plain'} + {resample_method}")
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
    all_fold_results = []
    tokenizer = get_tokenizer(model_name)
    device = get_device_info()
    
    start_time = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n[Fold {fold + 1}]")
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]

        X_train, y_train = apply_resampling(X_train, y_train, method=resample_method, random_state=seed, upper_cap=upper_cap, lower_cap=lower_cap)

        label_counts = Counter(y_train)
        print("Label distribution after resampling:")
        for lbl, cnt in sorted(label_counts.items()):
            print(f"  label {lbl}: {cnt}")
        
        train_dataset = CustomDataset(X_train, y_train, tokenizer, max_length)
        val_dataset = CustomDataset(X_val, y_val, tokenizer, max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
        num_labels = len(set(y))
        model = get_model(model_name, hybrid_type, num_labels=num_labels)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 🔥 加入 Weighted CrossEntropy Loss（平滑版）
        if use_weighted_loss:
            samples_per_class = [label_counts.get(i, 0) for i in range(num_labels)]
            weights = np.log((sum(samples_per_class) + 1e-6) / (np.array(samples_per_class) + 1e-6))
            weights = torch.tensor(weights, dtype=torch.float).to(device)
            print(f"Using weighted CrossEntropyLoss with weights: {weights}")
            criterion = CrossEntropyLoss(weight=weights)
        else:
            criterion = CrossEntropyLoss()

        epoch_results = []
        best_macro_f1 = 0
        epochs_without_improvement = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            metrics = eval_model(model, val_loader, criterion, device, num_labels)
            metrics["train_loss"] = train_loss
            print(metrics)
            epoch_results.append(metrics)

            current_macro_f1 = metrics["macro_f1-score"]
            if current_macro_f1 > best_macro_f1:
                best_macro_f1 = current_macro_f1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Early stopping.")
                break

        best_epoch_metrics = max(epoch_results, key=lambda x: x["macro_f1-score"])
        all_fold_results.append(best_epoch_metrics)

    end_time = time.time()
    total_seconds = int(end_time - start_time)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n === {model_name} + {hybrid_type or 'plain'} + {resample_method} Final {kfold}-fold Cross-Validation Results ===")
    
    final_metrics = {k: np.mean([f[k] for f in all_fold_results]) for k in all_fold_results[0].keys()}

    final_metrics["total_seconds"] = total_seconds
    final_metrics["total_time"] = f"{hours} hrs {minutes} mins {seconds} secs"
    
    for metric, value in final_metrics.items():
        if isinstance(value, (float, int)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    return final_metrics