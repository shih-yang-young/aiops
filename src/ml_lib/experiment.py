from dataset import CustomDataset
from train_eval import train_epoch, eval_model
from device import get_device_info
from hybrid_model import HybridLstmClassifier, HybridCnnClassifier, HybridBilstmClassifier

from transformers import BertTokenizer, RobertaTokenizer, DebertaTokenizer, AutoModelForSequenceClassification
import numpy as np

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
    else:
        raise ValueError("Unsupported hybrid type.")

def apply_resampling(X, y, method="none"):
    if method == "none":
        return X, y
    elif method == "ros":
        return RandomOverSampler().fit_resample(X, y)
    elif method == "smote":
        return SMOTE().fit_resample(X, y)
    elif method == "textgan":
        return textgan_augment(X, y)  # 假設你有自訂 function
    else:
        raise ValueError("Unknown resampling method")

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import numpy as np
import time

def run_kfold_experiment(
    X, y, model_name, hybrid_type, resample_method,
    kfold, seed, epochs, patience, max_length, batch_size, lr, weight_decay, 
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

        X_train, y_train = apply_resampling(X_train, y_train, method=resample_method)

        train_dataset = CustomDataset(X_train, y_train, tokenizer, max_length)
        val_dataset = CustomDataset(X_val, y_val, tokenizer, max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        num_labels = len(set(y))
        model = get_model(model_name, hybrid_type, num_labels=num_labels)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
    print(f"\n === {model_name} + {hybrid_type or 'plain'} + {resample_method} Final 10-fold Cross-Validation Results ===")
    print(f"Total time: {end_time - start_time:.2f}seconds")
    
    final_metrics = {k: np.mean([f[k] for f in all_fold_results]) for k in all_fold_results[0].keys()}
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return all_fold_results