from dataset import CustomDataset
from train_eval import train_epoch, eval_model
from device import get_device_info
from hybrid_model import HybridLstmClassifier, HybridCnnClassifier, HybridBilstmClassifier, HybridCnn1Filter234Drop3Classifier, HybridCnn1Filter234Drop5Classifier, HybridCnn1Filter345Drop3Classifier, HybridCnn1Filter345Drop5Classifier, HybridCnn2Filter234Drop3Classifier, HybridCnn2Filter234Drop5Classifier, HybridCnn2Filter345Drop3Classifier, HybridCnn2Filter345Drop5Classifier

from transformers import BertTokenizer, RobertaTokenizer, DebertaTokenizer, AutoModelForSequenceClassification
import numpy as np

from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling  import RandomOverSampler

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
    else:
        raise ValueError("Unsupported hybrid type.")

def apply_resampling(X, y, method="none",random_state=42):
    if method == "none":
        return X, y
    elif method == "ros":
        upper_cap=2000
        lower_cap=200

        # 1️⃣ 先把 list 轉成 numpy array，方便 index 對應
        X_arr = np.array(X, dtype=object)
        y_arr = np.array(y)

        # 2️⃣ 取得原始類別分佈
        counter = Counter(y_arr)

        # 3️⃣ 欠採樣策略：只針對 > upper_cap 的類別
        under_dict = {cls: upper_cap for cls, cnt in counter.items() if cnt > upper_cap}
        # 如果沒有大類需要欠採樣，under_dict 會是空 dict
        
        if under_dict:
            rus = RandomUnderSampler(sampling_strategy=under_dict, random_state=random_state)
            X_arr, y_arr = rus.fit_resample(X_arr.reshape(-1, 1), y_arr)
            X_arr = X_arr.ravel()          # 還原 shape

        # 4️⃣ 重新計算分佈，決定 ROS 目標
        counter = Counter(y_arr)
        over_dict = {cls: lower_cap for cls, cnt in counter.items() if cnt < lower_cap}
    
        if over_dict:
            ros = RandomOverSampler(sampling_strategy=over_dict, random_state=random_state)
            X_arr, y_arr = ros.fit_resample(X_arr.reshape(-1, 1), y_arr)
            X_arr = X_arr.ravel()
            
        return list(X_arr), list(y_arr)
        
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

        X_train, y_train = apply_resampling(X_train, y_train, method=resample_method, random_state=seed)

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
    print(f"Total time: {total_seconds} seconds")
    print(f"Total time: {hours} hrs {minutes} mins {seconds} secs")
    
    final_metrics = {k: np.mean([f[k] for f in all_fold_results]) for k in all_fold_results[0].keys()}
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return final_metrics