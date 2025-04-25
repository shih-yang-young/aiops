import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling  import RandomOverSampler
from gensim.models import Word2Vec
import random
from sentence_transformers import SentenceTransformer, util
import torch
from device import get_device_info
from config import *

def replace_random_words(text, model, num_replacements=REPLACE_WORDS_NUM):
    words = text.split()
    if not words:
        return text
    replaced_text = []
    replaced_indices = random.sample(range(len(words)), min(num_replacements, len(words)))
    for i, word in enumerate(words):
        if i in replaced_indices and word in model.wv.key_to_index:
            similar_words = model.wv.most_similar(word, topn=1)
            if similar_words:
                replaced_text.append(similar_words[0][0])
            else:
                replaced_text.append(word)
        else:
            replaced_text.append(word)
    return ' '.join(replaced_text)

def apply_resampling(X, y, method="none",random_state=SEED, upper_cap=UPPER_CAP, lower_cap=LOWER_CAP):
    if method == "none":
        return X, y
    elif method == "ros":

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
        
    elif method == "word2vec":
        print("🚀 Using Word2Vec for data augmentation")

        # 用 Counter 數每類有幾筆
        label_counts = Counter(y)

        # 訓練 Word2Vec（只用原始訓練資料）
        corpus = [text.split() for text in X]
        w2v_model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, sg=0)

        augmented_X, augmented_y = [], []

        for label in sorted(label_counts.keys()):
            samples = [X[i] for i in range(len(X)) if y[i] == label]
            count = label_counts[label]
            
            # 原始樣本先加進來
            augmented_X.extend(samples)
            augmented_y.extend([label] * count)

            if count < lower_cap:
                needed = lower_cap - count
                print(f"🔧 Augmenting label {label}: {count} → {lower_cap} (+{needed})")

                for _ in range(needed):
                    src = random.choice(samples)
                    augmented_sample = replace_random_words(src, w2v_model, num_replacements=3)
                    augmented_X.append(augmented_sample)
                    augmented_y.append(label)
        
        return augmented_X, augmented_y
    elif method == "eda":
        ## 幫我完成EDA
        return 
    elif method == "sbert":
        print("🚀 Using SBERT for data augmentation")
        label_counts = Counter(y)
        
        # 初始化 SBERT 模型（只初始化一次）
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        sbert_model.to(get_device_info())
        print("Model device:", sbert_model.device)
        
        def get_sbert_augmented_sentences(original_sentence: str, corpus_sentences: list, top_k: int = 5, score_threshold: float = 0.7):
            corpus_embeddings = sbert_model.encode(corpus_sentences, convert_to_tensor=True)
            query_embedding = sbert_model.encode(original_sentence, convert_to_tensor=True)
    
            cosine_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cosine_scores, k=min(top_k + 1, len(corpus_sentences)))  # +1 是為了避免選到自己
    
            results = []
            for score, idx in zip(top_results[0], top_results[1]):
                candidate = corpus_sentences[idx]
                if candidate != original_sentence and score >= score_threshold:
                    results.append(candidate)
            return results
    
        # 用於增強的語料庫（包含所有句子）
        all_corpus = list(set(X))
    
        augmented_X, augmented_y = [], []
    
        for label in sorted(label_counts.keys()):
            samples = [X[i] for i in range(len(X)) if y[i] == label]
            count = label_counts[label]
    
            # 原始樣本保留
            augmented_X.extend(samples)
            augmented_y.extend([label] * count)
    
            if count < lower_cap:
                needed = lower_cap - count
                print(f"🔧 Augmenting label {label}: {count} → {lower_cap} (+{needed})")
    
                added = 0
                for sentence in samples:
                    similar_sentences = get_sbert_augmented_sentences(
                        original_sentence=sentence,
                        corpus_sentences=all_corpus,
                        top_k=10,  # 可調整搜尋量
                        score_threshold=0.75  # 可調整語意標準
                    )
                    for aug in similar_sentences:
                        if added >= needed:
                            break
                        augmented_X.append(aug)
                        augmented_y.append(label)
                        added += 1
                    if added >= needed:
                        break
    
        return augmented_X, augmented_y
        
    elif method == "textgan":
        return textgan_augment(X, y)  # 假設你有自訂 function
    else:
        raise ValueError("Unknown resampling method")