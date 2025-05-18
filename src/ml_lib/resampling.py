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

def random_deletion(words, p=0.2):
    # 以機率 p 刪除每個單字
    if len(words) == 1:
        return words
    return [word for word in words if random.uniform(0, 1) > p]

def random_swap(words, n):
    if len(words) < 2:
        return words  # 太短無法交換，直接回傳原本
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return words
    
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
        print("🚀 Using Word2Vec for data augmentation + under-sampling")

        label_counts = Counter(y)
    
        # 訓練 Word2Vec 模型（只用原始資料）
        corpus = [text.split() for text in X]
        w2v_model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, sg=0)
    
        augmented_X, augmented_y = [], []
    
        for label in sorted(label_counts.keys()):
            samples = [X[i] for i in range(len(X)) if y[i] == label]
            count = label_counts[label]
    
            if count > upper_cap:
                # 欠採樣大類別
                print(f"🔻 Under-sampling label {label}: {count} → {upper_cap}")
                samples = random.sample(samples, upper_cap)
                count = upper_cap
    
            # 保留目前樣本
            augmented_X.extend(samples)
            augmented_y.extend([label] * count)
    
            if count < lower_cap:
                # 過採樣小類別
                needed = lower_cap - count
                print(f"🔧 Augmenting label {label}: {count} → {lower_cap} (+{needed})")
    
                for _ in range(needed):
                    src = random.choice(samples)
                    augmented_sample = replace_random_words(src, w2v_model, num_replacements=3)
                    augmented_X.append(augmented_sample)
                    augmented_y.append(label)
    
        return augmented_X, augmented_y
    elif method == "word2vec_eda":
        print("🚀 Using Word2Vec + EDA (replace + deletion/swap) for data augmentation + under-sampling")
    
        label_counts = Counter(y)
        augmented_X, augmented_y = [], []
    
        # 訓練 Word2Vec 模型
        corpus = [text.split() for text in X]
        w2v_model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, sg=0)
    
        for label in sorted(label_counts.keys()):
            samples = [X[i] for i in range(len(X)) if y[i] == label]
            count = label_counts[label]
    
            if count > upper_cap:
                # 欠採樣大類別
                print(f"🔻 Under-sampling label {label}: {count} → {upper_cap}")
                samples = random.sample(samples, upper_cap)
                count = upper_cap
    
            # 保留目前樣本
            augmented_X.extend(samples)
            augmented_y.extend([label] * count)
    
            if count < lower_cap:
                # 過採樣小類別
                needed = lower_cap - count
                print(f"🔧 Augmenting label {label}: {count} → {lower_cap} (+{needed})")
    
                added = 0
                while added < needed:
                    src = random.choice(samples)
                    # 第一步：用 Word2Vec 抽換
                    replaced = replace_random_words(src, w2v_model, num_replacements=3)
                    words = replaced.split()
    
                    # 第二步：再做刪除或交換
                    if random.random() < 0.5:
                        aug_words = random_deletion(words)
                    else:
                        aug_words = random_swap(words, n=2)
    
                    aug = ' '.join(aug_words)
                    augmented_X.append(aug)
                    augmented_y.append(label)
                    added += 1
    
        return augmented_X, augmented_y
    elif method == "t5":
        print("🚀 Using T5-paraphraser for data augmentation + under-sampling")

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        # 初始化 T5-paraphraser 模型
        t5_tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
        t5_model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_paraphraser")
        t5_model = t5_model.to(get_device_info(print_info=True))

        def t5_paraphrase(text, num_return_sequences=3, max_length=MAX_LENGTH):
            prompt = f"paraphrase: {text} </s>"
            input_ids = t5_tokenizer.encode(prompt, return_tensors="pt", truncation=True).to(t5_model.device)
            outputs = t5_model.generate(
                input_ids=input_ids,
                max_length=max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1.5,
                num_return_sequences=num_return_sequences
            )
            return [t5_tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

        label_counts = Counter(y)
        augmented_X, augmented_y = [], []

        for label in sorted(label_counts.keys()):
            samples = [X[i] for i in range(len(X)) if y[i] == label]
            count = label_counts[label]

            # 欠採樣大類別
            if count > upper_cap:
                print(f"🔻 Under-sampling label {label}: {count} → {upper_cap}")
                samples = random.sample(samples, upper_cap)
                count = upper_cap

            # 保留樣本
            augmented_X.extend(samples)
            augmented_y.extend([label] * count)

            # 過採樣小類別
            if count < lower_cap:
                needed = lower_cap - count
                print(f"🔧 Augmenting label {label}: {count} → {lower_cap} (+{needed})")

                added = 0
                while added < needed:
                    src = random.choice(samples)
                    try:
                        new_texts = t5_paraphrase(src, num_return_sequences=min(3, needed - added))
                    except Exception as e:
                        print(f"⚠️ Error generating paraphrase: {e}")
                        continue

                    for text in new_texts:
                        if added >= needed:
                            break
                        augmented_X.append(text)
                        augmented_y.append(label)
                        added += 1

        return augmented_X, augmented_y
        
    elif method == "textgan":
        return textgan_augment(X, y)  # 假設你有自訂 function
    else:
        raise ValueError("Unknown resampling method")