import torch
import torch.nn as nn
from transformers import AutoModel

class HybridCnn1Filter234Drop3Classifier(nn.Module):
    def __init__(self, model_name, num_labels, num_filters=100, filter_sizes=(2, 3, 4), dropout_rate=0.3):
        super(HybridCnn1Filter234Drop3Classifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.filter_sizes = filter_sizes
        self.num_labels = num_labels

        hidden_size = self.bert.config.hidden_size  # 通常為 768

        # CNN 卷積層：1D Conv → (batch_size, hidden_dim, seq_len)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=num_filters,
                kernel_size=fs,
                padding=fs // 2  # 保持長度不變
            )
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(len(filter_sizes) * num_filters, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # 取得 BERT 輸出：(batch_size, seq_len, hidden_dim)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state

        # 轉換為 CNN 輸入格式：(batch_size, hidden_dim, seq_len)
        x = x.transpose(1, 2)

        # 卷積 + ReLU + max pooling
        conved = [torch.relu(conv(x)) for conv in self.convs]  # list of (batch_size, num_filters, seq_len)
        pooled = [torch.max(c, dim=2)[0] for c in conved]      # list of (batch_size, num_filters)
        cat = torch.cat(pooled, dim=1)                         # (batch_size, num_filters * len(filter_sizes))

        out = self.dropout(cat)
        logits = self.classifier(out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class HybridCnn1Filter234Drop5Classifier(nn.Module):
    def __init__(self, model_name, num_labels, num_filters=100, filter_sizes=(2, 3, 4), dropout_rate=0.5):
        super(HybridCnn1Filter234Drop5Classifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.filter_sizes = filter_sizes
        self.num_labels = num_labels

        hidden_size = self.bert.config.hidden_size  # 通常為 768

        # CNN 卷積層：1D Conv → (batch_size, hidden_dim, seq_len)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=num_filters,
                kernel_size=fs,
                padding=fs // 2  # 保持長度不變
            )
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(len(filter_sizes) * num_filters, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # 取得 BERT 輸出：(batch_size, seq_len, hidden_dim)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state

        # 轉換為 CNN 輸入格式：(batch_size, hidden_dim, seq_len)
        x = x.transpose(1, 2)

        # 卷積 + ReLU + max pooling
        conved = [torch.relu(conv(x)) for conv in self.convs]  # list of (batch_size, num_filters, seq_len)
        pooled = [torch.max(c, dim=2)[0] for c in conved]      # list of (batch_size, num_filters)
        cat = torch.cat(pooled, dim=1)                         # (batch_size, num_filters * len(filter_sizes))

        out = self.dropout(cat)
        logits = self.classifier(out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}
class HybridCnn1Filter345Drop3Classifier(nn.Module):
    def __init__(self, model_name, num_labels, num_filters=100, filter_sizes=(3, 4, 5), dropout_rate=0.3):
        super(HybridCnn1Filter345Drop3Classifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.filter_sizes = filter_sizes
        self.num_labels = num_labels

        hidden_size = self.bert.config.hidden_size  # 通常為 768

        # CNN 卷積層：1D Conv → (batch_size, hidden_dim, seq_len)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=num_filters,
                kernel_size=fs,
                padding=fs // 2  # 保持長度不變
            )
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(len(filter_sizes) * num_filters, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # 取得 BERT 輸出：(batch_size, seq_len, hidden_dim)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state

        # 轉換為 CNN 輸入格式：(batch_size, hidden_dim, seq_len)
        x = x.transpose(1, 2)

        # 卷積 + ReLU + max pooling
        conved = [torch.relu(conv(x)) for conv in self.convs]  # list of (batch_size, num_filters, seq_len)
        pooled = [torch.max(c, dim=2)[0] for c in conved]      # list of (batch_size, num_filters)
        cat = torch.cat(pooled, dim=1)                         # (batch_size, num_filters * len(filter_sizes))

        out = self.dropout(cat)
        logits = self.classifier(out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}
class HybridCnn1Filter345Drop5Classifier(nn.Module):
    def __init__(self, model_name, num_labels, num_filters=100, filter_sizes=(3, 4, 5), dropout_rate=0.5):
        super(HybridCnn1Filter345Drop5Classifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.filter_sizes = filter_sizes
        self.num_labels = num_labels

        hidden_size = self.bert.config.hidden_size  # 通常為 768

        # CNN 卷積層：1D Conv → (batch_size, hidden_dim, seq_len)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=num_filters,
                kernel_size=fs,
                padding=fs // 2  # 保持長度不變
            )
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(len(filter_sizes) * num_filters, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # 取得 BERT 輸出：(batch_size, seq_len, hidden_dim)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state

        # 轉換為 CNN 輸入格式：(batch_size, hidden_dim, seq_len)
        x = x.transpose(1, 2)

        # 卷積 + ReLU + max pooling
        conved = [torch.relu(conv(x)) for conv in self.convs]  # list of (batch_size, num_filters, seq_len)
        pooled = [torch.max(c, dim=2)[0] for c in conved]      # list of (batch_size, num_filters)
        cat = torch.cat(pooled, dim=1)                         # (batch_size, num_filters * len(filter_sizes))

        out = self.dropout(cat)
        logits = self.classifier(out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class HybridCnn2Filter234Drop3Classifier(nn.Module):
    def __init__(self, model_name, num_labels, num_filters=100, filter_sizes=(2, 3, 4), dropout_rate=0.3):
        super(HybridCnn2Filter234Drop3Classifier, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.filter_sizes = filter_sizes
        self.num_labels = num_labels
        hidden_size = self.bert.config.hidden_size  # BERT hidden size (usually 768)

        # 多層 CNN 設計（兩層 1D Conv）
        self.conv_blocks = nn.ModuleList()
        for fs in filter_sizes:
            block = nn.Sequential(
                nn.Conv1d(hidden_size, num_filters, kernel_size=fs, padding=fs // 2),  # Conv Layer 1
                nn.ReLU(),
                nn.Conv1d(num_filters, num_filters, kernel_size=fs, padding=fs // 2),  # Conv Layer 2
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)  # 固定每個 filter 輸出為 (batch, num_filters, 1)
            )
            self.conv_blocks.append(block)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        x = x.permute(0, 2, 1)  # (batch_size, hidden_dim, seq_len)

        conv_outs = []
        for block in self.conv_blocks:
            out = block(x)                # (batch_size, num_filters, 1)
            out = out.squeeze(-1)         # -> (batch_size, num_filters)
            conv_outs.append(out)

        x = torch.cat(conv_outs, dim=1)   # (batch_size, num_filters * len(filter_sizes))
        x = self.dropout(x)
        logits = self.classifier(x)       # (batch_size, num_labels)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class HybridCnn2Filter234Drop5Classifier(nn.Module):
    def __init__(self, model_name, num_labels, num_filters=100, filter_sizes=(2, 3, 4), dropout_rate=0.5):
        super(HybridCnn2Filter234Drop5Classifier, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.filter_sizes = filter_sizes
        self.num_labels = num_labels
        hidden_size = self.bert.config.hidden_size  # BERT hidden size (usually 768)

        # 多層 CNN 設計（兩層 1D Conv）
        self.conv_blocks = nn.ModuleList()
        for fs in filter_sizes:
            block = nn.Sequential(
                nn.Conv1d(hidden_size, num_filters, kernel_size=fs, padding=fs // 2),  # Conv Layer 1
                nn.ReLU(),
                nn.Conv1d(num_filters, num_filters, kernel_size=fs, padding=fs // 2),  # Conv Layer 2
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)  # 固定每個 filter 輸出為 (batch, num_filters, 1)
            )
            self.conv_blocks.append(block)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        x = x.permute(0, 2, 1)  # (batch_size, hidden_dim, seq_len)

        conv_outs = []
        for block in self.conv_blocks:
            out = block(x)                # (batch_size, num_filters, 1)
            out = out.squeeze(-1)         # -> (batch_size, num_filters)
            conv_outs.append(out)

        x = torch.cat(conv_outs, dim=1)   # (batch_size, num_filters * len(filter_sizes))
        x = self.dropout(x)
        logits = self.classifier(x)       # (batch_size, num_labels)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class HybridCnn2Filter345Drop3Classifier(nn.Module):
    def __init__(self, model_name, num_labels, num_filters=100, filter_sizes=(3, 4, 5), dropout_rate=0.3):
        super(HybridCnn2Filter345Drop3Classifier, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.filter_sizes = filter_sizes
        self.num_labels = num_labels
        hidden_size = self.bert.config.hidden_size  # BERT hidden size (usually 768)

        # 多層 CNN 設計（兩層 1D Conv）
        self.conv_blocks = nn.ModuleList()
        for fs in filter_sizes:
            block = nn.Sequential(
                nn.Conv1d(hidden_size, num_filters, kernel_size=fs, padding=fs // 2),  # Conv Layer 1
                nn.ReLU(),
                nn.Conv1d(num_filters, num_filters, kernel_size=fs, padding=fs // 2),  # Conv Layer 2
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)  # 對每個 filter 輸出做 max pooling，固定輸出為 (batch, num_filters, 1)
            )
            self.conv_blocks.append(block)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # Step 1: BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # Step 2: Convert to CNN input format
        x = x.permute(0, 2, 1)  # (batch_size, hidden_dim, seq_len)

        # Step 3: Apply each CNN block
        conv_outs = []
        for block in self.conv_blocks:
            out = block(x)                # (batch_size, num_filters, 1)
            out = out.squeeze(-1)         # -> (batch_size, num_filters)
            conv_outs.append(out)

        # Step 4: Concatenate all filters
        x = torch.cat(conv_outs, dim=1)   # (batch_size, num_filters * len(filter_sizes))
        x = self.dropout(x)
        logits = self.classifier(x)       # (batch_size, num_labels)

        # Step 5: Loss or prediction
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class HybridCnn2Filter345Drop5Classifier(nn.Module):
    def __init__(self, model_name, num_labels, num_filters=100, filter_sizes=(3, 4, 5), dropout_rate=0.5):
        super(HybridCnn2Filter345Drop5Classifier, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.filter_sizes = filter_sizes
        self.num_labels = num_labels
        hidden_size = self.bert.config.hidden_size  # BERT hidden size (usually 768)

        # 多層 CNN 設計（兩層 1D Conv）
        self.conv_blocks = nn.ModuleList()
        for fs in filter_sizes:
            block = nn.Sequential(
                nn.Conv1d(hidden_size, num_filters, kernel_size=fs, padding=fs // 2),  # Conv Layer 1
                nn.ReLU(),
                nn.Conv1d(num_filters, num_filters, kernel_size=fs, padding=fs // 2),  # Conv Layer 2
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)  # 對每個 filter 輸出做 max pooling，固定輸出為 (batch, num_filters, 1)
            )
            self.conv_blocks.append(block)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # Step 1: BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # Step 2: Convert to CNN input format
        x = x.permute(0, 2, 1)  # (batch_size, hidden_dim, seq_len)

        # Step 3: Apply each CNN block
        conv_outs = []
        for block in self.conv_blocks:
            out = block(x)                # (batch_size, num_filters, 1)
            out = out.squeeze(-1)         # -> (batch_size, num_filters)
            conv_outs.append(out)

        # Step 4: Concatenate all filters
        x = torch.cat(conv_outs, dim=1)   # (batch_size, num_filters * len(filter_sizes))
        x = self.dropout(x)
        logits = self.classifier(x)       # (batch_size, num_labels)

        # Step 5: Loss or prediction
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}