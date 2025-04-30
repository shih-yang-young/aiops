import torch
import torch.nn as nn
from transformers import AutoModel

class HybridLstmClassifier(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=256, dropout_rate=0.3):
        super(HybridLstmClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # LSTM 層：BERT 輸出 768 → LSTM
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # BERT 輸出所有 token 的向量
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # shape: (B, L, 768)

        # 將 BERT 的輸出丟進 LSTM
        lstm_output, (h_n, c_n) = self.lstm(sequence_output)  # h_n shape: (1, B, hidden)

        # 取出最後一層的 hidden state（[batch, hidden]）
        final_hidden = h_n[-1]

        # Dropout + Linear 分類
        output = self.dropout(final_hidden)
        logits = self.classifier(output)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class HybridCnnClassifier(nn.Module):
    def __init__(self, model_name, num_labels, num_filters=100, filter_sizes=(3, 4, 5), dropout_rate=0.5):
        super(HybridCnnClassifier, self).__init__()
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

class HybridBilstmClassifier(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=256, dropout_rate=0.3):
        super(HybridBilstmClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # BiLSTM：BERT 輸出 → BiLSTM
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Dropout & Linear
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)  # 因為是 BiLSTM 所以 ×2

    def forward(self, input_ids, attention_mask, labels=None):
        # BERT 輸出所有 token 的向量
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # BiLSTM 輸出
        lstm_output, _ = self.bilstm(sequence_output)  # (batch_size, seq_len, hidden_size * 2)

        # 取出最後一個時間步的向量
        last_hidden = lstm_output[:, -1, :]  # (batch_size, hidden_size * 2)

        out = self.dropout(last_hidden)
        logits = self.classifier(out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}