import torch
import torch.nn as nn
from transformers import AutoModel

class Lstm1Layer256Hidden3Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=256, dropout_rate=0.3):
        super(Lstm1Layer256Hidden3Dropout, self).__init__()
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
        final_hidden = h_n[h_n.size(0) - 1]

        # Dropout + Linear 分類
        output = self.dropout(final_hidden)
        logits = self.classifier(output)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}
class Lstm1Layer256Hidden5Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=256, dropout_rate=0.5):
        super(Lstm1Layer256Hidden5Dropout, self).__init__()
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
        final_hidden = h_n[h_n.size(0) - 1]

        # Dropout + Linear 分類
        output = self.dropout(final_hidden)
        logits = self.classifier(output)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Lstm1Layer512Hidden3Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=512, dropout_rate=0.3):
        super(Lstm1Layer512Hidden3Dropout, self).__init__()
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
        final_hidden = h_n[h_n.size(0) - 1]

        # Dropout + Linear 分類
        output = self.dropout(final_hidden)
        logits = self.classifier(output)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Lstm1Layer512Hidden5Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=512, dropout_rate=0.5):
        super(Lstm1Layer512Hidden5Dropout, self).__init__()
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
        final_hidden = h_n[h_n.size(0) - 1]

        # Dropout + Linear 分類
        output = self.dropout(final_hidden)
        logits = self.classifier(output)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Lstm2Layer256Hidden3Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=256, dropout_rate=0.3):
        super(Lstm2Layer256Hidden3Dropout, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, 768)

        lstm_output, (h_n, c_n) = self.lstm(sequence_output)  # h_n: (2, B, hidden)

        final_hidden = h_n[h_n.size(0) - 1]

        output = self.dropout(final_hidden)
        logits = self.classifier(output)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}
class Lstm2Layer256Hidden5Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=256, dropout_rate=0.5):
        super(Lstm2Layer256Hidden5Dropout, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # 修改：LSTM 使用兩層
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, 768)

        lstm_output, (h_n, c_n) = self.lstm(sequence_output)  # h_n: (2, B, hidden)

        final_hidden = h_n[h_n.size(0) - 1]

        output = self.dropout(final_hidden)
        logits = self.classifier(output)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Lstm2Layer512Hidden3Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=512, dropout_rate=0.3):
        super(Lstm2Layer512Hidden3Dropout, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # 修改：LSTM 使用兩層
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, 768)

        lstm_output, (h_n, c_n) = self.lstm(sequence_output)  # h_n: (2, B, hidden)

        final_hidden = h_n[h_n.size(0) - 1]

        output = self.dropout(final_hidden)
        logits = self.classifier(output)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}
class Lstm2Layer512Hidden5Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=512, dropout_rate=0.5):
        super(Lstm2Layer512Hidden5Dropout, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # 修改：LSTM 使用兩層
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, 768)

        lstm_output, (h_n, c_n) = self.lstm(sequence_output)  # h_n: (2, B, hidden)

        final_hidden = h_n[h_n.size(0) - 1]

        output = self.dropout(final_hidden)
        logits = self.classifier(output)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Lstm3Layer256Hidden3Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=256, dropout_rate=0.3):
        super(Lstm3Layer256Hidden3Dropout, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # 修改：LSTM 使用兩層
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, 768)

        lstm_output, (h_n, c_n) = self.lstm(sequence_output)  # h_n: (2, B, hidden)

        final_hidden = h_n[h_n.size(0) - 1]

        output = self.dropout(final_hidden)
        logits = self.classifier(output)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Lstm3Layer256Hidden5Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=256, dropout_rate=0.5):
        super(Lstm3Layer256Hidden5Dropout, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # 修改：LSTM 使用兩層
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, 768)

        lstm_output, (h_n, c_n) = self.lstm(sequence_output)  # h_n: (2, B, hidden)

        final_hidden = h_n[h_n.size(0) - 1]

        output = self.dropout(final_hidden)
        logits = self.classifier(output)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Lstm3Layer512Hidden3Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=512, dropout_rate=0.3):
        super(Lstm3Layer512Hidden3Dropout, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # 修改：LSTM 使用兩層
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, 768)

        lstm_output, (h_n, c_n) = self.lstm(sequence_output)  # h_n: (2, B, hidden)

        final_hidden = h_n[h_n.size(0) - 1]

        output = self.dropout(final_hidden)
        logits = self.classifier(output)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Lstm3Layer512Hidden5Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=512, dropout_rate=0.5):
        super(Lstm3Layer512Hidden5Dropout, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # 修改：LSTM 使用兩層
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, 768)

        lstm_output, (h_n, c_n) = self.lstm(sequence_output)  # h_n: (2, B, hidden)

        final_hidden = h_n[h_n.size(0) - 1]

        output = self.dropout(final_hidden)
        logits = self.classifier(output)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}