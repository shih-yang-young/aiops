import torch
import torch.nn as nn
from transformers import AutoModel

class Bilstm1Layer256Hidden3Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=256, dropout_rate=0.3):
        super(Bilstm1Layer256Hidden3Dropout, self).__init__()
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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # BiLSTM 輸出
        lstm_output, _ = self.bilstm(sequence_output)

        # 取出最後一個時間步的向量
        last_hidden = torch.mean(lstm_output, dim=1)

        out = self.dropout(last_hidden)
        logits = self.classifier(out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Bilstm1Layer256Hidden5Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=256, dropout_rate=0.5):
        super(Bilstm1Layer256Hidden5Dropout, self).__init__()
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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # BiLSTM 輸出
        lstm_output, _ = self.bilstm(sequence_output)

        # 取出最後一個時間步的向量
        last_hidden = torch.mean(lstm_output, dim=1)

        out = self.dropout(last_hidden)
        logits = self.classifier(out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Bilstm1Layer512Hidden3Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=512, dropout_rate=0.3):
        super(Bilstm1Layer512Hidden3Dropout, self).__init__()
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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # BiLSTM 輸出
        lstm_output, _ = self.bilstm(sequence_output)

        # 取出最後一個時間步的向量
        last_hidden = torch.mean(lstm_output, dim=1)

        out = self.dropout(last_hidden)
        logits = self.classifier(out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Bilstm1Layer512Hidden5Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=512, dropout_rate=0.5):
        super(Bilstm1Layer512Hidden5Dropout, self).__init__()
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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # BiLSTM 輸出
        lstm_output, _ = self.bilstm(sequence_output)

        # 取出最後一個時間步的向量
        last_hidden = torch.mean(lstm_output, dim=1)

        out = self.dropout(last_hidden)
        logits = self.classifier(out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Bilstm2Layer256Hidden3Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=256, dropout_rate=0.3):
        super(Bilstm2Layer256Hidden3Dropout, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # BiLSTM：BERT 輸出 → BiLSTM
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Dropout & Linear
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)  # 因為是 BiLSTM 所以 ×2

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # BiLSTM 輸出
        lstm_output, _ = self.bilstm(sequence_output)

        # 取出最後一個時間步的向量
        last_hidden = torch.mean(lstm_output, dim=1)

        out = self.dropout(last_hidden)
        logits = self.classifier(out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Bilstm2Layer256Hidden5Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=256, dropout_rate=0.5):
        super(Bilstm2Layer256Hidden5Dropout, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # BiLSTM：BERT 輸出 → BiLSTM
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Dropout & Linear
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)  # 因為是 BiLSTM 所以 ×2

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # BiLSTM 輸出
        lstm_output, _ = self.bilstm(sequence_output)

        # 取出最後一個時間步的向量
        last_hidden = torch.mean(lstm_output, dim=1)

        out = self.dropout(last_hidden)
        logits = self.classifier(out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Bilstm2Layer512Hidden3Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=512, dropout_rate=0.3):
        super(Bilstm2Layer512Hidden3Dropout, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # BiLSTM：BERT 輸出 → BiLSTM
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Dropout & Linear
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)  # 因為是 BiLSTM 所以 ×2

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # BiLSTM 輸出
        lstm_output, _ = self.bilstm(sequence_output)

        # 取出最後一個時間步的向量
        last_hidden = torch.mean(lstm_output, dim=1)

        out = self.dropout(last_hidden)
        logits = self.classifier(out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Bilstm2Layer512Hidden5Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=512, dropout_rate=0.5):
        super(Bilstm2Layer512Hidden5Dropout, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # BiLSTM：BERT 輸出 → BiLSTM
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Dropout & Linear
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)  # 因為是 BiLSTM 所以 ×2

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # BiLSTM 輸出
        lstm_output, _ = self.bilstm(sequence_output)

        # 取出最後一個時間步的向量
        last_hidden = torch.mean(lstm_output, dim=1)

        out = self.dropout(last_hidden)
        logits = self.classifier(out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Bilstm3Layer256Hidden3Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=256, dropout_rate=0.3):
        super(Bilstm3Layer256Hidden3Dropout, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # BiLSTM：BERT 輸出 → BiLSTM
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )

        # Dropout & Linear
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)  # 因為是 BiLSTM 所以 ×2

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # BiLSTM 輸出
        lstm_output, _ = self.bilstm(sequence_output)

        # 取出最後一個時間步的向量
        last_hidden = torch.mean(lstm_output, dim=1)

        out = self.dropout(last_hidden)
        logits = self.classifier(out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Bilstm3Layer256Hidden5Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=256, dropout_rate=0.5):
        super(Bilstm3Layer256Hidden5Dropout, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # BiLSTM：BERT 輸出 → BiLSTM
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )

        # Dropout & Linear
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)  # 因為是 BiLSTM 所以 ×2

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # BiLSTM 輸出
        lstm_output, _ = self.bilstm(sequence_output)

        # 取出最後一個時間步的向量
        last_hidden = torch.mean(lstm_output, dim=1)

        out = self.dropout(last_hidden)
        logits = self.classifier(out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Bilstm3Layer512Hidden3Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=512, dropout_rate=0.3):
        super(Bilstm3Layer512Hidden3Dropout, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # BiLSTM：BERT 輸出 → BiLSTM
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )

        # Dropout & Linear
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)  # 因為是 BiLSTM 所以 ×2

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # BiLSTM 輸出
        lstm_output, _ = self.bilstm(sequence_output)

        # 取出最後一個時間步的向量
        last_hidden = torch.mean(lstm_output, dim=1)

        out = self.dropout(last_hidden)
        logits = self.classifier(out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class Bilstm3Layer512Hidden5Dropout(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=512, dropout_rate=0.5):
        super(Bilstm3Layer512Hidden5Dropout, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # BiLSTM：BERT 輸出 → BiLSTM
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )

        # Dropout & Linear
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)  # 因為是 BiLSTM 所以 ×2

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # BiLSTM 輸出
        lstm_output, _ = self.bilstm(sequence_output)

        # 取出最後一個時間步的向量
        last_hidden = torch.mean(lstm_output, dim=1)

        out = self.dropout(last_hidden)
        logits = self.classifier(out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}