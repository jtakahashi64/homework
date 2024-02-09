import math
from collections import OrderedDict
import random
import numpy as np
import torch
import torch.nn as nn
# import sentencepiece

LAYER_SIZE = 4
HEADS_SIZE = 8
EPOCH_SIZE = 10000
BATCH_SIZE = 8
CONTEXT_WINDOW_SIZE = 32
VOCAB_SIZE = -1
EMBED_SIZE = 48
TEXT_SIZE = 5
MAX_LENGTH = 500
LOG_INTERVAL = 10


class DataLoader:
    def __init__(self, name='input', filename='input.txt', batch_size=BATCH_SIZE, context_window_size=CONTEXT_WINDOW_SIZE, vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE):
        self.name = name
        self.filename = filename
        self.batch_size = batch_size
        self.context_window_size = context_window_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size

    def train(self):
        # sentencepiece.SentencePieceTrainer.train(input=self.filename, model_prefix=self.name, vocab_size=self.vocab_size)
        pass

    def load(self):
        # self.model = sentencepiece.SentencePieceProcessor(model_file=self.name + '.model')

        # with open(self.filename, 'r') as file:
        #     data = file.read()
        #     tokens = self.encode(data)
        #     self.dataset = torch.tensor(tokens, dtype=torch.long)
        lines = open(self.filename, 'r').read()
        vocab = sorted(list(set(lines)))
        self.vocab_size = len(vocab)
        print(self.vocab_size)
        self.itos = {i: ch for i, ch in enumerate(vocab)}
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        tokens = self.encode(lines)
        self.dataset = torch.tensor(tokens, dtype=torch.int8)

    def encode(self, s):
        return [self.stoi[ch] for ch in s]
        # return self.model.encode(s)

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
        # return self.model.decode(l)

    def get_batches(self):
        train_xs, train_ys = self.get_batch(self.dataset[:int(.8 * len(self.dataset))])
        valid_xs, valid_ys = self.get_batch(self.dataset[int(.8 * len(self.dataset)):int(.9 * len(self.dataset))])
        test_xs,  test_ys = self.get_batch(self.dataset[int(.9 * len(self.dataset)):])

        self.batches = {
            "train": {"x": train_xs, "y": train_ys},
            "valid": {"x": valid_xs, "y": valid_ys},
            "test":  {"x": test_xs,  "y": test_ys},
        }

        return self.batches

    def get_batch(self, batch_data):
        x_array = []
        y_array = []

        for _ in range(self.batch_size):
            r = random.randint(0, batch_data.size(0) - self.context_window_size - 1)
            x_array.append(batch_data[r + 0:r + self.context_window_size + 0])
            y_array.append(batch_data[r + 1:r + self.context_window_size + 1])

        return torch.stack(x_array).long(), torch.stack(y_array).long()


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout=0.1, max_length=MAX_LENGTH):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))

        pe = torch.zeros(max_length, 1, embed_size)

        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)

        return x


class HeadAttention(torch.nn.Module):
    def __init__(self, embed_size):
        super().__init__()

        # Q, K, V の重み
        self.w_q = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.w_k = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.w_v = torch.nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=.1, is_causal=True)

        return scaled_dot_product_attention


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, heads_size, embed_size):
        super().__init__()

        self.heads = torch.nn.ModuleList([
            HeadAttention(embed_size) for _ in range(heads_size)
        ])

        self.linear = torch.nn.Linear(heads_size * embed_size, embed_size)
        self.dropout = torch.nn.Dropout(.1)

    def forward(self, x):
        heads = [h(x) for h in self.heads]

        x = torch.cat(heads, dim=-1)

        x = self.linear(x)
        x = self.dropout(x)

        return x


class Block(torch.nn.Module):
    def __init__(self, heads_size, embed_size):
        super().__init__()

        self.norm = torch.nn.LayerNorm(embed_size)

        self.mha = MultiHeadAttention(heads_size, embed_size)

        self.ffn = torch.nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
        )

    def forward(self, x):
        # 正規化
        x = self.norm(x)
        # Attention Is All You Need
        #   残差接続
        x = x + self.mha(x)
        # 正規化
        x = self.norm(x)
        # 活性化
        #   残差接続
        x = x + self.ffn(x)

        return x


class Model(torch.nn.Module):
    def __init__(self, layer_size, heads_size, batch_size, content_window_size, vocab_size, embed_size):
        super().__init__()

        self.layer_size = layer_size
        self.heads_size = heads_size
        self.batch_size = batch_size
        self.content_window_size = content_window_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embedding = torch.nn.Embedding(self.vocab_size, self.embed_size)

        self.position_encoding = PositionalEncoding(self.embed_size)

        self.blocks = nn.Sequential(
            OrderedDict([(f"blocks{i}", Block(self.heads_size, self.embed_size)) for i in range(layer_size)])
        )

        self.ffn = nn.Sequential(
            nn.Linear(self.embed_size, self.embed_size),
            nn.ReLU(),
            nn.Linear(self.embed_size, self.vocab_size),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.position_encoding(x)
        x = self.blocks(x)
        logits = self.ffn(x)
        return logits


class Trainer:
    def __init__(self, data_loader, model, log_interval=LOG_INTERVAL):
        self.model = model
        self.data_loader = data_loader
        self.vocab_size = self.data_loader.vocab_size
        self.embed_size = self.data_loader.embed_size
        self.log_interval = log_interval

        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train(self, epoch_size):
        for epoch in range(epoch_size):
            self.optimizer.zero_grad()

            batches = self.data_loader.get_batches()

            x, y = batches["train"]["x"], batches["train"]["y"],

            logits = self.model(x)
            fixers = y

            flattened_logits = logits.view(-1, self.vocab_size)
            flattened_fixers = fixers.view(-1)

            loss = torch.nn.functional.cross_entropy(flattened_logits, flattened_fixers)

            loss.backward()

            self.optimizer.step()

            if epoch % self.log_interval == 0:
                print(loss)


class Generator:
    def __init__(self, data_loader, model, text_size=TEXT_SIZE, max_length=MAX_LENGTH):
        self.model = model
        self.data_loader = data_loader
        self.context_window_size = self.data_loader.context_window_size
        self.vocab_size = self.data_loader.vocab_size
        self.embed_size = self.data_loader.embed_size
        self.text_size = TEXT_SIZE
        self.max_length = MAX_LENGTH

    def generate(self):
        # テキストサイズ分のテキスト（最初の文字はid:0）を生成
        texts = torch.zeros(self.text_size, 1).long()

        for _ in range(self.max_length):
            # 入力
            #   0次元目 バッチ分のテキストごと
            #   1次元目 テキストの末尾（コンテキストウィンドウの文字数分）
            # 出力
            #   0次元目 バッチ分のテキストごと
            #   1次元目 コンテキストウィンドウ分のトークンごと
            #   2次元目 各々のトークンの次のトークンの確率（埋め込み行列）
            logits = self.model(texts[:, -self.context_window_size:])

            # 入力
            #   0次元目 バッチ分のテキストごと
            #   1次元目 コンテキストウィンドウ分のトークンごと 最後の要素のみ
            #   2次元目 各々のトークンの次のトークンの確率（埋め込み行列）
            # 出力
            #   0次元目 コンテキストウィンドウ分のトークンごと
            #   1次元目 最後のトークンの次のトークンの確率（埋め込み行列）
            predictions = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)

            # 確率分布から各テキストごとの次のトークンを生成
            add_texts_token = torch.multinomial(
                predictions, num_samples=1
            )

            # 各テキストの末尾に次のトークンを追加
            texts = torch.cat([texts, add_texts_token], dim=-1)

        return [self.data_loader.decode(token) for token in texts.tolist()]


data_loader = DataLoader()
data_loader.train()
data_loader.load()

model = Model(LAYER_SIZE, HEADS_SIZE, data_loader.batch_size, data_loader.context_window_size, data_loader.vocab_size, data_loader.embed_size)

trainer = Trainer(data_loader, model)
generator = Generator(data_loader, model)

trainer.train(EPOCH_SIZE)

print(generator.generate())

print(data_loader.decode([0]))
