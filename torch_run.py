# -*- coding: utf-8 -*-

import torch
from torch_train import AttentionClassifier, MAX_LEN, tokenize

checkpoint = torch.load("./checkpoint.pth")
    
vocab = checkpoint["vocab"]
DEVICE = torch.device("cpu")

model = AttentionClassifier(vocab).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

def encode(text):
    tokens = tokenize(text)

    ids = [vocab.get(token, 1) for token in tokens]

    if len(ids) > MAX_LEN:
        ids = ids[:MAX_LEN]
    else:
        ids += [0] * (MAX_LEN - len(ids))

    return ids

@torch.no_grad()
def predict(text):
    model.eval()

    x = torch.tensor(
        [encode(text)],
        dtype=torch.long,
    ).to(DEVICE)

    logits = model(x)

    pred = torch.argmax(logits, dim=1).item()

    return "positive" if pred == 1 else "negative"

print("===================")
print("./checkpoint.pth loaded.")
print("Press Ctrl+C to interrupt, Ctrl+Z(Windows)/+D(macOS/Linux) to end inputting.")
lines = []
while True:
    try:
        lines.append(input("> "))
    except EOFError:
        print("---\n" + predict(" ".join(lines)) + "\n")
        lines = []
    except KeyboardInterrupt:
        break