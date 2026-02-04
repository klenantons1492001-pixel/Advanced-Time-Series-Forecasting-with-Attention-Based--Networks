#### &nbsp;**Advanced Time Series Forecasting with Attention-Based Networks**



import numpy as np

import torch

import torch.nn as nn

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean\_squared\_error



\# ---------------- CONFIG ----------------

DEVICE = torch.device("cuda" if torch.cuda.is\_available() else "cpu")

WINDOW = 40

HORIZON = 10

EPOCHS = 30

LR = 0.001



np.random.seed(42)

torch.manual\_seed(42)



\# ==============================================================

\# 1. COMPLEX SYNTHETIC DATASET (>=1000, multi-seasonal + regime)

\# ==============================================================



n = 1700

t = np.arange(n)



season\_short = 8 \* np.sin(2 \* np.pi \* t / 24)

season\_long = 5 \* np.sin(2 \* np.pi \* t / 168)

trend = 0.004 \* t

noise = np.random.normal(0, 1.2, n)

regime = np.where(t > 950, 12, 0)



series = season\_short + season\_long + trend + regime + noise

series = series.reshape(-1, 1)



\# ---------------- SCALING ----------------

scaler = MinMaxScaler()

series\_scaled = scaler.fit\_transform(series)



\# ==============================================================

\# 2. SLIDING WINDOW MULTI‑STEP DATASET

\# ==============================================================



def create\_sequences(data, window, horizon):

&nbsp;   X, y = \[], \[]

&nbsp;   for i in range(len(data) - window - horizon):

&nbsp;       X.append(data\[i:i+window])

&nbsp;       y.append(data\[i+window:i+window+horizon])

&nbsp;   return np.array(X), np.array(y)



X, y = create\_sequences(series\_scaled, WINDOW, HORIZON)



split = int(0.8 \* len(X))

X\_train, X\_test = X\[:split], X\[split:]

y\_train, y\_test = y\[:split], y\[split:]



X\_train = torch.tensor(X\_train, dtype=torch.float32).to(DEVICE)

y\_train = torch.tensor(y\_train, dtype=torch.float32).to(DEVICE)

X\_test = torch.tensor(X\_test, dtype=torch.float32).to(DEVICE)

y\_test = torch.tensor(y\_test, dtype=torch.float32).to(DEVICE)



\# ==============================================================

\# 3. TRANSFORMER WITH ATTENTION WEIGHT EXTRACTION

\# ==============================================================



class TransformerForecast(nn.Module):

&nbsp;   def \_\_init\_\_(self, input\_dim=1, d\_model=64, nhead=4, layers=2, horizon=10):

&nbsp;       super().\_\_init\_\_()



&nbsp;       self.embedding = nn.Linear(input\_dim, d\_model)



&nbsp;       self.attn\_layer = nn.MultiheadAttention(

&nbsp;           embed\_dim=d\_model,

&nbsp;           num\_heads=nhead,

&nbsp;           batch\_first=True

&nbsp;       )



&nbsp;       encoder\_layer = nn.TransformerEncoderLayer(

&nbsp;           d\_model=d\_model,

&nbsp;           nhead=nhead,

&nbsp;           batch\_first=True,

&nbsp;           dim\_feedforward=128

&nbsp;       )



&nbsp;       self.transformer = nn.TransformerEncoder(encoder\_layer, num\_layers=layers)

&nbsp;       self.fc = nn.Linear(d\_model, horizon)



&nbsp;       self.last\_attn = None  # store attention weights



&nbsp;   def forward(self, x):

&nbsp;       x = self.embedding(x)



&nbsp;       attn\_output, attn\_weights = self.attn\_layer(x, x, x, need\_weights=True)

&nbsp;       self.last\_attn = attn\_weights.detach().cpu()



&nbsp;       x = self.transformer(attn\_output)

&nbsp;       x = x\[:, -1, :]

&nbsp;       return self.fc(x)



\# ==============================================================

\# 4. LSTM BASELINE MODEL

\# ==============================================================



class LSTMForecast(nn.Module):

&nbsp;   def \_\_init\_\_(self, input\_dim=1, hidden=64, horizon=10):

&nbsp;       super().\_\_init\_\_()

&nbsp;       self.lstm = nn.LSTM(input\_dim, hidden, batch\_first=True)

&nbsp;       self.fc = nn.Linear(hidden, horizon)



&nbsp;   def forward(self, x):

&nbsp;       \_, (h, \_) = self.lstm(x)

&nbsp;       return self.fc(h\[-1])



\# ==============================================================

\# 5. TRAINING WITH LR SCHEDULER

\# ==============================================================



def train(model, X, y, epochs=30, lr=0.001):

&nbsp;   model.to(DEVICE)

&nbsp;   opt = torch.optim.Adam(model.parameters(), lr=lr)

&nbsp;   scheduler = torch.optim.lr\_scheduler.StepLR(opt, step\_size=12, gamma=0.5)

&nbsp;   loss\_fn = nn.MSELoss()



&nbsp;   for epoch in range(epochs):

&nbsp;       model.train()

&nbsp;       opt.zero\_grad()



&nbsp;       preds = model(X)

&nbsp;       loss = loss\_fn(preds, y.squeeze())



&nbsp;       loss.backward()

&nbsp;       opt.step()

&nbsp;       scheduler.step()



&nbsp;       if (epoch + 1) % 5 == 0:

&nbsp;           print(f"Epoch {epoch+1}/{epochs}  Loss: {loss.item():.4f}")



\# ==============================================================

\# 6. METRICS: RMSE, MASE, sMAPE

\# ==============================================================



def rmse(true, pred):

&nbsp;   return np.sqrt(mean\_squared\_error(true, pred))





def mase(true, pred):

&nbsp;   naive = np.mean(np.abs(np.diff(true)))

&nbsp;   return np.mean(np.abs(true - pred)) / naive





def smape(true, pred):

&nbsp;   return 100 \* np.mean(2 \* np.abs(pred - true) / (np.abs(true) + np.abs(pred) + 1e-8))





def evaluate(model, X, y):

&nbsp;   model.eval()

&nbsp;   with torch.no\_grad():

&nbsp;       preds = model(X).cpu().numpy()

&nbsp;       true = y.cpu().numpy()



&nbsp;   return rmse(true.flatten(), preds.flatten()), mase(true.flatten(), preds.flatten()), smape(true.flatten(), preds.flatten()), preds



\# ==============================================================

\# 7. TRAIN BOTH MODELS

\# ==============================================================



transformer = TransformerForecast(horizon=HORIZON)

lstm = LSTMForecast(horizon=HORIZON)



print("\\nTraining Transformer...")

train(transformer, X\_train, y\_train, EPOCHS, LR)



print("\\nTraining LSTM...")

train(lstm, X\_train, y\_train, EPOCHS, LR)



\# ==============================================================

\# 8. EVALUATION RESULTS

\# ==============================================================



tr\_rmse, tr\_mase, tr\_smape, tr\_preds = evaluate(transformer, X\_test, y\_test)

ls\_rmse, ls\_mase, ls\_smape, ls\_preds = evaluate(lstm, X\_test, y\_test)



print("\\n===== FINAL METRICS =====")

print(f"Transformer → RMSE: {tr\_rmse:.4f} | MASE: {tr\_mase:.4f} | sMAPE: {tr\_smape:.2f}%")

print(f"LSTM        → RMSE: {ls\_rmse:.4f} | MASE: {ls\_mase:.4f} | sMAPE: {ls\_smape:.2f}%")



\# ==============================================================

\# 9. FORECAST VISUALIZATION

\# ==============================================================



true\_vals = y\_test.cpu().numpy().flatten()



plt.figure()

plt.plot(true\_vals, label="True")

plt.plot(tr\_preds.flatten(), label="Transformer")

plt.plot(ls\_preds.flatten(), label="LSTM")

plt.title("Forecast Comparison")

plt.legend()

plt.show()



\# ==============================================================

\# 10. ATTENTION WEIGHT VISUALIZATION 

\# ==============================================================



if transformer.last\_attn is not None:

&nbsp;   attn = transformer.last\_attn\[0].mean(0).numpy()  # average heads



&nbsp;   plt.figure()

&nbsp;   plt.imshow(attn, aspect="auto")

&nbsp;   plt.colorbar()

&nbsp;   plt.title("Transformer Attention Weights")

&nbsp;   plt.xlabel("Time Step")

&nbsp;   plt.ylabel("Time Step")

&nbsp;   plt.show()



