import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('dataset_simple.csv')

X = torch.Tensor(df[['age']].values)
y = torch.Tensor(df['income'].values)

X_mean, X_std = X.mean(), X.std()
y_mean, y_std = y.mean(), y.std()

Xn = (X - X_mean) / X_std
yn = (y - y_mean) / y_std

class NNet_regression(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(NNet_regression, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )
    def forward(self, X):
        return self.layers(X)

net = NNet_regression(in_size=1, hidden_size=5, out_size=1)
lossFn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  

epochs = 4000
for i in range(epochs):
    pred = net(Xn)
    loss = lossFn(pred.squeeze(), yn)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 200 == 0:
        print(f'Эпоха [{i+1}/{epochs}], Ошибка: {loss.item():.6f}')

with torch.no_grad():
    pred_norm = net(Xn)
    pred_real = pred_norm * y_std + y_mean

print("\nПервые предсказания (доход в реальных единицах):")
for a, p in zip(df['age'], pred_real.squeeze().numpy()):
    print(f'Возраст {a}: предсказанный доход ≈ {p:.2f}')

mae = torch.mean(abs(y - pred_real.squeeze()))
print(f'\nСредняя абсолютная ошибка : {mae.item():.2f}')

plt.figure(figsize=(6,4))
plt.scatter(df['age'], df['income'], color='blue', label='Реальные данные')
plt.plot(df['age'], pred_real.detach().numpy(), color='red', label='Модель')
plt.xlabel('Возраст')
plt.ylabel('Доход')
plt.title('Предсказание дохода по возрасту')
plt.legend()
plt.show()
