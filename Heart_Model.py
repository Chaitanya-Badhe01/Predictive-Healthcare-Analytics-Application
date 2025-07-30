import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, in_features=18, h1=64, h2=28, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

def graph(epochs: int, losses: list[float]) -> None:
    plt.plot(range(epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_plot.png')
    plt.show()

# Set random seed for reproducibility
torch.manual_seed(392)

# Initialize model
model = Model()

# Load dataset
dataframe = pd.read_csv(r'C:\Users\LENOVO\Downloads\DHONI\heart_disease_risk_dataset_earlymed.csv')

# Split features and target
X = dataframe.drop('Heart_Risk', axis=1).values
y = dataframe['Heart_Risk'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=392)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

# Save training data for LIME explanations
np.save('heart_training_data.npy', X_train)
print("Saved heart training data with shape:", X_train.shape)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training loop
epochs = 1000
losses = []
print('Training...')
for i in range(1, epochs + 1):
    y_pred = model.forward(X_train_tensor)

    loss = criterion(y_pred, y_train_tensor)
    losses.append(loss.item())

    if i % 10 == 0:
        print(f'Epoch: {i} Loss: {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('Training finished!\n')

# Plot training loss
graph(epochs, losses)

# Test individual predictions
print(f'Testing...')
with torch.no_grad():
    for i, y in enumerate(X_test_tensor[:5]):  # Show only first 5 test samples for brevity
        y_val = model.forward(y)
        print(f'Test {i+1:3d} | Prediction: {y_val.tolist()} | Actual: {y_test_tensor[i]} | Class: {y_val.argmax().item()}')

# Evaluate model accuracy
with torch.no_grad():
    predictions = model(X_test_tensor).argmax(dim=1)
correct = (predictions == y_test_tensor).sum().item()
print(f'{correct}/{len(y_test_tensor)} correct! ({accuracy_score(y_test_tensor, predictions) * 100:.2f}%)')

# Save the trained model
torch.save(model.state_dict(), 'heart_model.pth')
print("Model saved as heart_model.pth")

# Create an explanation of the feature columns for reference
feature_names = dataframe.drop('Heart_Risk', axis=1).columns.tolist()
print("\nFeature names for reference:")
for i, name in enumerate(feature_names):
    print(f"{i}: {name}")

# Save feature names if needed
with open('heart_feature_names.txt', 'w') as f:
    for name in feature_names:
        f.write(f"{name}\n")