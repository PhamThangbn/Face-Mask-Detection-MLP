from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import pickle
import os
import cv2
import numpy as np
import glob
import time
import math
import matplotlib.pyplot as plt

# Your configuration and data
DATASET_FOLDER = r"dataset"
LABELS = ["with_mask", "without_mask"]

# Load data and labels
data = []
labels = []

# Load images from the folder
for label in LABELS:
    path = os.path.join(DATASET_FOLDER, label)
    files = glob.glob(os.path.join(path, "*.jpg")) + glob.glob(os.path.join(path, "*.png"))
    for file in files:
        image = cv2.imread(file)
        image = cv2.resize(image, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        data.append(image)
        labels.append(label)

# Convert labels to binary format
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
data = np.array(data, dtype="float32")
labels = np.ravel(labels)

# Split the data into training and testing sets
train_X, test_X, train_Y, test_Y = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=math.floor(time.time())
)

# Normalize the data
train_mean = np.mean(train_X, axis=0)
train_X -= train_mean
test_X -= train_mean

scaler = StandardScaler()
train_X = scaler.fit_transform(train_X.reshape(train_X.shape[0], -1))
test_X = scaler.transform(test_X.reshape(test_X.shape[0], -1))

# MLPClassifier model with early stopping and learning rate scheduler
model = MLPClassifier(
    hidden_layer_sizes=(128,),
    max_iter=1,
    alpha=1e-4,
    solver='adam',
    random_state=math.floor(time.time()),
    verbose=False,
    learning_rate_init=1e-4,
    warm_start=True,
    early_stopping=True,
)

# Train the model
train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []

for epoch in range(1, 51):  # Train for 50 epochs
    model.fit(train_X, train_Y)

    # Predictions
    train_pred = model.predict(train_X)
    test_pred = model.predict(test_X)

    # Evaluate accuracy
    train_acc = accuracy_score(train_Y, train_pred)
    test_acc = accuracy_score(test_Y, test_pred)

    # Evaluate loss
    train_loss = log_loss(train_Y, model.predict_proba(train_X))
    test_loss = log_loss(test_Y, model.predict_proba(test_X))

    # Store values
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f"Epoch {epoch}/50 - Train Acc: {train_acc*100:.2f}% - Test Acc: {test_acc*100:.2f}% - Loss: {test_loss:.4f}")

# Evaluate the model
print("\nEvaluating the model...")
print("Training set size:", len(train_X))
print("Testing set size:", len(test_X))

print("\nClassification Report (Train):")
print(classification_report(train_Y, train_pred, target_names=LABELS))
print("\nClassification Report (Test):")
print(classification_report(test_Y, test_pred, target_names=LABELS))

# Save the model
save_dir = r"model"
os.makedirs(save_dir, exist_ok=True)
file_name = os.path.join(save_dir, "Model_MLP_new.pkl")
with open(file_name, 'wb') as f:
    pickle.dump(model, f)
print("\nSaved the trained model to file:", file_name)

# Plot accuracy and loss graphs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
