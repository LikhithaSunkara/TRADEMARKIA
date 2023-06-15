import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define the architecture of the classification model
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return out

# Split the BERT-encoded data and class labels into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(contextualized_reps, classes, test_size=0.2, random_state=42)

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Encode the class labels
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

# Convert the encoded labels to tensors
y_train = torch.tensor(y_train_encoded)
y_val = torch.tensor(y_val_encoded)

# Define the model
input_size = contextualized_reps.shape[1]
num_classes = len(label_encoder.classes_)  # Total number of classes
model = Classifier(input_size, num_classes)

# Define the optimizer, loss function, and evaluation metric
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Define the training loop
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    correct = 0

    # Perform mini-batch training
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i+batch_size]
        labels = y_train[i:i+batch_size]

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

    # Compute accuracy and average loss for the epoch
    train_accuracy = correct / len(X_train)
    train_loss = epoch_loss / (len(X_train) // batch_size)

    # Evaluate on the validation set
    model.eval()
    val_outputs = model(X_val)
    val_loss = criterion(val_outputs, y_val)
    _, val_predicted = torch.max(val_outputs.data, 1)
    val_accuracy = (val_predicted == y_val).sum().item() / len(X_val)

    # Print training and validation metrics
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
    print()

# Fine-tune hyperparameters as needed
# ...

# Save the trained model
torch.save(model.state_dict(), 'classifier_model.pth')