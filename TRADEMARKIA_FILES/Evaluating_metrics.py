from sklearn.metrics import classification_report

# Evaluate the trained model on the testing set
model.eval()
test_outputs = model(X_test)
_, test_predicted = torch.max(test_outputs.data, 1)

# Convert test labels to numpy array for evaluation
y_test = y_test.numpy()
test_predicted = test_predicted.numpy()

# Calculate evaluation metrics
accuracy = (test_predicted == y_test).sum() / len(y_test)
report = classification_report(y_test, test_predicted)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Classification Report:\n{report}")