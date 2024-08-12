class LSTMModel(nn.Module):
    def __init__(self, words, embedding_dim, hidden_dim, output_size, num_layers):
        super(LSTMModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(words, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(0.5)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        # Get the output from the last layer of LSTM for each sentence
        lstm_out = self.dropout(lstm_out)  # Apply dropout
        final_feat = lstm_out[:, -1, :]
        final_out = self.fc(final_feat)
        return final_out


# Hyperparameters
words = total_words  # Total vocabulary size
embedding_size = 128  # Size of the embedding vectors
hidden_dim = 64  # LSTM hidden states dimension
output_size = 3  # Number of classes (positive, neutral, negative)
num_layers = 2  # Number of LSTM layers

# Initialize model, loss function, and optimizer
model = LSTMModel(words, embedding_size, hidden_dim, output_size, num_layers)
lossfun = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


def ApplyLstm(model, dataloader, optimizer, lossfun, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for text, labels in dataloader:
            optimizer.zero_grad()
            output = model(text)  # Forward pass
            # print(output)
            # Calculate the loss
            loss = lossfun(output, labels)
            total_loss += loss.item()

            # Make predictions
            predicted_classes = torch.argmax(output, dim=1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted_classes == labels).sum().numpy()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions * 100

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)  # This will transform -1, 0, 1 to 0, 1, 2

y_encoded = torch.tensor(y_encoded).long()
print(X[0])
print(y_encoded)
dataset = TensorDataset(X, y_encoded)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
# Training the model
ApplyLstm(model, dataloader, optimizer, lossfun, epochs=50)

test = pd.read_csv("/kaggle/input/neural/test _no_label.csv")
# Apply preprocessing steps to the 'review_description' column of the test data
test['review_description'] = test['review_description'].apply(remove_duplicate_arabic_words)
test['review_description'] = test['review_description'].apply(
    lambda x: " ".join([word for word in x.split() if word not in stopWords]))
test['review_description'] = test['review_description'].apply(lambda x: emojiTextTransform(x))
test['review_description'] = test['review_description'].apply(
    lambda x: ''.join([word for word in x if not word.isdigit()]))
test['review_description'] = test['review_description'].apply(
    lambda x: " ".join([arabic_stemmer.light_stem(word) for word in x.split()]))

# Convert test data to padded sequences
test_seq = tokenizer.texts_to_sequences(test['review_description'])
test_seq_pad = pad_sequences(test_seq, maxlen=75, padding="post", truncating="post")
X_test = torch.from_numpy(test_seq_pad).long()

# Predict using the trained model
model.eval()
with torch.no_grad():
    output = model(X_test)
    predicted_classes = torch.argmax(output, dim=1)

# Decode the predicted classes back to the original labels using label encoder
predicted_labels = le.inverse_transform(predicted_classes.numpy())
