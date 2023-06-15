import torch
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence

# Load the BERT pre-trained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Tokenize the preprocessed text data using BERT tokenizer
tokenized_descriptions = [tokenizer.tokenize(" ".join(desc)) for desc in preprocessed_descriptions]

# Process the data in batches
batch_size = 16
num_batches = (len(tokenized_descriptions) + batch_size - 1) // batch_size

contextualized_reps = []

# Iterate over batches
for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(tokenized_descriptions))

    # Tokenize and convert to input IDs
    batch_tokenized = tokenized_descriptions[start_idx:end_idx]
    input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in batch_tokenized]

    # Pad the input sequences
    padded_input_ids = pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True)

    # Convert input IDs to LongTensor
    padded_input_ids = padded_input_ids.long()

    # Encode the input features using the BERT model
    with torch.no_grad():
        encoded_layers = model(padded_input_ids)[0]

    # Get the contextualized representations for each description
    batch_contextualized_reps = encoded_layers[:, 0, :]
    contextualized_reps.append(batch_contextualized_reps)

# Concatenate the representations from all batches
contextualized_reps = torch.cat(contextualized_reps, dim=0)

# Print the shape of the contextualized representations
print("Shape of contextualized representations:", contextualized_reps.shape)