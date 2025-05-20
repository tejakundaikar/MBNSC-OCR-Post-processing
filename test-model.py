import torch
from transformers import BertTokenizer, BertForMaskedLM
import textdistance

# Parameters
top_k = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
model.to(device)
model.eval()

# Input sentence (must include [MASK])
text = "सुत्रां [MASK] तांणी एके बायले"

# Tokenize input
tokenized_text = tokenizer.tokenize(text)
tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']

# Get mask index
masked_index = tokenized_text.index('[MASK]')

# Convert tokens to IDs
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [0] * len(tokenized_text)

# Convert to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens]).to(device)
segments_tensors = torch.tensor([segments_ids]).to(device)

# Predict masked token
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs.logits

# Softmax to get probabilities
probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
top_k_weights, top_k_indices = torch.topk(probs, probs.size(-1), sorted=True)

# Optional: similarity filtering (if needed)
original_token = ""  # You can specify expected token for correction logic
suggestions = {}

for idx, pred_idx in enumerate(top_k_indices):
    predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
    prob = float(top_k_weights[idx])

    # Optional: Filter by similarity if original_token is known
    if original_token and len(original_token) > 3:
        if textdistance.levenshtein.normalized_similarity(predicted_token, original_token) > 0.5:
            suggestions[predicted_token] = prob
    else:
        suggestions[predicted_token] = prob

    if len(suggestions) >= top_k:
        break

# Output suggestions
print("Top suggestions for [MASK]:")
for token, score in suggestions.items():
    print(f"{token}: {score:.4f}")
