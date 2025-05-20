import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load tokenizer and model once
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def get_mask_suggestion(text, top_k=1):
    """
    Returns the top suggestion(s) for the [MASK] token in the input text.
    
    Parameters:
        text (str): Input sentence with exactly one [MASK] token
        top_k (int): Number of top suggestions to return
    
    Returns:
        list of (token, probability) tuples, or a single token string if top_k=1
    """

    if '[MASK]' not in text:
        raise ValueError("The input text must contain a [MASK] token.")

    # Tokenize and prepare inputs
    tokenized_text = tokenizer.tokenize(text)
    tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']
    masked_index = tokenized_text.index('[MASK]')
    
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs.logits

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_weights, top_indices = torch.topk(probs, top_k, sorted=True)

    # Convert token IDs to strings
    results = [
        (tokenizer.convert_ids_to_tokens([idx])[0], float(prob))
        for idx, prob in zip(top_indices, top_weights)
    ]

    return results[0][0] if top_k == 1 else results


text = "सुत्रां [MASK] तांणी एके बायले"
suggestion = get_mask_suggestion(text)
print("Top suggestion:", suggestion)

# Get top 6 suggestions
suggestions = get_mask_suggestion(text, top_k=6)
for token, prob in suggestions:
    print(f"{token} : {prob:.4f}")
