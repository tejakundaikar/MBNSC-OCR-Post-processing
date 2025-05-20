import torch
from transformers import BertTokenizer, BertForMaskedLM
from norvig_spell import NorvigSpellCorrector

# Load model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('XXX')
# model = BertForMaskedLM.from_pretrained('XXX')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load Norvig spell corrector
norvig = NorvigSpellCorrector('/data/train_corpus.txt')

def get_mask_suggestion(text, top_k=1):
    """Returns top suggestion(s) from MLM for [MASK] token."""
    tokenized_text = tokenizer.tokenize(text)
    tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']
    masked_index = tokenized_text.index('[MASK]')
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)
    
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)
    
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs.logits

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_weights, top_indices = torch.topk(probs, top_k, sorted=True)

    top_token = tokenizer.convert_ids_to_tokens([top_indices[0]])[0]
    return top_token, float(top_weights[0])


def are_lengths_similar(word1, word2, max_diff=2):
    """
    Returns True if the length difference between two words is <= max_diff.
    
    Args:
        word1 (str): First word.
        word2 (str): Second word.
        max_diff (int): Maximum allowed difference in length (default is 2).
        
    Returns:
        bool: True if the words have similar lengths, False otherwise.
    """
    return abs(len(word1) - len(word2)) <= max_diff
    
def select_best_suggestion(original_word, masked_text):
    """
    Combines Norvig and MLM suggestions based on defined conditions.
    
    Params:
        original_word (str): The possibly incorrect word
        masked_text (str): The sentence with [MASK] in place of the original word
    
    Returns:
        str: Selected suggestion
    """
    # Norvig correction and probability
    norvig_suggestion = norvig.correction(original_word)
    norvig_prob = norvig.probability(norvig_suggestion)

    # MLM suggestion
    mlm_suggestion, mlm_prob = get_mask_suggestion(masked_text)

    # Rule-based decision
    if norvig_prob == 1:
        return norvig_suggestion
    elif are_lengths_similar(mlm_suggestion, original_word):
        return mlm_suggestion
    else:
        return original_word  # fallback

# Example usage
if __name__ == "__main__":
    ocred_original_word = "सांग"
    text_with_mask = "सुत्रां [MASK] तांणी एके बायले"
    final_suggestion = select_best_suggestion(ocred_original_word, text_with_mask)
    print("Selected Suggestion:", final_suggestion)
