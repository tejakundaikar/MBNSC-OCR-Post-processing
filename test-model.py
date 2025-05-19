import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from nltk.tokenize import WordPunctTokenizer
# import textdistance
import numpy as np
import time
import textdistance
top_k=5
text="सुत्रां [MASK] तांणी एके बायले"
text = tokenizer.tokenize(text)
text.insert(0,'[CLS]')
text.insert(len(text),'[SEP]')
text
for i,token in enumerate(text):
    copy_text = text[:]

    if token not in ('[CLS]','[SEP]'):
        print(copy_text[i])
        original_token = copy_text[i]
        copy_text[i]='[MASK]'
        copy_text = ' '.join(copy_text)
        tokenized_text = tokenizer.tokenize(copy_text)

        masked_index = tokenized_text.index('[MASK]')
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # Create the segments tensors.
        segments_ids = [0] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Load pre-trained model (weights)

        model.eval()

        # Predict all tokens
        with torch.no_grad():
            predictions = model(tokens_tensor, segments_tensors)

        probs = torch.nn.functional.softmax(predictions[0][0][masked_index], dim=-1)

        # All redicted tokens are selected here. len(probs) can be replaced with any number suitable to the use case
        top_k_weights, top_k_indicies = torch.topk(probs, len(probs), sorted=True)

        output_dict={}
        for i, pred_idx in enumerate(top_k_indicies):
            predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
            token_weight = top_k_weights[i]
            if len(original_token)>3:
                if textdistance.levenshtein.normalized_similarity(predicted_token,original_token)>0.5:
                    output_dict[predicted_token]=float(token_weight)
            else:
                output_dict[predicted_token]=float(token_weight)
        output_dict = dict(list(output_dict.items())[0: top_k])
        print(output_dict)