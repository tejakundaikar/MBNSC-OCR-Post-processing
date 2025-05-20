#https://github.com/huggingface/tokenizers/blob/main/bindings/python/examples/train_bert_wordpiece.py
from tokenizers import BertWordPieceTokenizer

# Paths
input_file = "/data/train_corpus.txt"
tokenizer_path = "/tokenizer/"

# Initialize tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False,
)

# Train tokenizer
tokenizer.train(
    files=[input_file],
    vocab_size=30000,
    min_frequency=2,
    limit_alphabet=1000,
    wordpieces_prefix="##",
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Save tokenizer
tokenizer.save_model(tokenizer_path)
