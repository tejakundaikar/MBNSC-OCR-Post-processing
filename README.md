# MBNSC-OCR-Post-processing

MBNSC-OCR-Post-processing/
│
├── data/
│   └── train_corpus.txt  ← your large raw text corpus
│
├── tokenizer/
│   └── (WordPiece vocab files will go here)
│
├── model/
│   └── (Trained model will go here)
│
├── build-tokenizer.py
├── model-building.py
├── test-model.py
├──spell.py



### Prerequisites

- Python 3.8+
- Recommended packages:
  - `transformers`
  - `tokenizers`
  - `torch`
  - `numpy`

Install dependencies via pip:

pip install transformers tokenizers torch numpy

-1. Prepare Your Data
Place your raw text corpus in data/train_corpus.txt.
-2. Build the Tokenizer
python build-tokenizer.py
This will generate a tokenizer and save vocab files in the tokenizer/ directory.
-3. Train the Model
python model-building.py
Model checkpoints and config files will be saved in the model/ directory.
-4. Test the Model
 python test-model.py
