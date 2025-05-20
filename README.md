# MBNSC-OCR-Post-processing
```
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
├──norvig_spell.py
```


### Prerequisites

- Python 3.8+
- Recommended packages:
  - `transformers`
  - `tokenizers`
  - `torch`
  - `numpy`

Install dependencies via pip:

pip install transformers tokenizers torch numpy

### 1. Prepare Data

Ensure your training corpus is saved as:

```bash
data/train_corpus.txt
```

### 2. Build the Tokenizer

```bash
python build-tokenizer.py
```

Generates tokenizer files into the `tokenizer/` directory.

### 3. Train the Model

```bash
python model-building.py
```

Model outputs will be stored in the `model/` folder.

### 4. Evaluate the Model

```bash
python test-model.py
```

Test your model with various inputs and check the outputs.
