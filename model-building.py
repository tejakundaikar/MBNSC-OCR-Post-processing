from transformers import (
    BertConfig,
    BertTokenizerFast,
    BertForMaskedLM,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# Load tokenizer from local directory
tokenizer = BertTokenizerFast.from_pretrained("/content/sample_data/tokenizer/")

# Define BERT configuration (custom small model)
config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    max_position_embeddings=512,
    type_vocab_size=2,
)

# Initialize model from config (not from pre-trained "bert-base-uncased")
model = BertForMaskedLM(config=config)

# Load dataset from plain text (each line is a training sample)
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/content/sample_data/train_corpus.txt",
    block_size=128,
)

# Create data collator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="/content/sample_data/model/",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train model
trainer.train()

# Save model and tokenizer
model1=trainer.save_model("/content/sample_data/model/")
tokenizer.save_pretrained("/content/sample_data/model/")