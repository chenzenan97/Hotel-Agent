import os 
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.utils.data import Dataset, random_split
# Define the dataset class for the CSV dataset
class CsvDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data.iloc[idx]["Question"]
        answer = self.data.iloc[idx]["Answer"]
        conversation = f"User:{question}Hotel Agent:{answer}{self.tokenizer.eos_token}"
        #inputs = self.tokenizer(conversation, truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs = self.tokenizer(conversation, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        inputs['labels'] = inputs['input_ids'].clone()
        return inputs

# Load the DialoGPT-medium model and tokenizer
model_name = "microsoft/DialoGPT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# Read the CSV dataset
dataset_file = "./data/QANDA.csv"
data = pd.read_csv(dataset_file)

# Create CsvDataset and split into train and validation sets
full_dataset = CsvDataset(data, tokenizer)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./model/hotel_agent",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    save_steps=300,
    save_total_limit=2,
    evaluation_strategy="steps",
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train and validate the model
trainer.train()
print("Finished training")
output_dir = "./model/hotel_agent"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
