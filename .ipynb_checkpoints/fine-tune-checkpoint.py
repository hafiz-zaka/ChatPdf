import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from langchain_community.document_loaders import PDFMinerLoader

def load_documents():
    documents = []
    for root, _, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                loader = PDFMinerLoader(os.path.join(root, file))
                documents.extend(loader.load())
    return documents

def preprocess_function(examples, tokenizer):
    inputs = [doc['content'] for doc in examples]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    return model_inputs

def main():
    checkpoint = "MBZUAI/LaMini-T5-738M"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    
    documents = load_documents()
    dataset = Dataset.from_list([{"content": doc.page_content} for doc in documents])
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=3,
        save_steps=10,
        logging_dir="./logs",
        logging_steps=10,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    trainer.save_model("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

if __name__ == "__main__":
    main()
