import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuration
MODEL_NAME = "roberta-base"
OUTPUT_DIR = "./saved_mood_model"
NUM_EPOCHS = 3 # Increase for better accuracy
BATCH_SIZE = 16

def compute_metrics(eval_pred):
    """Helper to calculate comprehensive metrics during training"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )
    
    return {
        "accuracy": accuracy,
        "f1_weighted": f1,
        "f1_macro": f1_macro,
        "precision_weighted": precision,
        "precision_macro": precision_macro,
        "recall_weighted": recall,
        "recall_macro": recall_macro,
    }

def train_model():
    print(f"Loading {MODEL_NAME} and Emotion dataset...")
    
    # 2. Load Data (Using dair-ai/emotion as a standard 'imotion' dataset)
    # Labels: 0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise
    dataset = load_dataset("dair-ai/emotion")
    
    # 3. Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 4. Model Setup
    # Mapping labels to our API's expected mood strings
    id2label = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
    label2id = {v: k for k, v in id2label.items()}

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=6, 
        id2label=id2label, 
        label2id=label2id
    )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7. Train
    print("Starting training...")
    trainer.train()

    # 8. Final Evaluation on Test Set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print("\nTest Set Metrics:")
    for key, value in test_results.items():
        if key.startswith('eval_'):
            metric_name = key.replace('eval_', '').title()
            if isinstance(value, float):
                print(f"{metric_name:20}: {value:.4f}")
            else:
                print(f"{metric_name:20}: {value}")
    
    # 9. Detailed Analysis with Predictions
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION ANALYSIS")
    print("="*60)
    
    # Get predictions for test set
    test_predictions = trainer.predict(tokenized_datasets["test"])
    y_pred = np.argmax(test_predictions.predictions, axis=-1)
    y_true = test_predictions.label_ids
    
    # Class names for better readability
    class_names = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Emotion Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved as 'confusion_matrix.png'")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, (class_name, acc) in enumerate(zip(class_names, class_accuracy)):
        print(f"{class_name:12}: {acc:.4f} ({cm.diagonal()[i]}/{cm.sum(axis=1)[i]} correct)")
    
    # Sample predictions analysis
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS ANALYSIS")
    print("="*60)
    
    # Get some sample texts and their predictions
    test_texts = tokenized_datasets["test"]["text"][:20]
    test_labels_sample = y_true[:20]
    test_preds_sample = y_pred[:20]
    
    # Get confidence scores for samples
    probabilities = torch.softmax(torch.tensor(test_predictions.predictions[:20]), dim=-1)
    confidences = torch.max(probabilities, dim=-1)[0].numpy()
    
    print("\nSample Predictions (first 20 from test set):")
    print("-" * 120)
    print(f"{'Text':<40} {'True':<10} {'Pred':<10} {'Conf':<6} {'Correct'}")
    print("-" * 120)
    
    for i in range(20):
        text_preview = test_texts[i][:35] + "..." if len(test_texts[i]) > 35 else test_texts[i]
        true_label = class_names[test_labels_sample[i]]
        pred_label = class_names[test_preds_sample[i]]
        confidence = confidences[i]
        is_correct = "✓" if test_labels_sample[i] == test_preds_sample[i] else "✗"
        
        print(f"{text_preview:<40} {true_label:<10} {pred_label:<10} {confidence:<6.3f} {is_correct}")
    
    # 10. Save the final model for the API to use
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training metrics
    with open("training_metrics.txt", "w") as f:
        f.write("FINAL TRAINING METRICS\n")
        f.write("="*50 + "\n\n")
        f.write("Test Set Results:\n")
        for key, value in test_results.items():
            if key.startswith('eval_'):
                f.write(f"{key.replace('eval_', '').title()}: {value:.4f}\n")
        
        f.write(f"\nOverall Test Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"Per-Class Accuracy:\n")
        for i, (class_name, acc) in enumerate(zip(class_names, class_accuracy)):
            f.write(f"  {class_name}: {acc:.4f}\n")
    
    print("Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("Confusion matrix saved as: confusion_matrix.png")
    print("Training metrics saved as: training_metrics.txt")

if __name__ == "__main__":
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (Training will be slow)")
    
    train_model()
