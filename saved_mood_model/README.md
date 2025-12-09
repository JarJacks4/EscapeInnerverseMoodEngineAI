# Model File Placeholder

The trained RoBERTa model file (`model.safetensors`) is approximately 475MB and exceeds GitHub's file size limit.

## To get the complete trained model:

### Option 1: Train the model yourself
```bash
python train_mood_model.py
```
This will download the base RoBERTa model and fine-tune it on the emotion dataset, creating the `model.safetensors` file.

### Option 2: Download from releases
Check the [Releases](https://github.com/epsilon403/mood_engine/releases) section for pre-trained model downloads.

### Option 3: Use base model fallback
The API automatically falls back to the base `roberta-base` model if the fine-tuned model is not found, though with reduced accuracy.

## Model Performance
- **Architecture**: Fine-tuned RoBERTa-base
- **Dataset**: dair-ai/emotion (16,000 samples) 
- **Accuracy**: 93.0% on test set
- **File Size**: ~475MB
- **Training Time**: ~1 hour on RTX 2000 Ada GPU

## File Structure
```
saved_mood_model/
├── config.json              # Model configuration
├── merges.txt               # BPE merges file
├── model.safetensors        # [LARGE FILE] Trained model weights
├── special_tokens_map.json  # Special token mappings
├── tokenizer_config.json    # Tokenizer configuration
└── vocab.json              # Vocabulary file
```

The model will be automatically created in this directory when you run the training script.