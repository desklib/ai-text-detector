# Desklib AI Text Detector

This repository contains the code and resources for an AI-generated text detection model developed by Desklib. This model is designed to classify English text as either human-written or AI-generated.

## Overview

The model is a fine-tuned version of **microsoft/deberta-v3-large**, leveraging a transformer-based architecture to achieve high accuracy in identifying AI-generated content.  It is robust against various adversarial attacks across different text domains, making it a reliable tool for detecting AI-generated text in various contexts. This model is particularly useful in content moderation, academic integrity, journalism, and other applications where the authenticity of text is crucial.

**Key Features:**

*   **Robust Detection:**  Effectively identifies AI-generated text, even with adversarial modifications.
*   **High Accuracy:**  Achieves leading performance on the [RAID Benchmark for AI Detection](https://raid-bench.xyz/leaderboard?domain=all&decoding=all&repetition=all&attack=all) at time of submission.
*   **Easy to Use:**  Simple integration with the Hugging Face `transformers` library.
*   **Based on DeBERTa:**  Leverages the powerful `microsoft/deberta-v3-large` transformer model.
* **Developed by Desklib**: Desklib provides AI based tools for students, educators and universities.

**Links:**

*   **Hugging Face Model Hub:** [https://huggingface.co/desklib/ai-text-detector-v1.01](https://huggingface.co/desklib/ai-text-detector-v1.01)
*   **Try the model online!**: [Desklib AI Detector](https://desklib.com/ai-content-detector/)
* **RAID Benchmark Leaderboard**: [Visit RAID Leaderboard](https://raid-bench.xyz/leaderboard?domain=all&decoding=all&repetition=all&attack=all)
*  **Github Repo**: [https://github.com/desklib/ai-text-detector](https://github.com/desklib/ai-text-detector)

## Installation
This project requires Python 3.7+ and PyTorch.

1.  **Install dependencies:**

    ```bash
    pip install torch transformers
    ```
    (It is highly recommended to use a virtual environment (like `venv` or `conda`) to avoid conflicts with other projects)

## Usage

The script provides a simple example of how to use the model to predict whether a given text is AI-generated.  The core logic is encapsulated in the `predict_single_text` function.

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel

class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        # Initialize the base transformer model.
        self.model = AutoModel.from_config(config)
        # Define a classifier head.
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize weights (handled by PreTrainedModel)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through the transformer
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Classifier
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output

def predict_single_text(text, model, tokenizer, device, max_len=768, threshold=0.5):
    """
        Predicts whether the given text is AI-generated.
    """
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probability = torch.sigmoid(logits).item()

    label = 1 if probability >= threshold else 0
    return probability, label

def main():
    # --- Model and Tokenizer Directory ---
    model_directory = "desklib/ai-text-detector-v1.01"

    # --- Load tokenizer and model ---
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = DesklibAIDetectionModel.from_pretrained(model_directory)

    # --- Set up device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Example Input text ---
    text_ai = "AI detection refers to the process of identifying whether a given piece of content, such as text, images, or audio, has been generated by artificial intelligence. This is achieved using various machine learning techniques, including perplexity analysis, entropy measurements, linguistic pattern recognition, and neural network classifiers trained on human and AI-generated data. Advanced AI detection tools assess writing style, coherence, and statistical properties to determine the likelihood of AI involvement. These tools are widely used in academia, journalism, and content moderation to ensure originality, prevent misinformation, and maintain ethical standards. As AI-generated content becomes increasingly sophisticated, AI detection methods continue to evolve, integrating deep learning models and ensemble techniques for improved accuracy."
    text_human = "It is estimated that a major part of the content in the internet will be generated by AI / LLMs by 2025. This leads to a lot of misinformation and credibility related issues. That is why if is important to have accurate tools to identify if a content is AI generated or human written"

    # --- Run prediction ---
    probability, predicted_label = predict_single_text(text_ai, model, tokenizer, device)
    print(f"Probability of being AI generated: {probability:.4f}")
    print(f"Predicted label: {'AI Generated' if predicted_label == 1 else 'Not AI Generated'}")

    probability, predicted_label = predict_single_text(text_human, model, tokenizer, device)
    print(f"Probability of being AI generated: {probability:.4f}")
    print(f"Predicted label: {'AI Generated' if predicted_label == 1 else 'Not AI Generated'}")

if __name__ == "__main__":
    main()
