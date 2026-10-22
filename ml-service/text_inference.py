import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Fallback to the cached/pretrained huggingface repository if local weights are not on disk
DEFAULT_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
LOCAL_MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "models",
    "text_emotion_model"
)


class TextEmotionPredictor:

    def __init__(self, model_dir=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading text emotion model on: {self.device}")

        # Determine target model directory or Hugging Face ID
        target_dir = model_dir or LOCAL_MODEL_DIR
        has_local_weights = (
            os.path.isdir(target_dir)
            and any(
                f in ("pytorch_model.bin", "model.safetensors")
                for f in os.listdir(target_dir)
            )
        )

        model_source = target_dir if has_local_weights else DEFAULT_MODEL_NAME

        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_source
        ).to(self.device)
        self.model.eval()

        # Load emotion metadata if available
        meta_path = os.path.join(
            target_dir if os.path.isdir(target_dir) else "",
            "emotion_config.json"
        )

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.id2label = {
                int(k): v
                for k, v in meta.get("id2label", {}).items()
            }
            self.classes = meta.get("target_emotions", list(self.id2label.values()))
        elif hasattr(self.model.config, "id2label") and self.model.config.id2label:
            self.id2label = {
                int(k): v
                for k, v in self.model.config.id2label.items()
            }
            self.classes = list(self.id2label.values())
        else:
            self.id2label = {
                0: "anger",
                1: "disgust",
                2: "fear",
                3: "joy",
                4: "neutral",
                5: "sadness",
                6: "surprise"
            }
            self.classes = list(self.id2label.values())

        print("Text emotion model loaded successfully!")

    def predict(self, text: str, max_chunk_size: int = 480, overlap: int = 64, batch_size: int = 4):
        """
        Production-grade sliding window emotion classification for long and very long text answers.

        Strategy:
        1. Short answers (<= 480 tokens): Single forward pass with full attention.
        2. Long and very long answers (> 480 tokens / 400 - 3000+ words):
           - Segment into overlapping sliding chunks (default 480 tokens, 64 token overlap).
           - Stride = max_chunk_size - overlap ensures seamless semantic continuity across chunk borders.
           - ZERO words dropped: 100% of chunks are evaluated in vectorized mini-batches.
           - Length-weighted probability aggregation across all chunks.
        """
        cleaned_text = text.strip() if text else ""
        if not cleaned_text:
            return {
                "success": False,
                "error": "Empty text provided"
            }

        # Tokenize entire text without truncation
        encoding = self.tokenizer(
            cleaned_text,
            add_special_tokens=False,
            return_attention_mask=False
        )
        input_ids = encoding["input_ids"]

        # Case 1: Short to medium answer (fits in single RoBERTa context window)
        if len(input_ids) <= max_chunk_size:
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(
                    outputs.logits,
                    dim=-1
                ).squeeze(0).cpu().numpy()

            top_idx = int(probabilities.argmax())
            top_emotion = self.id2label[top_idx]
            confidence = float(probabilities[top_idx])

            return {
                "success": True,
                "text": cleaned_text,
                "emotion": top_emotion,
                "confidence": confidence,
                "predictions": {
                    self.id2label[i]: float(probabilities[i])
                    for i in range(len(self.classes))
                },
                "tokens_analyzed": len(input_ids),
                "total_tokens": len(input_ids),
                "chunks_analyzed": 1,
                "total_chunks": 1,
                "sampling_method": "single_pass"
            }

        # Case 2: Segment into overlapping sliding chunks with zero missing words
        stride = max(max_chunk_size - overlap, 64)
        all_chunks = []
        chunk_weights = []
        start = 0
        while start < len(input_ids):
            end = min(start + max_chunk_size, len(input_ids))
            tokens = input_ids[start:end]
            all_chunks.append(tokens)
            chunk_weights.append(len(tokens))
            if end == len(input_ids):
                break
            start += stride

        total_chunks = len(all_chunks)
        selected_chunks = all_chunks
        selected_weights = chunk_weights

        # Case 3: Vectorized Batched Tensor Inference across all sliding chunks
        all_probs = []
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1

        for b_start in range(0, len(selected_chunks), batch_size):
            b_chunks = selected_chunks[b_start:b_start + batch_size]
            max_b_len = max(len(c) for c in b_chunks) + 2

            batch_input_ids = []
            batch_attn_masks = []
            for c in b_chunks:
                c_ids = [cls_id] + c + [sep_id]
                pad_len = max_b_len - len(c_ids)
                batch_input_ids.append(c_ids + [pad_id] * pad_len)
                batch_attn_masks.append([1] * len(c_ids) + [0] * pad_len)

            b_tensor = torch.tensor(batch_input_ids, dtype=torch.long, device=self.device)
            b_mask = torch.tensor(batch_attn_masks, dtype=torch.long, device=self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=b_tensor, attention_mask=b_mask)
                b_probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.extend(b_probs)

        # Case 4: Length-weighted aggregation across all evaluated chunks
        total_tokens = sum(selected_weights)
        aggregated_probs = np.zeros(len(self.classes), dtype=np.float32)

        for probs, weight in zip(all_probs, selected_weights):
            aggregated_probs += probs * (weight / total_tokens)

        # Normalize probabilities
        aggregated_probs = aggregated_probs / np.sum(aggregated_probs)

        top_idx = int(aggregated_probs.argmax())
        top_emotion = self.id2label[top_idx]
        confidence = float(aggregated_probs[top_idx])

        return {
            "success": True,
            "text": cleaned_text,
            "emotion": top_emotion,
            "confidence": confidence,
            "predictions": {
                self.id2label[i]: float(aggregated_probs[i])
                for i in range(len(self.classes))
            },
            "tokens_analyzed": sum(selected_weights),
            "total_tokens": len(input_ids),
            "total_chunks": total_chunks,
            "chunks_analyzed": len(selected_chunks),
            "sampling_method": "full_coverage_sliding_window"
        }


# Create one singleton model instance
text_predictor = TextEmotionPredictor()
