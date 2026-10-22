# ============================================================
# FACIAL INFERENCE
# ViT-Base-Patch16-224 facial emotion model
#
# Input  : cropped face image (numpy RGB)
# Output : dict {emotion: confidence}
# ============================================================

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import numpy as np
import cv2
import timm

from pathlib import Path
from transformers import AutoModelForImageClassification


# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

MODEL_NAME = "trpakov/vit-face-expression"

TARGET_CLASSES = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
]


class EfficientNetEmotionModel(nn.Module):
    """Matches the trained EfficientNet-B0 checkpoint (facial_emotion_model.pt)"""
    def __init__(self, num_classes=7, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0', pretrained=False, num_classes=0
        )
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


# ============================================================
# ViT MODEL
# ============================================================

class FacialEmotionModel(nn.Module):

    def __init__(self, num_classes=7):

        super().__init__()

        self.vit = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values):

        outputs = self.vit(
            pixel_values=pixel_values
        )

        return outputs.logits


# ============================================================
# FACIAL INFERENCE
# ============================================================

class FacialInference:

    # ImageNet normalization
    MEAN = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    STD = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    IMAGE_SIZE = 224

    def __init__(
        self,
        model_path: str,
        classes_path: str
    ):

        print(
            "\n[FacialInference] "
            "── Initializing ViT ─────────────────"
        )

        # ====================================================
        # Validate model path
        # ====================================================

        if not Path(model_path).exists():

            raise FileNotFoundError(
                f"[FacialInference] ❌ "
                f"Model file not found: {model_path}\n"
                f"Make sure the new ViT model is inside models/"
            )

        model_size_mb = (
            Path(model_path).stat().st_size / 1e6
        )

        print(
            f"  ✅ model file found: "
            f"{model_path} "
            f"({model_size_mb:.2f} MB)"
        )


        # ====================================================
        # Validate classes file
        # ====================================================

        if not Path(classes_path).exists():

            raise FileNotFoundError(
                f"[FacialInference] ❌ "
                f"Classes file not found: {classes_path}"
            )

        print(
            f"  ✅ classes file found: "
            f"{classes_path}"
        )


        # ====================================================
        # Device
        # ====================================================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"  Device: {self.device}"
        )


        # ====================================================
        # Load classes
        # ====================================================

        loaded_classes = np.load(
            classes_path,
            allow_pickle=True
        ).tolist()

        print(
            f"  Classes from file: "
            f"{loaded_classes}"
        )


        self.classes = list(loaded_classes)

        # ====================================================
        # Load model weights (auto-detects EfficientNet or ViT)
        # ====================================================

        try:
            state_dict = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False
            )

            if any(k.startswith("backbone.") for k in state_dict.keys()):
                print("  [FacialInference] Loading trained EfficientNet-B0 checkpoint")
                self.model = EfficientNetEmotionModel(num_classes=len(self.classes))
            else:
                print(f"  [FacialInference] Loading ViT architecture: {MODEL_NAME}")
                self.model = FacialEmotionModel(num_classes=len(self.classes))

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            print("  ✅ Facial model weights loaded successfully")

        except Exception as e:
            raise RuntimeError(
                f"[FacialInference] ❌ Failed to load model: {e}"
            )


        # ====================================================
        # Warmup
        # ====================================================

        try:

            dummy = torch.zeros(
                1,
                3,
                self.IMAGE_SIZE,
                self.IMAGE_SIZE,
                device=self.device
            )

            with torch.no_grad():

                output = self.model(
                    dummy
                )

            expected_shape = (
                1,
                len(self.classes)
            )

            if tuple(output.shape) != expected_shape:

                raise RuntimeError(
                    f"Expected output "
                    f"{expected_shape}, "
                    f"got {tuple(output.shape)}"
                )

            print(
                f"  ✅ Warmup forward pass OK "
                f"— output shape: {output.shape}"
            )

        except Exception as e:

            raise RuntimeError(
                "[FacialInference] ❌ "
                f"Warmup failed: {e}"
            )


        print(
            "[FacialInference] "
            "── ViT Ready ─────────────────────────\n"
        )


    # ========================================================
    # PREPROCESSING
    # ========================================================

    def preprocess(
        self,
        face_rgb: np.ndarray
    ) -> torch.Tensor:

        """
        face_rgb:
            H × W × 3 numpy RGB uint8 image

        Returns:
            (1, 3, 224, 224) tensor
        """

        if face_rgb is None:
            raise ValueError(
                "face_rgb is None"
            )

        if face_rgb.size == 0:
            raise ValueError(
                "face_rgb is empty"
            )


        # Resize exactly as friend's notebook
        img = cv2.resize(
            face_rgb,
            (
                self.IMAGE_SIZE,
                self.IMAGE_SIZE
            )
        )


        # Convert uint8 → float32 [0,1]
        img = img.astype(
            np.float32
        ) / 255.0


        # ImageNet normalization
        img = (
            img - self.MEAN
        ) / self.STD


        # HWC → CHW
        img = img.transpose(
            2,
            0,
            1
        )


        tensor = torch.tensor(
            img,
            dtype=torch.float32
        ).unsqueeze(0)


        return tensor.to(
            self.device
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    @torch.no_grad()
    def predict(
        self,
        face_rgb: np.ndarray
    ) -> dict:

        """
        Returns:

        {
            "anger": ...,
            "disgust": ...,
            "fear": ...,
            "happiness": ...,
            "sadness": ...,
            "surprise": ...,
            "neutral": ...
        }

        Returns all-zero dict if face is invalid.
        """

        if (
            face_rgb is None
            or face_rgb.size == 0
        ):

            return {
                c: 0.0
                for c in self.classes
            }


        try:

            # =================================================
            # Original image
            # =================================================

            tensor = self.preprocess(
                face_rgb
            )


            # =================================================
            # Original prediction
            # =================================================

            logits = self.model(
                tensor
            )

            probs_original = torch.softmax(
                logits,
                dim=1
            )


            # =================================================
            # Horizontal flip TTA
            # Exactly as friend's notebook
            # =================================================

            flipped_tensor = torch.flip(
                tensor,
                dims=[3]
            )

            logits_flip = self.model(
                flipped_tensor
            )

            probs_flip = torch.softmax(
                logits_flip,
                dim=1
            )


            # =================================================
            # Average original + flipped predictions
            # =================================================

            probs = (
                probs_original
                +
                probs_flip
            ) / 2.0


            probs = (
                probs
                .squeeze(0)
                .cpu()
                .numpy()
            )


            # =================================================
            # Validate probabilities
            # =================================================

            if not np.isfinite(
                probs
            ).all():

                print(
                    "[FacialInference] "
                    "⚠️ Non-finite probabilities"
                )

                return {
                    c: 0.0
                    for c in self.classes
                }


            # =================================================
            # Return emotion probabilities
            # =================================================

            return dict(
                zip(
                    self.classes,
                    probs.tolist()
                )
            )


        except Exception as e:

            print(
                "[FacialInference] "
                f"⚠️ Prediction error: {e}"
            )

            return {
                c: 0.0
                for c in self.classes
            }