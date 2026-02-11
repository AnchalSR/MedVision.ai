"""
MedVision.ai - Medical Image Preprocessor
Handles loading, normalization, and modality detection for medical images.
"""
import io
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)


class MedicalImagePreprocessor:
    """
    Preprocessing pipeline for medical images.
    Supports X-ray, MRI, CT, and standard image formats.
    """

    def __init__(self, target_size: int = None):
        from config.settings import IMAGE_SIZE
        self.target_size = target_size or IMAGE_SIZE

    def load_image(self, source) -> Image.Image:
        """
        Load an image from various sources.

        Args:
            source: File path (str/Path), bytes, or PIL Image.

        Returns:
            PIL Image in RGB mode.
        """
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        elif isinstance(source, bytes):
            return Image.open(io.BytesIO(source)).convert("RGB")
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() == ".dcm":
                return self._load_dicom(path)
            return Image.open(path).convert("RGB")
        else:
            raise ValueError(f"Unsupported image source type: {type(source)}")

    def preprocess(self, image: Image.Image, modality: str = "auto") -> Image.Image:
        """
        Apply preprocessing pipeline appropriate for the detected modality.

        Args:
            image: PIL Image.
            modality: 'xray', 'mri', 'ct', or 'auto' for auto-detection.

        Returns:
            Preprocessed PIL Image.
        """
        if modality == "auto":
            modality = self.detect_modality(image)

        # Resize to target size
        image = self._resize(image)

        # Apply modality-specific preprocessing
        if modality == "xray":
            image = self._preprocess_xray(image)
        elif modality == "mri":
            image = self._preprocess_mri(image)
        elif modality == "ct":
            image = self._preprocess_ct(image)

        return image

    def detect_modality(self, image: Image.Image) -> str:
        """
        Detect the imaging modality based on image characteristics.
        Uses heuristics on pixel distribution and aspect ratio.
        """
        arr = np.array(image.convert("L"))  # Convert to grayscale for analysis
        mean_val = arr.mean()
        std_val = arr.std()
        aspect_ratio = image.width / max(image.height, 1)

        # X-rays: typically high contrast, predominantly dark background
        if std_val > 50 and mean_val < 120:
            return "xray"
        # MRI: typically moderate contrast, various intensity patterns
        elif std_val > 40 and 100 < mean_val < 180:
            return "mri"
        # CT: typically has specific windowing patterns
        elif std_val < 40 and mean_val < 100:
            return "ct"
        else:
            return "general"

    def _resize(self, image: Image.Image) -> Image.Image:
        """Resize while maintaining aspect ratio, then center crop."""
        w, h = image.size
        scale = self.target_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Center crop to target_size x target_size
        left = (new_w - self.target_size) // 2
        top = (new_h - self.target_size) // 2
        return image.crop((left, top, left + self.target_size, top + self.target_size))

    def _preprocess_xray(self, image: Image.Image) -> Image.Image:
        """X-ray specific preprocessing: contrast enhancement, noise reduction."""
        # Enhance contrast for better feature visibility
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)

        # Slight sharpening
        image = image.filter(ImageFilter.SHARPEN)
        return image

    def _preprocess_mri(self, image: Image.Image) -> Image.Image:
        """MRI specific preprocessing: intensity normalization, denoising."""
        # Brightness normalization
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)

        # Slight blur to reduce MRI artifacts
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        return image

    def _preprocess_ct(self, image: Image.Image) -> Image.Image:
        """CT specific preprocessing: windowing simulation."""
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        return image

    def _load_dicom(self, path: Path) -> Image.Image:
        """Load a DICOM file and convert to PIL Image."""
        try:
            import pydicom

            ds = pydicom.dcmread(str(path))
            arr = ds.pixel_array.astype(np.float32)

            # Normalize to 0-255
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            arr = arr.astype(np.uint8)

            return Image.fromarray(arr).convert("RGB")
        except ImportError:
            logger.warning("pydicom not installed. Cannot load DICOM files.")
            raise ValueError(
                "DICOM support requires pydicom. Install with: pip install pydicom"
            )

    def get_image_info(self, image: Image.Image) -> dict:
        """Extract image metadata for analysis context."""
        arr = np.array(image.convert("L"))
        modality = self.detect_modality(image)

        return {
            "width": image.width,
            "height": image.height,
            "modality": modality,
            "mean_intensity": float(arr.mean()),
            "std_intensity": float(arr.std()),
            "mode": image.mode,
        }

    def create_context_description(self, image: Image.Image) -> str:
        """Generate a textual description of image properties for LLM context."""
        info = self.get_image_info(image)
        modality_names = {
            "xray": "X-ray radiograph",
            "mri": "MRI scan",
            "ct": "CT scan",
            "general": "medical image",
        }
        modality_name = modality_names.get(info["modality"], "medical image")

        return (
            f"This appears to be a {modality_name} "
            f"(resolution: {info['width']}x{info['height']}, "
            f"mean intensity: {info['mean_intensity']:.1f}, "
            f"contrast level: {info['std_intensity']:.1f}). "
            f"The image has been preprocessed and encoded using CLIP visual features "
            f"for similarity matching against the medical knowledge base."
        )
