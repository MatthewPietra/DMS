"""
Image Processing Module

Provides real-time image processing, optimization, and quality enhancement
for captured images in the YOLO Vision Studio capture system.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, Optional, Dict, Any, Union
import logging

from ..utils.logger import get_component_logger
from ..utils.config import CaptureConfig


class ImageProcessor:
    """
    Real-time image processor for capture system.

    Handles image resizing, quality enhancement, format conversion,
    and optimization for YOLO training.
    """

    def __init__(self, config: CaptureConfig):
        self.config = config
        self.logger = get_component_logger("image_processor")

        # Processing parameters
        self.enhancement_enabled = True
        self.noise_reduction_enabled = True
        self.auto_contrast_enabled = True

    def process_image(
        self, image: Image.Image, target_resolution: Tuple[int, int]
    ) -> Image.Image:
        """
        Process captured image with optimization for YOLO training.

        Args:
            image: Input PIL Image
            target_resolution: Target (width, height)

        Returns:
            Processed PIL Image
        """
        try:
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize image
            processed_image = self._resize_image(image, target_resolution)

            # Apply enhancements if enabled
            if self.enhancement_enabled:
                processed_image = self._enhance_image(processed_image)

            # Noise reduction
            if self.noise_reduction_enabled:
                processed_image = self._reduce_noise(processed_image)

            # Auto contrast
            if self.auto_contrast_enabled:
                processed_image = self._auto_contrast(processed_image)

            return processed_image

        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            # Return original image on error
            return image

    def _resize_image(
        self, image: Image.Image, target_resolution: Tuple[int, int]
    ) -> Image.Image:
        """Resize image maintaining aspect ratio."""
        target_width, target_height = target_resolution
        original_width, original_height = image.size

        # Calculate scaling factor to fit within target resolution
        scale_w = target_width / original_width
        scale_h = target_height / original_height
        scale = min(scale_w, scale_h)

        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create new image with target resolution (pad if necessary)
        if new_width != target_width or new_height != target_height:
            # Create black background
            padded_image = Image.new("RGB", target_resolution, (0, 0, 0))

            # Calculate position to center the resized image
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2

            # Paste resized image onto padded background
            padded_image.paste(resized_image, (x_offset, y_offset))
            return padded_image

        return resized_image

    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply image enhancements."""
        try:
            # Enhance sharpness slightly
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)

            # Enhance color saturation slightly
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.05)

            return image

        except Exception as e:
            self.logger.error(f"Error enhancing image: {e}")
            return image

    def _reduce_noise(self, image: Image.Image) -> Image.Image:
        """Apply noise reduction."""
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Apply bilateral filter for noise reduction while preserving edges
            denoised = cv2.bilateralFilter(cv_image, 9, 75, 75)

            # Convert back to PIL
            denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
            return Image.fromarray(denoised_rgb)

        except Exception as e:
            self.logger.error(f"Error reducing noise: {e}")
            return image

    def _auto_contrast(self, image: Image.Image) -> Image.Image:
        """Apply automatic contrast adjustment."""
        try:
            # Convert to numpy array
            img_array = np.array(image)

            # Calculate histogram
            hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])

            # Find 1st and 99th percentiles
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()

            # Find cutoff points
            total_pixels = img_array.size
            lower_percentile = 0.01
            upper_percentile = 0.99

            lower_cutoff = np.searchsorted(cdf, total_pixels * lower_percentile)
            upper_cutoff = np.searchsorted(cdf, total_pixels * upper_percentile)

            if upper_cutoff > lower_cutoff:
                # Apply contrast stretching
                img_stretched = np.clip(
                    (img_array - lower_cutoff) * 255.0 / (upper_cutoff - lower_cutoff),
                    0,
                    255,
                ).astype(np.uint8)

                return Image.fromarray(img_stretched)

            return image

        except Exception as e:
            self.logger.error(f"Error applying auto contrast: {e}")
            return image

    def extract_roi(
        self, image: Image.Image, bbox: Tuple[int, int, int, int]
    ) -> Image.Image:
        """Extract region of interest from image."""
        try:
            x1, y1, x2, y2 = bbox

            # Ensure coordinates are within image bounds
            width, height = image.size
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(x1, min(x2, width))
            y2 = max(y1, min(y2, height))

            # Extract ROI
            roi = image.crop((x1, y1, x2, y2))
            return roi

        except Exception as e:
            self.logger.error(f"Error extracting ROI: {e}")
            return image

    def create_thumbnail(
        self, image: Image.Image, size: Tuple[int, int] = (320, 320)
    ) -> Image.Image:
        """Create thumbnail for preview."""
        try:
            # Create thumbnail maintaining aspect ratio
            thumbnail = image.copy()
            thumbnail.thumbnail(size, Image.Resampling.LANCZOS)

            # Create background with target size
            background = Image.new("RGB", size, (0, 0, 0))

            # Center thumbnail on background
            x_offset = (size[0] - thumbnail.width) // 2
            y_offset = (size[1] - thumbnail.height) // 2
            background.paste(thumbnail, (x_offset, y_offset))

            return background

        except Exception as e:
            self.logger.error(f"Error creating thumbnail: {e}")
            return image

    def apply_augmentation(
        self, image: Image.Image, augmentation_type: str = "random"
    ) -> Image.Image:
        """Apply data augmentation for training diversity."""
        try:
            if augmentation_type == "random":
                # Randomly choose augmentation
                import random

                augmentations = ["brightness", "contrast", "rotation", "flip"]
                augmentation_type = random.choice(augmentations)

            if augmentation_type == "brightness":
                enhancer = ImageEnhance.Brightness(image)
                factor = np.random.uniform(0.8, 1.2)
                image = enhancer.enhance(factor)

            elif augmentation_type == "contrast":
                enhancer = ImageEnhance.Contrast(image)
                factor = np.random.uniform(0.8, 1.2)
                image = enhancer.enhance(factor)

            elif augmentation_type == "rotation":
                angle = np.random.uniform(-5, 5)
                image = image.rotate(angle, fillcolor=(0, 0, 0))

            elif augmentation_type == "flip":
                if np.random.random() > 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)

            return image

        except Exception as e:
            self.logger.error(f"Error applying augmentation: {e}")
            return image

    def get_image_statistics(self, image: Image.Image) -> Dict[str, Any]:
        """Get image statistics for quality assessment."""
        try:
            # Convert to numpy array
            img_array = np.array(image)

            # Calculate statistics
            stats = {
                "width": image.width,
                "height": image.height,
                "channels": len(img_array.shape),
                "mean_brightness": float(np.mean(img_array)),
                "std_brightness": float(np.std(img_array)),
                "min_value": int(np.min(img_array)),
                "max_value": int(np.max(img_array)),
                "file_size_estimate": image.width * image.height * 3,  # RGB
            }

            # Calculate histogram
            if len(img_array.shape) == 3:  # Color image
                # Calculate per-channel statistics
                stats["mean_r"] = float(np.mean(img_array[:, :, 0]))
                stats["mean_g"] = float(np.mean(img_array[:, :, 1]))
                stats["mean_b"] = float(np.mean(img_array[:, :, 2]))

                # Color balance assessment
                channel_means = [stats["mean_r"], stats["mean_g"], stats["mean_b"]]
                stats["color_balance"] = float(np.std(channel_means))

            # Estimate blur (Laplacian variance)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            stats["blur_score"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # Estimate noise level
            stats["noise_estimate"] = float(
                np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)
            )

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating image statistics: {e}")
            return {"error": str(e)}

    def validate_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Validate image quality for training suitability."""
        stats = self.get_image_statistics(image)

        quality_assessment = {
            "suitable_for_training": True,
            "quality_score": 1.0,
            "issues": [],
            "recommendations": [],
        }

        try:
            # Check resolution
            min_res = min(self.config.min_resolution)
            if image.width < min_res or image.height < min_res:
                quality_assessment["suitable_for_training"] = False
                quality_assessment["issues"].append("Resolution too low")
                quality_assessment["recommendations"].append(
                    f"Minimum resolution: {min_res}x{min_res}"
                )

            # Check brightness
            if "mean_brightness" in stats:
                brightness = stats["mean_brightness"]
                if brightness < 50:
                    quality_assessment["issues"].append("Image too dark")
                    quality_assessment["recommendations"].append(
                        "Increase brightness or improve lighting"
                    )
                    quality_assessment["quality_score"] *= 0.8
                elif brightness > 200:
                    quality_assessment["issues"].append("Image too bright")
                    quality_assessment["recommendations"].append(
                        "Reduce brightness or exposure"
                    )
                    quality_assessment["quality_score"] *= 0.8

            # Check blur
            if "blur_score" in stats:
                blur_score = stats["blur_score"]
                if blur_score < 100:  # Threshold for blur detection
                    quality_assessment["issues"].append("Image appears blurry")
                    quality_assessment["recommendations"].append(
                        "Ensure proper focus and reduce motion blur"
                    )
                    quality_assessment["quality_score"] *= 0.6

            # Check noise
            if "noise_estimate" in stats:
                noise_level = stats["noise_estimate"]
                if noise_level > 20:  # Threshold for excessive noise
                    quality_assessment["issues"].append("High noise level detected")
                    quality_assessment["recommendations"].append(
                        "Improve lighting conditions or reduce ISO"
                    )
                    quality_assessment["quality_score"] *= 0.7

            # Overall quality assessment
            if quality_assessment["quality_score"] < 0.6:
                quality_assessment["suitable_for_training"] = False

            return quality_assessment

        except Exception as e:
            self.logger.error(f"Error validating image quality: {e}")
            quality_assessment["suitable_for_training"] = False
            quality_assessment["issues"].append(f"Validation error: {e}")
            return quality_assessment

    def set_processing_options(self, **options):
        """Set image processing options."""
        if "enhancement_enabled" in options:
            self.enhancement_enabled = options["enhancement_enabled"]

        if "noise_reduction_enabled" in options:
            self.noise_reduction_enabled = options["noise_reduction_enabled"]

        if "auto_contrast_enabled" in options:
            self.auto_contrast_enabled = options["auto_contrast_enabled"]

        self.logger.info(f"Updated processing options: {options}")

    def get_processing_options(self) -> Dict[str, Any]:
        """Get current processing options."""
        return {
            "enhancement_enabled": self.enhancement_enabled,
            "noise_reduction_enabled": self.noise_reduction_enabled,
            "auto_contrast_enabled": self.auto_contrast_enabled,
        }
