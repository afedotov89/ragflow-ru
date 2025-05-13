#
# Based on work from InfiniFlow/deepdoc and JaidedAI/EasyOCR projects
# This file was created by Alexander Fedotov as an extension to the original project
# (Assisted by AI)
#

# EasyOCR Wrapper Module for DeepDoc

## Overview

This module (`easyocr_wrapper`) serves as a wrapper around the EasyOCR library, primarily to integrate ONNX-based inference for text detection and recognition within the DeepDoc project. The main goals are to maintain compatibility with the existing OCR interface in DeepDoc, leverage the potential performance benefits of ONNX, and ensure robust multi-language support, including Russian.

The development, particularly of the ONNX-based text detector, involved a significant amount of experimentation to replicate the post-processing logic of EasyOCR's CRAFT model and to adapt to the specific output characteristics of the converted ONNX models.

## Key Features

*   **EasyOCR Compatibility:** Provides a familiar `EasyOCR` class interface.
*   **ONNX Inference:** Implements `ONNXDetector` and `ONNXRecognizer` classes for running EasyOCR's detection (CRAFT) and recognition models in ONNX format.
*   **Dynamic Model Loading:** ONNX models can be loaded from a local cache or automatically downloaded from a specified Hugging Face repository.
*   **Flexible Runtime Providers:** Supports CUDA, CoreML (though temporarily disabled during development for stability on some platforms), and CPU execution providers for ONNX Runtime.
*   **Model Conversion Utilities:** Includes scripts and functions to convert EasyOCR's PyTorch models to the ONNX format.

## Module Components

*   **`ocr.py`**:
    *   `EasyOCR`: The main wrapper class. It initializes EasyOCR's standard PyTorch reader and then attempts to load and use ONNX models if available and `onnxruntime` is installed. It orchestrates detection and recognition, falling back to PyTorch if ONNX models fail to load or are disabled.
    *   `ONNXDetector`: Handles text detection using an ONNX version of the CRAFT model. The core of recent development efforts has been to refine its post-processing logic.
    *   `ONNXRecognizer`: Handles text recognition using ONNX versions of EasyOCR's recognition models (e.g., CRNN, VGG).

*   **`converter.py`**:
    *   Provides functionalities to convert EasyOCR's PyTorch models (`.pth`) to ONNX (`.onnx`).
    *   Includes helper functions for downloading/uploading models to/from Hugging Face Hub.
    *   Addresses various challenges encountered during ONNX export, such as handling dynamic input/output shapes and specific PyTorch operations that require workarounds for ONNX compatibility (e.g., `AdaptiveAvgPool2d`).

*   **`cli.py`**:
    *   A command-line interface, primarily for triggering the model conversion process defined in `converter.py`.

## ONNXDetector: In-Depth

The `ONNXDetector` aims to replicate the text detection capabilities of EasyOCR's CRAFT model using an ONNX-exported version.

### 1. Input Preprocessing

*   **Resizing:** The input image is resized to the dimensions expected by the ONNX CRAFT model (e.g., 640x640, or dynamically determined from the model's input shape).
*   **Normalization:**
    *   Pixel values are first scaled to the `[0, 1]` range.
    *   **ImageNet Normalization:** The image is then normalized using ImageNet mean and standard deviation values (`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`). This step was found to be crucial, as the ONNX model (likely trained on ImageNet-normalized data) produced significantly different output distributions without it, making threshold-based post-processing difficult.
*   **Layout Transposition:** The image layout is changed from HWC (Height, Width, Channels) to CHW (Channels, Height, Width) and a batch dimension is added.

### 2. Model Inference & Output

*   The preprocessed image is fed into the ONNX CRAFT model.
*   The model is expected to output two main feature maps (often combined into a single tensor):
    *   `region_score_map`: Represents the probability of each pixel being part of a text region.
    *   `link_score_map`: Represents the probability of affinity (linkage) between pixels, helping to connect characters into words.

### 3. Postprocessing

This has been the most challenging part, involving iterative refinement to match EasyOCR's conceptual postprocessing flow while adapting to the ONNX model's specific output characteristics and the current implementation choices.

**Conceptual Similarities with EasyOCR CRAFT Postprocessing:**

*   **Core Maps:** Utilizes both a `region_score_map` (text presence) and a `link_score_map` (text connectivity).
*   **Thresholding Concepts:** Employs thresholds conceptually similar to EasyOCR's `low_text`, `link_threshold`, and `text_threshold` for binarization and confidence filtering.
*   **Sigmoid Activation:** Applies a sigmoid function to score maps to get probability-like values.
*   **Segmentation Map (`segmap`):** Creates a combined binary map from region and link scores to identify potential text areas.
*   **Dilation:** Applies `cv2.dilate` to the `segmap` to thicken text regions and connect components.
*   **Filtering:** Filters detected components by minimum size and a text confidence score derived from the original `region_score_map`.

**Key Differences from Standard EasyOCR CRAFT Postprocessing:**

*   **1. Bounding Box Generation (Major Difference):**
    *   **Current `ONNXDetector`:** Uses `skimage.measure.label` on the (dilated) `segmap` to find connected components. For each component, an axis-aligned bounding box is generated based on the min/max row and column coordinates of the component's pixels.
    *   **Standard EasyOCR:** Typically uses `cv2.findContours` on the processed `segmap`. For each contour, `cv2.minAreaRect` is called to find the smallest possible enclosing rectangle (which can be rotated). The four corner points of this rotated rectangle are then calculated using `cv2.boxPoints`.
    *   **Impact:** The current `ONNXDetector` produces only axis-aligned boxes, which are less accurate for rotated or non-horizontal text compared to EasyOCR's rotated boxes. This is marked as a "Future Improvement."

*   **2. Threshold Value Adaptation (Parameterization Difference):**
    *   **Current `ONNXDetector`:** Due to the specific output characteristics of the ONNX CRAFT model used (yielding generally higher and more compressed scores), the *values* for `low_text`, `link_threshold`, and `text_threshold` had to be experimentally adjusted to be significantly higher than EasyOCR's typical defaults. (See "Thresholds & Experimental Adaptation" below).
    *   **Standard EasyOCR:** Uses default thresholds (e.g., `low_text=0.4`, `link_threshold=0.4`, `text_threshold=0.7`) that are tuned for the typical output of its PyTorch CRAFT model.
    *   **Impact:** While the *role* of the thresholds is similar, their *values* are different, reflecting an adaptation to the model's behavior rather than a fundamental logic change.

*   **3. Details of `segmap` creation and use (Minor implementation detail, but aligned in principle):**
    *   **Current `ONNXDetector`:**
        1.  `region_score_map` is binarized by `low_text`.
        2.  `link_score_map` is binarized by `link_threshold`.
        3.  These binary maps are added and clipped to create `segmap`.
        4.  `segmap` is dilated.
        5.  Components are found on the dilated `segmap`.
        6.  Components are filtered by `min_size`.
        7.  Components are filtered if `max(original_region_score_map_pixels_in_component) < text_threshold`.
    *   **Standard EasyOCR (`craft_utils.getDetBoxes` and `utils.getDetBoxes_core`):** The logic is very similar. It also binarizes `textmap` by `low_text` and `linkmap` by `link_threshold`, combines them (e.g., `text_score_comb = np.clip(text_score + link_score, 0, 1)`), dilates this combined map, finds contours, and then filters these contours/boxes using `min_size` and a check against `text_threshold` on the original (non-binarized) `textmap`.
    *   **Impact:** The core sequence of binarization, combination, dilation, component finding, and filtering is conceptually aligned. The primary divergence remains the axis-aligned vs. rotated box generation.

### 4. Thresholds & Experimental Adaptation

A critical aspect of development was tuning the thresholds: `low_text`, `link_threshold`, and `text_threshold`. This was necessary because the ONNX CRAFT model used (even after ImageNet normalization) produced `region_score_map` and `link_score_map` values that were generally higher and more "compressed" (less sparse) than what standard EasyOCR defaults are designed for.

*   **Initial Challenge:** With EasyOCR's default thresholds (e.g., `low_text=0.4`), the high scores from the ONNX model often resulted in the entire image being detected as a single text box because the binarized maps were almost entirely `True`.
*   **Experimental Adjustments:** To counteract this, the thresholds were experimentally increased. The current "best" working set (as of the time of writing this README), when used with dilation, is:
    *   `low_text = 0.6`
    *   `link_threshold = 0.55`
    *   `text_threshold = 0.65`
    These values differ from EasyOCR's defaults and were identified through an iterative process of observing the model's output maps and the resulting bounding boxes. This adaptation highlights a practical difference when working with specific ONNX model conversions.

## ONNXRecognizer

The `ONNXRecognizer` takes cropped image regions (presumably containing single lines of text) from the detector and feeds them into an ONNX recognition model.

*   **Preprocessing:** Involves resizing the image crop to a fixed height while maintaining aspect ratio, converting to grayscale, and normalizing pixel values.
*   **Inference:** The processed image is passed to the ONNX recognition model.
*   **Postprocessing:** The model output (typically a sequence of character probabilities over time steps) is decoded using a simplified CTC (Connectionist Temporal Classification) decoding approach:
    *   Find the character index with the maximum probability at each time step.
    *   Remove repeated indices.
    *   Remove blank characters.
    *   Map remaining indices to characters using a vocabulary.

## Model Conversion (`converter.py`)

The `converter.py` script was developed to:

1.  Load EasyOCR's PyTorch models.
2.  Export them to the ONNX format using `torch.onnx.export`.
3.  Handle dummy inputs required for the export process, including specific handling for the recognizer model which often requires two inputs (image and a text sequence tensor for teacher forcing during training, though only the image part is used for inference logic during export).
4.  Address specific operator compatibility issues encountered during export (e.g., a ` cuello.AdaptiveAvgPool2d` in a VGG-based recognizer model required modifications to its parameters and a subsequent `squeeze` operation in the model's `forward` method to become ONNX-compatible).
5.  Manage potential device (CPU/GPU/MPS) issues during conversion, sometimes forcing CPU for stability.
6.  Integrate with `huggingface_hub` to upload the converted ONNX models and character list (`vocab.txt`) to a repository, and also to download them in the main `EasyOCR` wrapper if not found locally.

## Usage

The `EasyOCR` class in `ocr.py` is designed to be a drop-in replacement (or close alternative) for the standard EasyOCR reader. It's instantiated similarly, and its `__call__` method or `readtext` (if fully implemented) can be used to perform OCR. The `deepdoc.vision.t_ocr.py` script provides an example of its usage.

## Future Improvements

*   **`cv2.minAreaRect`:** Transition from `skimage.measure.label` and axis-aligned bounding boxes to `cv2.findContours` followed by `cv2.minAreaRect` and `cv2.boxPoints`. This will provide more accurate, rotated bounding boxes, which is standard in EasyOCR and better for non-horizontal text.
*   **Threshold Robustness:** Investigate if the ONNX model's output can be further normalized or if a more adaptive thresholding technique can be employed to reduce sensitivity to specific threshold values.
*   **Comprehensive `readtext` Equivalence:** Ensure all parameters and return structures of the original `EasyOCR.Reader.readtext()` method are fully replicated for seamless compatibility if needed.
*   **Investigate ONNX Model Output:** Further explore why the ONNX CRAFT model produces consistently high-score maps compared to expectations, to see if it's an artifact of conversion or a characteristic of the specific pre-trained model version used.

## ⚠️ Critical Note on Character List and vocab.txt

- The `vocab.txt` used for ONNX model decoding is generated from `reader.character`. This ensures that the character order matches the exact order used by the model for decoding output indices.
- A blank symbol (an empty string) is added as the first line in `vocab.txt`, corresponding to the CTC blank index (index 0).
  Example of `vocab.txt` generation:
  ```python
  vocab = list(reader.character)
  vocab_with_blank = [''] + vocab # Blank symbol first
  with open('vocab.txt', 'w', encoding='utf-8') as f:
      f.write('\n'.join(vocab_with_blank))
  ```
- Using a character list that is inconsistent with the one used during model training/export, or an incorrect blank symbol handling, will result in garbage or unpredictable decoding outputs, even if the model weights are correct.
- The CTC decoding process relies on the blank symbol being at index 0 (the first line of `vocab.txt`).

## Troubleshooting ONNX Decoding Issues

- If you see systematic symbol shifts (e.g., all letters off by one), check that you have a blank symbol as the first line in vocab.txt.
- If you see random or nonsensical output, check that your vocab.txt was generated from `reader.character` **in the same session as ONNX export**.
- If you use multiple languages, always set `character_list` explicitly when creating the EasyOCR Reader to avoid randomization due to Python set behavior.
- For best reproducibility, always save and reuse the exact vocab.txt used during ONNX export.

## ONNX model and vocab.txt

- The ONNX model uses a vocab.txt file generated from `reader.character`, preserving the exact order of characters as used by the model.
- The first line in vocab.txt is a blank symbol (an empty string), corresponding to index 0 for CTC decoding.
- This approach guarantees correct mapping between model output indices and characters during decoding.
- Using any other character list or changing the order leads to incorrect recognition (such as shifted characters or random output).
- For multilingual models, the `character_list` is explicitly set when creating the EasyOCR Reader to ensure reproducibility.
- Systematic character shifts (e.g., all letters shifted by one) indicate a missing blank symbol as the first line in vocab.txt.
- Random or nonsensical output may indicate that vocab.txt was not generated from `reader.character` or does not match the model used for ONNX export.
- For reproducibility, the exact vocab.txt used during ONNX export is saved and reused.