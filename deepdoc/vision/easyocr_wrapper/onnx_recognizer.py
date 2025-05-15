#
# Based on work from InfiniFlow project
# This file was created by Alexander Fedotov as an extension to the original project
# Licensed under the Apache License, Version 2.0
#

import cv2
import numpy as np
import logging
import onnxruntime as ort # Keep top-level for clarity

class ONNXRecognizer:
    """
    ONNX Implementation of EasyOCR recognizer
    """
    def __init__(self, model_path, vocab_path=None, use_gpu=True):
        # Configuration for ONNX Runtime session
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Determine compute provider
        providers = []
        if use_gpu:
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                cuda_provider_options = {
                    "device_id": 0,
                    "gpu_mem_limit": 512 * 1024 * 1024, # 512MB, adjust as needed
                    "arena_extend_strategy": "kNextPowerOfTwo",
                }
                providers.append(('CUDAExecutionProvider', cuda_provider_options))
                logging.info("Using CUDAExecutionProvider for ONNX recognizer.")
            # Temporarily disable CoreML to default to CPU if CUDA is not available
            # elif 'CoreMLExecutionProvider' in ort.get_available_providers():
            # providers.append('CoreMLExecutionProvider')
            # logging.info("Using CoreMLExecutionProvider for ONNX recognizer.")
            else:
                logging.warning("CUDAExecutionProvider not available. Falling back to CPUExecutionProvider for ONNX recognizer.")
                providers.append('CPUExecutionProvider')
        else:
            providers.append('CPUExecutionProvider')
            logging.info("Using CPUExecutionProvider for ONNX recognizer (GPU explicitly disabled).")

        # Load the model
        self.session = ort.InferenceSession(model_path, options=options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()] # Initialize output_names

        # Define input and output shapes
        logging.info(f"Recognizer input shape: {self.input_shape}")

        # Load vocabulary if specified
        self.vocab = None
        if vocab_path:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = f.read().splitlines()
        else:
            # Use standard vocabulary for English characters
            self.vocab = list("0123456789abcdefghijklmnopqrstuvwxyz")

    def recognize(self, image):
        """
        Recognize text in an image

        Args:
            image: Input image as numpy array (cropped text region, RGB)

        Returns:
            Recognized text string
        """
        # Prepare image for recognizer
        # Usually requires fixed height and variable width image
        h, w = image.shape[:2]
        target_height = 48  # Standard height for recognizer

        # Convert to grayscale if the image has 3 channels (e.g., RGB or BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Assuming input is RGB
        elif len(image.shape) == 2: # Already grayscale
            image_gray = image
        else:
            raise ValueError(f"Unsupported image shape for ONNXRecognizer: {image.shape}")

        # Scale image preserving aspect ratio
        ratio = target_height / image_gray.shape[0] # Use h_gray
        target_width = int(image_gray.shape[1] * ratio) # Use w_gray

        # Check minimum size
        target_width = max(target_width, 16)  # Minimum width

        # Ensure target_width does not exceed a reasonable maximum if self.rec_image_shape[2] (max_width) is set and non-zero
        # max_width_from_shape = self.rec_image_shape[2]
        # if max_width_from_shape and target_width > max_width_from_shape:
        #    target_width = max_width_from_shape
        #    logging.warning(f"ONNXRecognizer: Cropped image width {int(target_height * aspect_ratio)} for height {target_height} "
        #                    f"exceeded max_width {max_width_from_shape}, capping to {target_width}.")

        img_resized = cv2.resize(image_gray, (target_width, target_height))

        # Normalize image to [-1, 1] range, which is common for EasyOCR recognizers
        img_normalized = (img_resized.astype(np.float32) / 255.0 - 0.5) * 2.0

        # Add channel dimension: (H, W) -> (1, H, W)
        img_chw = img_normalized[np.newaxis, :, :]

        # Add batch dimension for ONNX model: (1, H, W) -> (1, 1, H, W)
        input_tensor = img_chw[np.newaxis, :, :, :] # NCHW

        logging.debug(f"ONNXRecognizer: input_tensor shape before inference: {input_tensor.shape}")

        # ONNX inference
        try:
            input_feed = {self.input_name: input_tensor}
            # outputs_onnx will be a list of numpy arrays if multiple output_names, or single array if one
            outputs_onnx = self.session.run(self.output_names, input_feed)
            # Assuming the recognizer has one primary output for character probabilities
            preds_onnx = outputs_onnx[0]
            if preds_onnx.ndim == 3 and preds_onnx.shape[0] == 1:
                preds_onnx = preds_onnx[0] # Remove batch dimension if it's 1, so it's (time_steps, num_characters)
        except Exception as e:
            logging.error(f"ONNXRecognizer: Error during ONNX inference: {e}")
            return "" # Return empty string on error

        # Decode predictions
        text = self._decode_prediction(preds_onnx)

        confidence = 1.0 if text else 0.0
        return text, confidence

    def _decode_prediction(self, prediction):
        """
        Decode model prediction to text

        Args:
            prediction: Model output data, expected shape: (time_steps, num_characters_including_blank)

        Returns:
            Recognized text
        """
        # Find indices with maximum probability for each time step
        indices = np.argmax(prediction, axis=1) # shape: (time_steps,)

        # Remove repeated indices (greedy CTC decoding part 1)
        prev_index = -1 # Placeholder for an impossible index (assuming valid indices are >= 0)
        unique_indices = []
        for idx in indices:
            if idx != prev_index:
                unique_indices.append(idx)
                prev_index = idx

        # Define blank_index.
        # For some models, blank is 0. For others, it's len(vocab).
        # Let's revert to 0 as a test, as len(vocab) didn't resolve the issue.
        blank_index = 0

        # Remove blank characters (greedy CTC decoding part 2)
        result_indices = [idx for idx in unique_indices if idx != blank_index]

        # Convert valid indices to characters
        # self.vocab contains characters from index 0 to len(self.vocab)-1
        # We must ensure idx is within the bounds of self.vocab
        text = ''.join([self.vocab[idx] for idx in result_indices if idx < len(self.vocab)])

        return text