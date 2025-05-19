#
# Based on work from InfiniFlow project
# This file was created by Alexander Fedotov as an extension to the original project
# Licensed under the Apache License, Version 2.0
#

from api.utils.file_utils import get_project_base_directory
from rag.settings import EASYOCR_USE_ONNX, EASYOCR_MAX_CACHE_SIZE, EASYOCR_ONNX_REPO_ID
from huggingface_hub import snapshot_download

import cv2
import easyocr
import numpy as np
import os
import torch
from skimage.measure import label

import hashlib
import logging
import time
from collections import OrderedDict
from threading import RLock

from .onnx_detector import ONNXDetector
from .onnx_recognizer import ONNXRecognizer
from .pytorch_detector import PyTorchDetector
from .pytorch_recognizer import PyTorchRecognizer


def load_onnx_model(model_dir, use_gpu=True, detector_name=None, recognizer_name=None, vocab_path=None):
    """
    Load ONNX models for OCR

    Args:
        model_dir: Directory containing ONNX models
        use_gpu: Whether to use GPU for inference
        detector_name: Name of the detector model file (defaults to EasyOCR.DETECTOR_MODEL + ".onnx")
        recognizer_name: Name of the recognizer model file (auto-detected if None)
        vocab_path: Path to the vocabulary file for the recognizer.

    Returns:
        Tuple of (detector, recognizer)
    """
    # Use default detector name if not provided
    if detector_name is None:
        detector_name = f"{EasyOCR.DETECTOR_MODEL}.onnx"

    det_path = os.path.join(model_dir, detector_name)

    if recognizer_name is None:
        rec_files = [f for f in os.listdir(model_dir)
                    if f.endswith('.onnx') and f != detector_name]
        if rec_files:
            recognizer_name = rec_files[0]
            logging.info(f"Auto-detected recognizer model: {recognizer_name}")
        else:
            raise FileNotFoundError(f"No recognizer model found in {model_dir}, and auto-detection failed for recognizer_name=None.")

    rec_path = os.path.join(model_dir, recognizer_name)

    # vocab_path is now a required argument for ONNXRecognizer if not using default built-in.
    # load_onnx_model should ensure it's available or error.
    # The EasyOCR class will be responsible for ensuring vocab_path is correct.
    # For this function, we assume vocab_path is provided if a custom one is needed by ONNXRecognizer.
    # If vocab_path is None here, ONNXRecognizer will use its default.
    # However, a standard "vocab.txt" is usually expected alongside ONNX models.

    if not os.path.exists(det_path):
        raise FileNotFoundError(f"Detector model not found: {det_path}")
    if not os.path.exists(rec_path):
        raise FileNotFoundError(f"Recognizer model not found: {rec_path}")
    if vocab_path and not os.path.exists(vocab_path): # Check vocab_path only if provided
        logging.warning(f"Specified vocabulary file not found: {vocab_path}. ONNXRecognizer might use a default or fail if it requires this specific one.")


    detector = ONNXDetector(det_path, use_gpu=use_gpu)
    # Pass the vocab_path to ONNXRecognizer; it handles None internally (uses default).
    recognizer = ONNXRecognizer(rec_path, vocab_path=vocab_path, use_gpu=use_gpu)

    return detector, recognizer


class EasyOCR:

    LANGUAGES = ('en', 'ru')

    # Standard detector model name - typically everyone uses the same CRAFT detector
    DETECTOR_MODEL = "craft_mlt_25k"

    # Simplified version with a single model for en and ru
    RECOGNIZER_MODEL = "cyrillic_g2"  # One model for Russian and English

    @classmethod
    def get_detector_onnx_name(cls):
        """Get the standard ONNX detector model name"""
        return f"{cls.DETECTOR_MODEL}.onnx"

    @classmethod
    def get_recognizer_onnx_name(cls):
        """
        Get standardized recognizer ONNX model name
        We use cyrillic_g2 for recognizing both Russian and English text

        Returns:
            Name of the recognizer ONNX model file
        """
        return f"{cls.RECOGNIZER_MODEL}.onnx"  # Always return cyrillic_g2.onnx

    def __init__(self, model_dir=None, use_gpu=True, use_onnx_preference=None, max_cache_size=None):
        """
        Initialize EasyOCR reader with languages hardcoded to ['en', 'ru']

        Args:
            model_dir: Optional directory for EasyOCR models
            use_gpu: Whether to use GPU for inference if available
            use_onnx_preference: Whether to prefer ONNX models for inference. If None, read from config.
            max_cache_size: Maximum number of results to cache. If None, read from config.
        """
        try:
            # import easyocr # Already imported at top
            pass
        except ImportError:
            raise ImportError("EasyOCR is not installed. Please install it with pip install easyocr")

        # Use parameters if provided, otherwise use values from settings.py
        self._use_onnx_preference = use_onnx_preference if use_onnx_preference is not None else EASYOCR_USE_ONNX
        self._max_cache_size = max_cache_size if max_cache_size is not None else EASYOCR_MAX_CACHE_SIZE

        logging.info(f"EasyOCR initialized with use_onnx={self._use_onnx_preference}, max_cache_size={self._max_cache_size}")

        languages = self.LANGUAGES

        # Initialize easyocr.Reader (PyTorch backend) first for model paths, vocab, and as a fallback
        # Determine effective model directory for EasyOCR .pth models
        pth_model_storage_directory = None
        if model_dir:
            pth_model_storage_directory = model_dir
        else:
            try:
                pth_model_storage_directory = os.path.join(get_project_base_directory(), "rag/res/easyocr")
                os.makedirs(pth_model_storage_directory, exist_ok=True)
            except Exception as e:
                logging.warning(f"Failed to create default model directory for .pth: {e}. EasyOCR will use its default.")
                # pth_model_storage_directory remains None, EasyOCR handles it.

        # Determine GPU availability for PyTorch backend of easyocr.Reader
        pytorch_use_gpu = use_gpu
        if use_gpu:
            try:
                import platform
                if platform.system() == "Darwin" and platform.processor() == "arm":
                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        logging.info("Using Apple Silicon GPU (MPS) for EasyOCR.Reader (PyTorch backend).")
                    else:
                        pytorch_use_gpu = False
                        logging.warning("Apple Silicon GPU (MPS) not available for PyTorch, EasyOCR.Reader will use CPU.")
                elif not torch.cuda.is_available():
                    pytorch_use_gpu = False
                    logging.warning("CUDA GPU requested for EasyOCR.Reader (PyTorch) but not available, falling back to CPU.")
                else:
                    logging.info("Using CUDA GPU for EasyOCR.Reader (PyTorch backend).")
            except ImportError: # PyTorch not available
                pytorch_use_gpu = False
                logging.warning("PyTorch not available, EasyOCR.Reader (PyTorch backend) will use CPU (should not happen if easyocr imported).")

        logging.info(f"Initializing EasyOCR.Reader (PyTorch backend) with languages {languages}, GPU={pytorch_use_gpu}, model_storage_directory='{pth_model_storage_directory if pth_model_storage_directory else 'EasyOCR default'}'")
        self.reader = easyocr.Reader(
            lang_list=languages, # Use hardcoded languages
            gpu=pytorch_use_gpu,
            model_storage_directory=pth_model_storage_directory,
            download_enabled=True,
            detector=True,
            recognizer=True,
            verbose=False # Set to True for more detailed EasyOCR init logs
        )
        logging.info("EasyOCR.Reader (PyTorch backend) initialized.")

        # --- Backend Setup (ONNX or PyTorch) ---
        self.detector = None
        self.recognizer = None
        self.actual_backend_is_onnx = False # Flag to track which backend is effectively used

        # Determine ONNX model names using the centralized constants and helpers
        detector_onnx_filename = self.get_detector_onnx_name()
        recognizer_onnx_filename = self.get_recognizer_onnx_name()

        if self._use_onnx_preference:
            logging.info("Attempting to initialize ONNX backend.")
            logging.info(f"Using standardized ONNX model names - Detector: '{detector_onnx_filename}', Recognizer: '{recognizer_onnx_filename}'")
            try:
                import onnxruntime as ort # Check if onnxruntime is available
                from .converter import download_models_from_hf # For downloading

                # Determine ONNX model directory (where models are stored or downloaded to)
                onnx_model_dir_base = pth_model_storage_directory if pth_model_storage_directory else os.path.join(get_project_base_directory(), "rag/res/easyocr")
                onnx_model_dir_specific = os.path.join(onnx_model_dir_base, "onnx") # Specific subdirectory for ONNX models
                os.makedirs(onnx_model_dir_specific, exist_ok=True)

                # Define paths for ONNX models and vocab
                det_path_local_onnx = os.path.join(onnx_model_dir_specific, detector_onnx_filename)
                rec_path_local_onnx = os.path.join(onnx_model_dir_specific, recognizer_onnx_filename)
                vocab_path_local_onnx = os.path.join(onnx_model_dir_specific, "vocab.txt") # Standard vocab name

                # Check if models exist, if not, download
                models_exist = os.path.exists(det_path_local_onnx) and \
                               os.path.exists(rec_path_local_onnx) and \
                               os.path.exists(vocab_path_local_onnx)

                if not models_exist:
                    logging.info(f"ONNX models or vocab.txt not found in {onnx_model_dir_specific}. Attempting download from HF: {EASYOCR_ONNX_REPO_ID}")
                    try:
                        # This should download to onnx_model_dir_specific or a subdir within it.
                        # Ensure download_models_from_hf returns the effective directory.
                        # Update paths if download_models_from_hf changed the structure (e.g. by creating repo-named subdir)
                        # For simplicity, assume download_models_from_hf places them into onnx_model_dir_specific directly
                        # or that load_onnx_model can find them within effective_download_dir.
                        # Re-check paths:
                        det_path_local_onnx = os.path.join(effective_download_dir, detector_onnx_filename)
                        rec_path_local_onnx = os.path.join(effective_download_dir, recognizer_onnx_filename)
                        vocab_path_local_onnx = os.path.join(effective_download_dir, "vocab.txt")

                        models_exist = os.path.exists(det_path_local_onnx) and \
                                       os.path.exists(rec_path_local_onnx) and \
                                       os.path.exists(vocab_path_local_onnx)
                        if models_exist:
                             logging.info(f"ONNX models and vocab found after download in {effective_download_dir}.")
                        else:
                             logging.warning(f"Expected ONNX models/vocab still not found in {effective_download_dir} after download attempt.")
                             # load_onnx_model will try auto-detection if specific files not found.
                    except Exception as e_download:
                        logging.error(f"Failed to download ONNX models from Hugging Face: {e_download}. Proceeding with PyTorch or existing ONNX if any.")
                        # Fall through, ONNX loading might still fail if models are truly absent.

                # Attempt to load ONNX models (GPU flag here is for ONNX Runtime)
                # load_onnx_model needs the directory where models are, and specific names if known.
                # If specific models (detector_onnx_filename, recognizer_onnx_filename) were not found even after download,
                # load_onnx_model's auto-detection for recognizer_name might kick in if recognizer_name is passed as None.
                try:
                    self.detector, self.recognizer = load_onnx_model(
                        onnx_model_dir_specific, # Search in this directory
                        use_gpu=use_gpu,      # This is the main use_gpu for ONNX Runtime
                        detector_name=detector_onnx_filename,
                        recognizer_name=recognizer_onnx_filename,
                        vocab_path=vocab_path_local_onnx if os.path.exists(vocab_path_local_onnx) else None # Pass vocab path to recognizer
                    )
                    self.actual_backend_is_onnx = True
                    logging.info(f"Successfully initialized ONNX backend. Detector: {type(self.detector).__name__}, Recognizer: {type(self.recognizer).__name__}.")
                except Exception as e_onnx_load:
                    logging.error(f"Failed to initialize ONNX backend: {e_onnx_load}. Falling back to PyTorch backend.", exc_info=True)
                    self.actual_backend_is_onnx = False

            except ImportError:
                logging.warning("ONNX Runtime (onnxruntime) not available. Falling back to PyTorch backend.")
                self.actual_backend_is_onnx = False
            except FileNotFoundError as fnf_e:
                 logging.warning(f"ONNX FileNotFoundError during ONNX setup: {fnf_e}. Falling back to PyTorch backend.")
                 self.actual_backend_is_onnx = False
            except Exception as e_onnx_load:
                logging.error(f"Failed to initialize ONNX backend: {e_onnx_load}. Falling back to PyTorch backend.", exc_info=True)
                self.actual_backend_is_onnx = False

        else: # self._use_onnx_preference is False
            logging.info("ONNX backend explicitly disabled by preference. Using PyTorch backend.")
            self.actual_backend_is_onnx = False

        # Fallback or explicit choice for PyTorch backend
        if not self.actual_backend_is_onnx:
            self.detector = PyTorchDetector(self.reader)
            self.recognizer = PyTorchRecognizer(self.reader)
            logging.info(f"Using PyTorch backend. Detector: {type(self.detector).__name__}, Recognizer: {type(self.recognizer).__name__}.")

        # Common initialization
        self.drop_score = 0.5
        self._cache_lock = RLock()
        self._cache = OrderedDict()
        # self.crop_image_res_index = 0 # This seems unused

    def _cache_put(self, key, value):
        with self._cache_lock:
            if len(self._cache) >= self._max_cache_size:
                self._cache.popitem(last=False) # FIFO
            self._cache[key] = value

    def _cache_get(self, key):
        with self._cache_lock:
            if key in self._cache:
                value = self._cache.pop(key) # Move to end (most recently used)
                self._cache[key] = value
                return value
            return None

    def _generate_cache_key(self, img):
        if img is None: return None
        # Simplified key generation for brevity in example
        shape = img.shape
        # A more robust hash would sample pixels, but for now, shape + first few pixels
        sample_data = img.flat[:min(100, img.size)].tobytes()
        m = hashlib.md5()
        m.update(str(shape).encode())
        m.update(sample_data)
        return m.hexdigest()

    def _ensure_rgb(self, img):
        if img is None: return None
        if img.size == 0:
            logging.warning("Empty image provided to _ensure_rgb")
            return None
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Assume BGR input from cv2.imread, convert to RGB
            # This is a common convention, but could be wrong if image is already RGB.
            # For robustness, downstream components (detectors/recognizers) should clarify their input expectation.
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2: # Grayscale image
            # Convert to RGB for components that might expect 3 channels
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 4: # Image with Alpha channel (e.g., RGBA, BGRA)
            # Convert to RGB, removing alpha. Assuming BGRA from common cv2 operations.
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        # If it's already RGB or some other 3-channel format not BGR, or a format not handled above,
        # it might be returned as is, or logged.
        # For now, if not fitting above, return as is but log.
        logging.debug(f"_ensure_rgb: Image not explicitly converted to RGB. Shape: {img.shape}. May rely on downstream handling.")
        return img

    def sorted_boxes(self, dt_boxes):
        if not dt_boxes: # Handles None or empty list input
            return []
        try:
            # Ensure all elements are list/array-like and represent valid boxes (4 points, each with 2 coords)
            # Convert to numpy arrays of float32 for robust processing, then convert back to list of lists of ints.
            valid_boxes_np = []
            for i, b in enumerate(dt_boxes):
                if isinstance(b, (list, np.ndarray)) and len(b) == 4 and \
                   all(isinstance(p, (list, np.ndarray, tuple)) and len(p) == 2 for p in b):
                    try:
                        # Attempt to convert to float for sorting, then int for output
                        valid_boxes_np.append(np.array(b, dtype=np.float32))
                    except Exception as e_convert:
                        logging.warning(f"sorted_boxes: Error converting box {b} to np.float32 array: {e_convert}. Skipping.")
                        continue
                else:
                    logging.warning(f"sorted_boxes: Skipping malformed/incomplete box candidate [{i}]: {b}")

            if not valid_boxes_np:
                logging.debug("sorted_boxes: No valid boxes found after filtering. Returning [].")
                return []

            # Sort by top-y, then left-x coordinate of the first point (top-left point)
            _boxes_np = sorted(valid_boxes_np, key=lambda x_np: (x_np[0][1], x_np[0][0]))

            # Refine sorting for lines (bubble sort like logic from original EasyOCR)
            num_boxes = len(_boxes_np)
            for i in range(num_boxes - 1):
                for j in range(i, -1, -1): # Iterate backwards from i
                    if j + 1 < num_boxes:
                        # If y-coordinates are close (within 10px) and box[j+1] is to the left of box[j]
                        if abs(_boxes_np[j+1][0][1] - _boxes_np[j][0][1]) < 10 and \
                           (_boxes_np[j+1][0][0] < _boxes_np[j][0][0]):
                            # Swap
                            _boxes_np[j], _boxes_np[j+1] = _boxes_np[j+1], _boxes_np[j]
                        else:
                            # Inner loop condition not met, break to next i
                            break # to the outer loop (next i)

            # Convert final sorted boxes (numpy arrays) to list of lists of integers
            final_sorted_boxes = []
            for box_np in _boxes_np:
                try:
                    final_sorted_boxes.append([[int(p[0]), int(p[1])] for p in box_np])
                except Exception as e_int_convert:
                    logging.warning(f"sorted_boxes: Error converting box {box_np} to list of ints: {e_int_convert}. Skipping.")
            return final_sorted_boxes
        except Exception as e:
            logging.error(f"Error during box sorting: {e}. Input boxes (original): {dt_boxes}", exc_info=True)
            # Fallback: try to return original dt_boxes if they are already in list-of-lists format, else empty
            if isinstance(dt_boxes, list) and all(isinstance(b, list) for b in dt_boxes):
                return dt_boxes
            return []

    def detect(self, img, device_id=None): # device_id seems unused with new structure
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0} # cls is not used by EasyOCR
        if img is None:
            return [], time_dict # Return empty list for boxes

        cache_key = self._generate_cache_key(img)
        if cache_key:
            cached_result = self._cache_get(f"detect_{cache_key}")
            if cached_result:
                sorted_boxes_list, detection_time = cached_result
                time_dict['det'] = detection_time
                time_dict['all'] = detection_time
                logging.debug(f"Cache hit for detect: {cache_key}")
                # Для совместимости с deepdoc/vision/ocr.py возвращаем zip объект в том же формате
                return zip(sorted_boxes_list, [("", 0.0) for _ in range(len(sorted_boxes_list))])

        start_time = time.time()
        # Ensure image is RGB, as both PyTorchDetector and ONNXDetector might expect it
        # (though ONNXDetector internally handles normalization for 3-channel).
        img_rgb = self._ensure_rgb(img)
        if img_rgb is None:
            return []

        # Delegate to the configured detector
        detected_boxes = self.detector.detect(img_rgb)

        detection_time = time.time() - start_time
        time_dict['det'] = detection_time
        time_dict['all'] = detection_time # For now, 'all' is just detection time here

        # Sort boxes
        sorted_boxes_list = self.sorted_boxes(detected_boxes if detected_boxes else [])

        if cache_key:
            logging.debug(f"Caching result for detect: {cache_key}")
            self._cache_put(f"detect_{cache_key}", (sorted_boxes_list, detection_time))

        # Возвращаем в формате, совместимом с deepdoc/vision/ocr.py
        return zip(sorted_boxes_list, [("", 0.0) for _ in range(len(sorted_boxes_list))])

    def get_rotate_crop_image(self, img, points):
        if img is None or points is None:
            raise ValueError("Image or points cannot be None for get_rotate_crop_image")

        points_arr = np.array(points, dtype=np.float32)
        if points_arr.shape != (4, 2):
            if points_arr.size == 8:
                try: points_arr = points_arr.reshape((4,2))
                except ValueError: raise ValueError(f"Points array size 8 but cannot reshape to (4,2). Shape: {points_arr.shape}")
            else: raise ValueError(f"Shape of points must be 4x2. Got {points_arr.shape}")

        w_crop = int(max(np.linalg.norm(points_arr[0] - points_arr[1]), np.linalg.norm(points_arr[2] - points_arr[3])))
        h_crop = int(max(np.linalg.norm(points_arr[0] - points_arr[3]), np.linalg.norm(points_arr[1] - points_arr[2])))
        w_crop = max(1, w_crop); h_crop = max(1, h_crop)

        pts_std = np.float32([[0,0],[w_crop,0],[w_crop,h_crop],[0,h_crop]])
        try:
            M = cv2.getPerspectiveTransform(points_arr, pts_std)
        except cv2.error as e:
            logging.error(f"cv2.getPerspectiveTransform failed: {e}. Points: {points_arr}, Target: {pts_std}")
            num_channels = img.shape[2] if len(img.shape) == 3 else 1
            return np.zeros((h_crop, w_crop, num_channels) if num_channels > 1 else (h_crop, w_crop), dtype=img.dtype)

        dst_img = cv2.warpPerspective(img, M, (w_crop, h_crop), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
        dst_h, dst_w = dst_img.shape[0:2]
        if dst_h * 1.0 / max(1, dst_w) >= 1.5: # Avoid division by zero
            dst_img = np.rot90(dst_img)
        return dst_img

    def recognize(self, ori_im, box): # Removed device_id, seems unused
        """
        Recognizes text in a single bounding box from an original image.
        Handles cropping, caching, and applying drop_score.
        """
        if ori_im is None or box is None: return None # (text, confidence) is expected, so None is fine.

        # Generate cache key based on original image and box coordinates
        img_key_for_cache = self._generate_cache_key(ori_im)
        box_str_for_cache = "_".join(map(str, np.array(box).astype(int).flatten()))
        cache_key = None
        if img_key_for_cache:
            cache_key = f"recognize_{img_key_for_cache}_{box_str_for_cache}"
            cached_val = self._cache_get(cache_key)
            if cached_val:
                logging.debug(f"Recognize CACHE HIT for box {box_str_for_cache}: {cached_val}")
                return cached_val # Cached value is (text, confidence) or None

        try:
            img_crop = self.get_rotate_crop_image(ori_im, box)
            logging.debug(f"Recognize: Cropped image shape for {box_str_for_cache}: {img_crop.shape if img_crop is not None else 'None'}")
        except ValueError as e:
            logging.error(f"Failed to crop image for recognition (box {box_str_for_cache}): {e}")
            return None

        if img_crop is None or img_crop.size == 0:
            logging.warning(f"Cropped image for recognition is empty for box {box_str_for_cache}.")
            return None

        # Ensure crop is RGB for consistency, as recognizers might expect it.
        img_crop_rgb = self._ensure_rgb(img_crop) # PyTorchRecognizer and ONNXRecognizer might handle further specifics.
                                               # ONNXRecognizer converts to gray internally if needed.
                                               # PyTorchRecognizer passes RGB to reader.readtext which handles it.

        # Delegate to the configured recognizer
        # recognizer.recognize should return (text, confidence)
        recognition_result = self.recognizer.recognize(img_crop_rgb)

        text, confidence = "", 0.0
        if recognition_result:
            text, confidence = recognition_result

        logging.debug(f"Recognize BEFORE drop_score for box {box_str_for_cache}: Text='{text}', Confidence={confidence:.4f}, DropScore={self.drop_score}")

        final_result_tuple = (text, confidence) if confidence >= self.drop_score else None
        logging.debug(f"Recognize AFTER drop_score for box {box_str_for_cache}: ResultTuple={final_result_tuple}")

        if cache_key: self._cache_put(cache_key, final_result_tuple)
        return final_result_tuple

    def recognize_batch(self, img_list, device_id=None): # device_id unused
        # This method is less critical if __call__ processes one by one.
        # If batching is truly needed at recognizer level, detector and recognizer classes should expose batch methods.
        # For now, retain simple loop calling self.recognize, which is cached.
        # Or, PyTorchRecognizer could implement a batch method using reader.recognize_batched if available and useful.
        # ONNXRecognizer would need a separate batch implementation.
        # For simplicity, keeping the loop based on single recognize calls.
        logging.warning("EasyOCR.recognize_batch is currently implemented as a loop over single recognize calls. For performance, direct batch recognition in backend classes would be better if needed.")
        batch_output = []
        for img_crop_item in img_list:
            if img_crop_item is None or img_crop_item.size == 0:
                batch_output.append(None)
                continue

            # Here, img_crop_item is the already cropped image.
            # We need a cache key for this crop directly if we bypass self.recognize's caching.
            # However, self.recognize is called with (original_image, box).
            # This batch method takes list of CROPS.
            # For consistency with current recognize_batch, let's assume img_list contains pre-cropped images.

            # Ensure crop is RGB
            img_crop_rgb = self._ensure_rgb(img_crop_item)

            key_crop_batch = self._generate_cache_key(img_crop_rgb) # Cache based on crop itself
            cached_crop_res = None
            if key_crop_batch:
                cached_crop_res = self._cache_get(f"rec_batch_crop_{key_crop_batch}")

            if cached_crop_res:
                # Если в кеше уже есть результат, извлекаем текст или возвращаем пустую строку
                text = cached_crop_res[0] if cached_crop_res else ""
                batch_output.append(text)
                continue

            # Call the backend recognizer directly
            rec_res_tuple = self.recognizer.recognize(img_crop_rgb) # (text, confidence)

            current_text, current_conf = "", 0.0
            if rec_res_tuple:
                current_text, current_conf = rec_res_tuple

            final_crop_result = (current_text, current_conf) if current_conf >= self.drop_score else None

            if key_crop_batch:
                self._cache_put(f"rec_batch_crop_{key_crop_batch}", final_crop_result)

            # Для совместимости с pdf_parser.py возвращаем только текст
            # pdf_parser ожидает строку, а не кортеж (текст, уверенность)
            batch_output.append(current_text if current_conf >= self.drop_score else "")

        return batch_output

    def __call__(self, img, device_id=0, cls=True): # cls and device_id are not actively used with new structure
        time_dict = {'det': 0, 'rec': 0, 'all': 0}
        if img is None:
            return [] # List of (box, (text, confidence))

        # Full __call__ caching
        full_cache_key = self._generate_cache_key(img)
        if full_cache_key:
            cached_data = self._cache_get(f"call_{full_cache_key}")
            if cached_data:
                # Assuming cached_data is (ocr_final_results, time_dict_all_val)
                res_list, all_t_cached = cached_data
                logging.debug(f"Full __call__ cache hit for {full_cache_key}. Returning cached results list.")
                # time_dict is not fully reconstructed here, but caller gets the final list.
                # To be more accurate, cache time_dict as well or recalculate from components if needed.
                # For now, just return the results list.
                return res_list # Original EasyOCR returns list of (box, (text, conf))

        overall_start_time = time.time()

        # 1. Detect boxes - now detect returns a zip object, not a tuple
        detected_boxes = self.detect(img)

        # Превращаем итератор в список
        detected_boxes_list = list(detected_boxes)

        if not detected_boxes_list:
            time_dict['all'] = time.time() - overall_start_time
            if full_cache_key: self._cache_put(f"call_{full_cache_key}", ([], time_dict['all']))
            return []

        # 2. Recognize text in each box - using our own recognize implementation
        ocr_final_results = []
        recognition_start_time = time.time()

        for box, (_, _) in detected_boxes_list:
            # self.recognize handles cropping, backend call, drop_score, and its own caching
            rec_tuple = self.recognize(img, box) # Pass original image and box

            # Если распознавание вернуло результат, добавить в итоговый список
            if rec_tuple:
                text, confidence = rec_tuple
                box_to_store = box.tolist() if isinstance(box, np.ndarray) else box
                ocr_final_results.append((box_to_store, (text, confidence)))

        time_dict['rec'] = time.time() - recognition_start_time
        time_dict['all'] = time.time() - overall_start_time

        if full_cache_key:
            self._cache_put(f"call_{full_cache_key}", (ocr_final_results, time_dict['all']))

        return ocr_final_results

    def clear_cache(self):
        with self._cache_lock:
            self._cache.clear()
            logging.info("EasyOCR cache cleared")

    def _convert_to_onnx_models(self, output_dir=None, upload_to_hf=False, hf_repo_id=None, hf_token=None):
        # This method uses self.reader, which is initialized.
        # It's largely independent of whether ONNX or PyTorch is the *active* backend for inference.
        if self.actual_backend_is_onnx: # Check if ONNX is already active due to successful load
             logging.warning("ONNX models might already be in use or available. Conversion process can still run to regenerate them if needed.")

        from .converter import convert_to_onnx # Import here to avoid circularity if converter uses EasyOCR
        languages_for_conversion = self.reader.lang_list # Get languages from the initialized reader

        # Determine GPU for conversion process (can be different from runtime use_gpu)
        # convert_to_onnx itself forces CPU for stability, so this flag is advisory for its internal logic if changed.
        conversion_use_gpu_preference = True # Or make this a parameter of _convert_to_onnx_models

        if output_dir is None:
            try:
                proj_base = get_project_base_directory()
                output_dir = os.path.join(proj_base, "rag/res/easyocr/onnx_converted")
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e_conv_dir:
                logging.error(f"Could not create default ONNX conversion output_dir: {e_conv_dir}. Conversion may fail or use current dir.")
                output_dir = "."

        logging.info(f"Starting ONNX model conversion for languages: {languages_for_conversion}. Output to: {output_dir}")
        try:
            # convert_to_onnx takes lang_list, output_dir, use_gpu for its process, etc.
            # It uses its own EasyOCR.Reader instance internally for .pth models.
            # The self.reader here is mainly for getting lang_list.
            converted_output_path, det_name, rec_name = convert_to_onnx(
                languages=languages_for_conversion,
                output_dir=output_dir,
                use_gpu=conversion_use_gpu_preference, # Passed to convert_to_onnx
                upload_to_hf=upload_to_hf,
                hf_repo_id=hf_repo_id,
                hf_token=hf_token
            )
            logging.info(f"ONNX Models converted: Detector='{det_name}', Recognizer='{rec_name}'. Saved in '{converted_output_path}'.")
            logging.info("To use these converted models, re-initialize EasyOCR with 'use_onnx_preference=True' and ensure models are in the expected path.")
            return converted_output_path
        except Exception as e_conversion:
            logging.error(f"ONNX model conversion failed: {e_conversion}", exc_info=True)
            return None

    def recognize_text(self, img, min_confidence=None, paragraph=False, return_boxes=False):
        """
        Simple method for text recognition from an image.
        Performs detection and recognition in a single method and returns the recognized text.

        Args:
            img: Input image (numpy array, BGR or RGB)
            min_confidence: Minimum confidence threshold for including a result (if None, uses self.drop_score)
            paragraph: Merge text into paragraphs (True) or keep as separate lines (False)
            return_boxes: If True, return list of dicts with box/text/confidence (for lines) or box/text (for paragraphs)

        Returns:
            str: Recognized text (all lines joined with line breaks) if return_boxes is False
            list: List of dicts with box/text/confidence if return_boxes is True
        """
        if self._use_onnx_preference:
            raise NotImplementedError("ONNX recognition is not implemented yet")

        if img is None:
            return [] if return_boxes else ""

        img_rgb = self._ensure_rgb(img)
        cache_key = self._generate_cache_key(img_rgb)
        if cache_key and not return_boxes:
            cached_result = self._cache_get(f"recognize_text_{cache_key}_{paragraph}")
            if cached_result:
                logging.debug(f"Cache hit for recognize_text: {cache_key}")
                return cached_result

        confidence_threshold = self.drop_score if min_confidence is None else min_confidence
        try:
            start_time = time.time()
            results = self.reader.readtext(img_rgb, paragraph=paragraph)
            duration = time.time() - start_time
            logging.debug(f"Native EasyOCR readtext took {duration:.3f} seconds, returned {len(results)} results")

            if return_boxes:
                if paragraph:
                    # results: [ [box, text], ... ]
                    return [
                        {"box": r[0], "text": r[1]}
                        for r in results if len(r) >= 2
                    ]
                else:
                    # results: [ [box, text, confidence], ... ]
                    return [
                        {"box": r[0], "text": r[1], "confidence": r[2]}
                        for r in results if len(r) >= 3 and r[2] >= confidence_threshold
                    ]
            # Old behavior (text only)
            if paragraph:
                filtered_texts = [r[1] for r in results if len(r) >= 2]
            else:
                filtered_texts = [r[1] for r in results if len(r) >= 3 and r[2] >= confidence_threshold]
            result_text = "\n".join(filtered_texts).rstrip("\n")

            if cache_key and not return_boxes:
                self._cache_put(f"recognize_text_{cache_key}_{paragraph}", result_text)
            return result_text
        except Exception as e:
            logging.error(f"Error in recognize_text: {e}", exc_info=True)
            return [] if return_boxes else ""

# Potential alias for backward compatibility if other parts of the project import OCR from here.
# OCR = EasyOCR
# This should be handled by the importing module to decide which OCR class to use.