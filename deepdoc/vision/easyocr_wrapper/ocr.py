from api.utils.file_utils import get_project_base_directory
from deepdoc.vision.operators import cv2, logging, np
from huggingface_hub import snapshot_download


import cv2
import easyocr
print(f"easyocr module loaded from: {easyocr.__file__}") # DEBUGGING LINE
import numpy as np
import torch


import hashlib
import logging
import os
import time


class ONNXDetector:
    """
    ONNX Implementation of EasyOCR detector
    """
    def __init__(self, model_path, use_gpu=True):
        import onnxruntime as ort

        # Configuration for ONNX Runtime session
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Determine compute provider
        providers = []
        if use_gpu:
            # Try to use GPU
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                # NVIDIA GPU
                cuda_provider_options = {
                    "device_id": 0,
                    "gpu_mem_limit": 512 * 1024 * 1024,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                }
                providers.append(('CUDAExecutionProvider', cuda_provider_options))
                logging.info("Using CUDA for ONNX detector")
            elif 'CoreMLExecutionProvider' in ort.get_available_providers():
                # Apple Silicon
                providers.append('CoreMLExecutionProvider')
                logging.info("Using CoreML for ONNX detector")
            else:
                logging.warning("GPU requested but no GPU provider available for ONNX Runtime")
                providers.append('CPUExecutionProvider')
        else:
            providers.append('CPUExecutionProvider')

        # Load the model
        self.session = ort.InferenceSession(model_path, options=options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        # Define input and output shapes
        logging.info(f"Detector input shape: {self.input_shape}")

        # Configuration for detection
        self.min_size = 3
        self.text_threshold = 0.7
        self.low_text = 0.4
        self.link_threshold = 0.4

    def detect(self, image):
        """
        Detect text regions in an image

        Args:
            image: Input image as numpy array (RGB)

        Returns:
            List of bounding boxes in format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        # Prepare image for detector
        h, w = image.shape[:2]

        # Scale image to model input shape
        target_height, target_width = 640, 640  # Common size for detector
        ratio_h, ratio_w = h / target_height, w / target_width
        img_resized = cv2.resize(image, (target_width, target_height))

        # Normalize image
        img_norm = img_resized.astype(np.float32) / 255.0
        img_norm = img_norm.transpose(2, 0, 1)  # HWC -> CHW
        img_norm = np.expand_dims(img_norm, axis=0)  # Add batch dimension

        # Run inference
        outputs = self.session.run(None, {self.input_name: img_norm})

        # Process model output
        # Specific logic for processing detector outputs
        boxes = []
        # Simple implementation for example:
        # Assume output is a confidence map where higher value = higher text probability
        output_map = outputs[0][0]  # Take first feature map

        # Threshold processing
        binary_map = output_map > self.text_threshold

        # Find contours
        from skimage import measure
        labels = measure.label(binary_map.astype(np.uint8))
        regions = measure.regionprops(labels)

        for region in regions:
            # Filter out too small regions
            if region.area < self.min_size:
                continue

            # Get bounding rectangle for region
            min_row, min_col, max_row, max_col = region.bbox

            # Scale back to original size
            x1, y1 = min_col * ratio_w, min_row * ratio_h
            x2, y2 = max_col * ratio_w, min_row * ratio_h
            x3, y3 = max_col * ratio_w, max_row * ratio_h
            x4, y4 = min_col * ratio_w, max_row * ratio_h

            boxes.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

        return boxes


class ONNXRecognizer:
    """
    ONNX Implementation of EasyOCR recognizer
    """
    def __init__(self, model_path, vocab_path=None, use_gpu=True):
        import onnxruntime as ort

        # Configuration for ONNX Runtime session
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Determine compute provider
        providers = []
        if use_gpu:
            # Try to use GPU
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                # NVIDIA GPU
                cuda_provider_options = {
                    "device_id": 0,
                    "gpu_mem_limit": 512 * 1024 * 1024,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                }
                providers.append(('CUDAExecutionProvider', cuda_provider_options))
                logging.info("Using CUDA for ONNX recognizer")
            elif 'CoreMLExecutionProvider' in ort.get_available_providers():
                # Apple Silicon
                providers.append('CoreMLExecutionProvider')
                logging.info("Using CoreML for ONNX recognizer")
            else:
                logging.warning("GPU requested but no GPU provider available for ONNX Runtime")
                providers.append('CPUExecutionProvider')
        else:
            providers.append('CPUExecutionProvider')

        # Load the model
        self.session = ort.InferenceSession(model_path, options=options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

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

        # Scale image preserving aspect ratio
        ratio = target_height / h
        target_width = int(w * ratio)

        # Check minimum size
        target_width = max(target_width, 16)  # Minimum width

        # Resize image
        img_resized = cv2.resize(image, (target_width, target_height))

        # Normalize image
        img_norm = img_resized.astype(np.float32) / 255.0
        img_norm = img_norm.transpose(2, 0, 1)  # HWC -> CHW
        img_norm = np.expand_dims(img_norm, axis=0)  # Add batch dimension

        # Run inference
        outputs = self.session.run(None, {self.input_name: img_norm})

        # Process model output
        # Assume output contains predictions for each character
        predictions = outputs[0]

        # Decode character sequence
        # CTC decoding (simplified)
        text = self._decode_prediction(predictions[0])

        return text

    def _decode_prediction(self, prediction):
        """
        Decode model prediction to text

        Args:
            prediction: Model output data

        Returns:
            Recognized text
        """
        # Find indices with maximum probability for each time step
        indices = np.argmax(prediction, axis=1)

        # Remove repeated indices
        prev_index = -1
        result_indices = []
        for idx in indices:
            if idx != prev_index:
                result_indices.append(idx)
                prev_index = idx

        # Remove blank characters (usually index 0 or len(vocab))
        blank_index = 0
        result_indices = [idx for idx in result_indices if idx != blank_index]

        # Convert indices to characters
        text = ''.join([self.vocab[idx] if idx < len(self.vocab) else '' for idx in result_indices])

        return text


def load_onnx_model(model_dir, use_gpu=True, detector_name="craft_mlt_25k.onnx", recognizer_name=None):
    """
    Load ONNX models for OCR

    Args:
        model_dir: Directory containing ONNX models
        use_gpu: Whether to use GPU for inference
        detector_name: Name of the detector model file (default: craft_mlt_25k.onnx)
        recognizer_name: Name of the recognizer model file (auto-detected if None)

    Returns:
        Tuple of (detector, recognizer)
    """
    # Check for model files
    det_path = os.path.join(model_dir, detector_name)

    # If recognizer_name not specified, try to find it in the directory
    if recognizer_name is None:
        # Look for any .onnx file that is not the detector model
        rec_files = [f for f in os.listdir(model_dir)
                    if f.endswith('.onnx') and f != detector_name]
        if rec_files:
            recognizer_name = rec_files[0]
            logging.info(f"Auto-detected recognizer model: {recognizer_name}")
        else:
            raise FileNotFoundError(f"No recognizer model found in {model_dir}")

    rec_path = os.path.join(model_dir, recognizer_name)
    vocab_path = os.path.join(model_dir, "vocab.txt")

    if not os.path.exists(det_path):
        raise FileNotFoundError(f"Detector model not found: {det_path}")

    if not os.path.exists(rec_path):
        raise FileNotFoundError(f"Recognizer model not found: {rec_path}")

    # Load models
    detector = ONNXDetector(det_path, use_gpu=use_gpu)

    vocab_file = vocab_path if os.path.exists(vocab_path) else None
    recognizer = ONNXRecognizer(rec_path, vocab_path=vocab_file, use_gpu=use_gpu)

    return detector, recognizer


class EasyOCR:
    """
    OCR implementation using EasyOCR library to support multiple languages including Russian.
    Maintains interface compatibility with the original OCR class.
    """
    # Hugging Face repository for ONNX models
    # Can be changed to your own repository if needed
    ONNX_REPO_ID = "InfiniFlow/easyocr-onnx-models"

    def __init__(self, model_dir=None, languages=None, use_gpu=True, max_cache_size=100):
        """
        Initialize EasyOCR reader with specified languages

        Args:
            model_dir: Optional directory for EasyOCR models
            languages: List of language codes, defaults to ['en', 'ru'] if None
            use_gpu: Whether to use GPU for inference if available
            max_cache_size: Maximum number of results to cache
        """
        try:
            import easyocr
        except ImportError:
            raise ImportError("EasyOCR is not installed. Please install it with pip install easyocr")

        # Default languages if not specified
        if languages is None:
            languages = ['en', 'ru']

        # Determine GPU usage
        gpu = use_gpu
        if use_gpu:
            try:
                import torch
                import platform

                # Check for Apple M1/M2 Mac
                if platform.system() == "Darwin" and platform.processor() == "arm":
                    # Check MPS (Metal Performance Shaders) support
                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        gpu = True
                        logging.info("Using Apple Silicon GPU via MPS")
                    else:
                        gpu = False
                        logging.warning("Apple Silicon GPU via MPS not available")
                else:
                    # Standard CUDA check
                    gpu = torch.cuda.is_available()
                    if not gpu:
                        logging.warning("GPU requested but not available, falling back to CPU")
            except ImportError:
                gpu = False
                logging.warning("PyTorch not available, falling back to CPU")

        # Configure models directory
        download_dir = None
        if model_dir:
            download_dir = model_dir
        else:
            try:
                download_dir = os.path.join(
                    get_project_base_directory(),
                    "rag/res/easyocr")
                os.makedirs(download_dir, exist_ok=True)
            except Exception as e:
                logging.warning(f"Failed to create model directory: {e}")
                # Use default EasyOCR directory if project directory is not available

        # Initialize EasyOCR Reader first to get model paths
        logging.info(f"Initializing EasyOCR with languages {languages}, GPU={gpu}")
        self.reader = easyocr.Reader(
            languages,
            gpu=gpu,
            model_storage_directory=download_dir,
            download_enabled=True,
            detector=True,
            recognizer=True
        )

        # Automatic detection and use of ONNX models
        self.use_onnx = False
        self.onnx_detector = None
        self.onnx_recognizer = None

        # Get original model names
        try:
            detector_original = os.path.basename(self.reader.detector_path)
            recognizer_original = os.path.basename(self.reader.recognizer_path)

            # Expected ONNX model names (based on original names)
            detector_onnx = os.path.splitext(detector_original)[0] + ".onnx"
            recognizer_onnx = os.path.splitext(recognizer_original)[0] + ".onnx"

            logging.info(f"Original detector model: {detector_original}")
            logging.info(f"Original recognizer model: {recognizer_original}")
            logging.info(f"Expecting ONNX models: {detector_onnx} and {recognizer_onnx}")
        except Exception as e:
            logging.warning(f"Could not get original model names: {e}")
            detector_onnx = "craft_mlt_25k.onnx"
            recognizer_onnx = "cyrillic_g2.onnx"

        # Check for and initialize ONNX models
        try:
            import onnxruntime as ort
            from .converter import download_models_from_hf

            # Check for local ONNX models
            onnx_model_dir = os.path.join(download_dir, "onnx")
            os.makedirs(onnx_model_dir, exist_ok=True)

            # Check if expected models exist locally
            det_path = os.path.join(onnx_model_dir, detector_onnx)
            rec_path = os.path.join(onnx_model_dir, recognizer_onnx)

            models_exist = os.path.exists(det_path) and os.path.exists(rec_path)

            # If models not found locally, try to download from HF
            if not models_exist:
                try:
                    logging.info(f"ONNX models not found locally, trying to download from {self.ONNX_REPO_ID}")
                    onnx_model_dir = download_models_from_hf(self.ONNX_REPO_ID, onnx_model_dir)

                    # Check again if expected models exist after download
                    det_path = os.path.join(onnx_model_dir, detector_onnx)
                    rec_path = os.path.join(onnx_model_dir, recognizer_onnx)
                    models_exist = os.path.exists(det_path) and os.path.exists(rec_path)

                    if not models_exist:
                        # Look for any ONNX models
                        onnx_files = [f for f in os.listdir(onnx_model_dir) if f.endswith('.onnx')]
                        if len(onnx_files) >= 2:
                            logging.info(f"Found alternative ONNX models: {onnx_files}")
                except Exception as e:
                    logging.warning(f"Failed to download ONNX models: {e}")

            # Try to load ONNX models
            try:
                if models_exist:
                    # Load with exact model names
                    self.onnx_detector, self.onnx_recognizer = load_onnx_model(
                        onnx_model_dir,
                        use_gpu=gpu,
                        detector_name=detector_onnx,
                        recognizer_name=recognizer_onnx
                    )
                    self.use_onnx = True
                    logging.info(f"Using ONNX models: {detector_onnx} and {recognizer_onnx}")
                else:
                    # Try auto-detection
                    logging.info("Trying automatic ONNX model detection...")
                    self.onnx_detector, self.onnx_recognizer = load_onnx_model(
                        onnx_model_dir,
                        use_gpu=gpu
                    )
                    self.use_onnx = True
                    logging.info("Successfully loaded ONNX models through auto-detection")
            except Exception as e:
                logging.error(f"Failed to load ONNX models: {e}")
        except ImportError:
            logging.warning("ONNX Runtime not available, using PyTorch models")
        except Exception as e:
            logging.warning(f"Error checking ONNX models: {e}")

        # For compatibility with the original OCR class
        self.drop_score = 0.5

        # Thread-safe cache with size limit
        from collections import OrderedDict
        from threading import RLock
        self._max_cache_size = max_cache_size
        self._cache_lock = RLock()
        self._cache = OrderedDict()
        self.crop_image_res_index = 0  # For compatibility with original OCR

    def _cache_put(self, key, value):
        """Thread-safe cache insertion with size limit"""
        with self._cache_lock:
            # If cache is full, remove oldest item
            if len(self._cache) >= self._max_cache_size:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def _cache_get(self, key):
        """Thread-safe cache retrieval"""
        with self._cache_lock:
            if key in self._cache:
                # Move accessed item to end (most recently used)
                value = self._cache.pop(key)
                self._cache[key] = value
                return value
            return None

    def _generate_cache_key(self, img):
        """Generate a cache key for an image using a memory-efficient approach"""
        if img is None:
            return None
        # Use image shape and a sample of pixels instead of the full image bytes
        shape = img.shape
        # Sample pixels from different regions of the image
        sample_size = min(1000, img.size // 10)  # Limit sample size
        step = max(1, img.size // sample_size)
        samples = img.ravel()[::step]
        # Create a hash from the shape and samples
        import hashlib
        m = hashlib.md5()
        m.update(str(shape).encode())
        m.update(samples.tobytes())
        return m.hexdigest()

    def _ensure_rgb(self, img):
        """
        Ensure image is in RGB format for EasyOCR processing

        Args:
            img: Input image

        Returns:
            RGB version of the image
        """
        if img is None:
            return None

        # Check if image is empty
        if img.size == 0:
            logging.warning("Empty image provided to _ensure_rgb")
            return None

        # Check if image has 3 channels (color image)
        if len(img.shape) == 3 and img.shape[2] == 3:
            # OpenCV uses BGR, convert to RGB
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Grayscale or other format, no need to convert
        return img

    def sorted_boxes(self, dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right.
        This method is copied from the original OCR implementation to maintain compatibility.

        Args:
            dt_boxes: Detected text boxes

        Returns:
            Sorted boxes
        """
        if not dt_boxes or len(dt_boxes) == 0:
            return []

        num_boxes = len(dt_boxes)
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if j + 1 < len(_boxes) and abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                        (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def detect(self, img, device_id=None):
        """
        Detect text in an image without full recognition.
        For compatibility with the original OCR class.

        Args:
            img: Input image as numpy array
            device_id: For compatibility with original interface

        Returns:
            Iterator of (box, ("", 0)) pairs and time_dict with timing info
        """
        if device_id is None:
            device_id = 0

        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}

        if img is None:
            return None, None, time_dict

        # Generate a cache key for this image
        cache_key = self._generate_cache_key(img)
        cached_result = self._cache_get(f"detect_{cache_key}")
        if cached_result:
            # Return cached result
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("Using cached detection results")
            boxes, timing = cached_result
            time_dict['det'] = timing
            time_dict['all'] = timing
            return zip(boxes, [("", 0) for _ in range(len(boxes))])

        start = time.time()

        # Convert to RGB for processing
        img_rgb = self._ensure_rgb(img)
        if img_rgb is None:
            time_dict['all'] = 0
            return None, None, time_dict

        # Perform detection
        if self.use_onnx and self.onnx_detector:
            # Use ONNX model for detection
            boxes = self.onnx_detector.detect(img_rgb)
        else:
            # Use PyTorch model from EasyOCR
            results = self.reader.readtext(img_rgb)
            boxes = [box for box, _, _ in results]

        end = time.time()
        timing = end - start
        time_dict['det'] = timing
        time_dict['all'] = timing

        # Sort boxes in a similar way to the original OCR's sorted_boxes method
        sorted_boxes = self.sorted_boxes(boxes)

        # Cache the result
        self._cache_put(f"detect_{cache_key}", (sorted_boxes, timing))

        return zip(sorted_boxes, [("", 0) for _ in range(len(sorted_boxes))])

    def get_rotate_crop_image(self, img, points):
        """
        Crop and potentially rotate an image based on the given points.
        This method is copied from the original OCR implementation to maintain compatibility.
        """
        if img is None or points is None:
            logging.warning("Null image or points provided to get_rotate_crop_image")
            raise ValueError("Image or points cannot be None")

        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float32)

        # This assert follows the original implementation
        assert len(points) == 4, "shape of points must be 4*2"

        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))

        # Check for very small crops which may cause issues
        if img_crop_width < 5 or img_crop_height < 5:
            logging.warning(f"Very small crop requested: {img_crop_width}x{img_crop_height}")

        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                            [img_crop_width, img_crop_height],
                            [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def recognize(self, ori_im, box, device_id=None):
        """
        Recognize text in a specific bounding box of an image.

        Args:
            ori_im: Original image as numpy array
            box: Coordinates of the bounding box [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            device_id: For compatibility with original interface

        Returns:
            Recognized text string
        """
        if device_id is None:
            device_id = 0

        if ori_im is None or box is None:
            return ""

        # Generate cache key
        box_hash = hash(str(box))
        img_key = self._generate_cache_key(ori_im)
        if img_key:
            cache_key = f"recognize_{img_key}_{box_hash}"
            cached_result = self._cache_get(cache_key)
            if cached_result:
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug("Using cached recognition result")
                return cached_result

        # Convert box to numpy array if it's not already
        if not isinstance(box, np.ndarray):
            box = np.array(box, dtype=np.float32)

        # First, get a cropped image using the same cropping method as original OCR
        img_crop = self.get_rotate_crop_image(ori_im, box)

        # Convert to RGB for processing
        img_rgb = self._ensure_rgb(img_crop)

        # Recognize text in the cropped image
        if self.use_onnx and self.onnx_recognizer:
            # Use ONNX model for recognition
            text = self.onnx_recognizer.recognize(img_rgb)
            results = [(None, text, 1.0)] if text else []
        else:
            # Use PyTorch model from EasyOCR
            results = self.reader.readtext(img_rgb)

        # If no text was found, return empty string
        if not results:
            return ""

        # Combine all detected text with spaces
        # (There might be multiple text blocks in the cropped image)
        texts = []
        for _, text, confidence in results:
            if confidence >= self.drop_score:
                texts.append(text)

        combined_text = " ".join(texts)

        # Cache the result
        if img_key:
            self._cache_put(cache_key, combined_text)

        return combined_text

    def recognize_batch(self, img_list, device_id=None):
        """
        Recognize text in a batch of images.

        Args:
            img_list: List of images as numpy arrays (cropped text regions)
            device_id: For compatibility with original interface

        Returns:
            List of recognized text strings
        """
        if device_id is None:
            device_id = 0

        if not img_list:
            return []

        texts = []

        for img in img_list:
            if img is None:
                texts.append("")
                continue

            # Try to find in cache first
            cache_key = self._generate_cache_key(img)
            if cache_key:
                cached_result = self._cache_get(f"recognize_batch_{cache_key}")
                if cached_result:
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug("Using cached batch recognition result")
                    texts.append(cached_result)
                    continue

            # Convert to RGB for processing
            img_rgb = self._ensure_rgb(img)
            if img_rgb is None:
                texts.append("")
                continue

            # Recognize text
            if self.use_onnx and self.onnx_recognizer:
                # Use ONNX model for recognition
                try:
                    text = self.onnx_recognizer.recognize(img_rgb)
                    results = [(None, text, 1.0)] if text else []
                except Exception as e:
                    logging.error(f"Error with ONNX recognition: {e}")
                    results = []
            else:
                # Use PyTorch model from EasyOCR
                try:
                    results = self.reader.readtext(img_rgb)
                except Exception as e:
                    logging.error(f"Error with EasyOCR recognition: {e}")
                    results = []

            # Extract text with confidence above threshold
            img_texts = []
            for _, text, confidence in results:
                if confidence >= self.drop_score:
                    img_texts.append(text)

            # Join multiple detected texts with spaces or use empty string if none found
            text = " ".join(img_texts) if img_texts else ""

            # Cache the result
            if cache_key:
                self._cache_put(f"recognize_batch_{cache_key}", text)

            texts.append(text)

        return texts

    def __call__(self, img, device_id=0, cls=True):
        """
        Process an image with OCR to detect and recognize text.

        Args:
            img: Input image as numpy array (BGR format from OpenCV)
            device_id: For compatibility with original OCR interface
            cls: For compatibility with original OCR interface

        Returns:
            List of tuples (box_coords, (text, confidence)), time_dict with timing info
        """
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}

        if img is None:
            return None, None, time_dict

        # Generate a cache key for this image
        cache_key = self._generate_cache_key(img)
        if cache_key:
            cached_result = self._cache_get(f"call_{cache_key}")
            if cached_result:
                # Return cached result
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug("Using cached OCR results")
                formatted_results, timing = cached_result
                time_dict['all'] = timing
                time_dict['det'] = timing * 0.7  # Approximation
                time_dict['rec'] = timing * 0.3  # Approximation
                return formatted_results

        start = time.time()

        # Convert to RGB for processing
        img_rgb = self._ensure_rgb(img)
        if img_rgb is None:
            return None, None, time_dict

        if self.use_onnx and self.onnx_detector and self.onnx_recognizer:
            # Use ONNX models
            # Detection
            det_start = time.time()
            boxes = self.onnx_detector.detect(img_rgb)
            det_end = time.time()

            # Text recognition for each box
            rec_start = time.time()
            results = []
            for box in boxes:
                # Crop and recognize text
                img_crop = self.get_rotate_crop_image(img, box)
                img_rgb_crop = self._ensure_rgb(img_crop)
                text = self.onnx_recognizer.recognize(img_rgb_crop)
                confidence = 1.0  # ONNX models may not return confidence
                results.append((box, text, confidence))
            rec_end = time.time()

            time_dict['det'] = det_end - det_start
            time_dict['rec'] = rec_end - rec_start
        else:
            # Use standard EasyOCR
            results = self.reader.readtext(img_rgb)

        # Convert results to the format expected by the original OCR class
        formatted_results = []
        for box_coords, text, confidence in results:
            # Only include results with confidence above threshold
            if confidence >= self.drop_score:
                # Convert box_coords format to match original OCR:
                # EasyOCR returns [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                # which is the same format expected by the original OCR
                formatted_results.append((box_coords, (text, confidence)))

        end = time.time()
        timing = end - start
        time_dict['all'] = timing
        # Set approximate timings for compatibility if not explicitly defined
        if 'det' not in time_dict or time_dict['det'] == 0:
            time_dict['det'] = timing * 0.7  # Approximation
        if 'rec' not in time_dict or time_dict['rec'] == 0:
            time_dict['rec'] = timing * 0.3  # Approximation

        # Cache the result
        if cache_key:
            self._cache_put(f"call_{cache_key}", (formatted_results, timing))

        if not formatted_results:
            return None, None, time_dict

        return formatted_results

    def clear_cache(self):
        """Clear the internal cache to free memory"""
        with self._cache_lock:
            self._cache.clear()
            logging.info("EasyOCR cache cleared")

    def _convert_to_onnx_models(self, output_dir=None, upload_to_hf=False, hf_repo_id=None, hf_token=None):
        """
        Convert PyTorch models to ONNX format for better performance (internal method)

        Args:
            output_dir: Directory to save ONNX models
            upload_to_hf: Whether to upload models to Hugging Face
            hf_repo_id: Hugging Face repository ID
            hf_token: Hugging Face API token

        Returns:
            Path to output directory
        """
        if self.use_onnx:
            logging.warning("Already using ONNX models, conversion not needed")
            return None

        from .converter import convert_to_onnx

        # Get languages from reader
        languages = self.reader.lang_list

        # Convert models
        result = convert_to_onnx(
            languages=languages,
            output_dir=output_dir,
            use_gpu=True,
            upload_to_hf=upload_to_hf,
            hf_repo_id=hf_repo_id,
            hf_token=hf_token
        )

        # New version returns tuple with model names
        if isinstance(result, tuple) and len(result) == 3:
            output_path, detector_name, recognizer_name = result
            logging.info(f"Models converted and saved to {output_path}")
            logging.info(f"Detector: {detector_name}, Recognizer: {recognizer_name}")
        else:
            output_path = result
            logging.info(f"Models converted and saved to {output_path}")

        return output_path