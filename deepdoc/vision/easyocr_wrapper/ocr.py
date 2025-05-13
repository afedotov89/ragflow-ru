from api.utils.file_utils import get_project_base_directory
from huggingface_hub import snapshot_download

import cv2
import easyocr
# print(f"easyocr module loaded from: {easyocr.__file__}") # DEBUGGING LINE
import numpy as np
import torch
from skimage.measure import label

import hashlib
import logging
import os
import time

from .onnx_detector import ONNXDetector
from .onnx_recognizer import ONNXRecognizer

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
    det_path = os.path.join(model_dir, detector_name)

    if recognizer_name is None:
        rec_files = [f for f in os.listdir(model_dir)
                    if f.endswith('.onnx') and f != detector_name]
        if rec_files:
            recognizer_name = rec_files[0]
            logging.info(f"Auto-detected recognizer model: {recognizer_name}")
        else:
            # Try a common default if auto-detection fails, or raise error
            # This behavior depends on how robust you want it to be.
            # For now, let's assume a common pattern or raise.
            # Example default: recognizer_name = "cyrillic_g2.onnx" # Or based on expected languages
            raise FileNotFoundError(f"No recognizer model found in {model_dir}, and auto-detection failed.")

    rec_path = os.path.join(model_dir, recognizer_name)
    vocab_path = os.path.join(model_dir, "vocab.txt")

    if not os.path.exists(det_path):
        raise FileNotFoundError(f"Detector model not found: {det_path}")
    if not os.path.exists(rec_path):
        raise FileNotFoundError(f"Recognizer model not found: {rec_path}")

    detector = ONNXDetector(det_path, use_gpu=use_gpu)
    vocab_file = vocab_path if os.path.exists(vocab_path) else None
    recognizer = ONNXRecognizer(rec_path, vocab_path=vocab_file, use_gpu=use_gpu)

    return detector, recognizer

class EasyOCR:
    """
    OCR implementation using EasyOCR library to support multiple languages including Russian.
    Maintains interface compatibility with the original OCR class.
    """
    ONNX_REPO_ID = "afedotov/easyocr-onnx-models"

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
            # import easyocr # Already imported at top
            pass
        except ImportError:
            raise ImportError("EasyOCR is not installed. Please install it with pip install easyocr")

        if languages is None:
            languages = ['en', 'ru']

        gpu = use_gpu
        if use_gpu:
            try:
                # import torch # Already imported at top
                import platform
                if platform.system() == "Darwin" and platform.processor() == "arm":
                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        gpu = True
                        logging.info("Using Apple Silicon GPU via MPS")
                    else:
                        gpu = False
                        logging.warning("Apple Silicon GPU via MPS not available, falling back to CPU for EasyOCR Reader.")
                else:
                    gpu = torch.cuda.is_available()
                    if not gpu:
                        logging.warning("GPU requested for EasyOCR Reader but not available, falling back to CPU")
            except ImportError:
                gpu = False
                logging.warning("PyTorch not available, EasyOCR Reader will use CPU")

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
                logging.warning(f"Failed to create model directory: {download_dir}. Error: {e}. EasyOCR will use its default.")
                download_dir = None # Let EasyOCR handle its default path

        logging.info(f"Initializing EasyOCR.Reader with languages {languages}, GPU={gpu}, model_storage_directory='{download_dir if download_dir else 'EasyOCR default'}'")
        self.reader = easyocr.Reader(
            languages,
            gpu=gpu, # This GPU flag is for the PyTorch backend of EasyOCR.Reader
            model_storage_directory=download_dir,
            download_enabled=True,
            detector=True, # Ensure PyTorch detector model is loaded/downloaded by Reader
            recognizer=True # Ensure PyTorch recognizer model is loaded/downloaded by Reader
        )

        self.use_onnx = False
        self.onnx_detector = None
        self.onnx_recognizer = None

        # Determine default ONNX model names based on loaded PyTorch models by EasyOCR.Reader
        # This makes it more dynamic if EasyOCR changes its internal model naming/selection.
        try:
            detector_original_filename = os.path.basename(self.reader.detector_path)
            recognizer_original_filename = os.path.basename(self.reader.recognizer_path)

            # Default ONNX names derived from original .pth files
            default_detector_onnx = os.path.splitext(detector_original_filename)[0] + ".onnx"
            default_recognizer_onnx = os.path.splitext(recognizer_original_filename)[0] + ".onnx"

            logging.info(f"EasyOCR.Reader loaded PyTorch detector: {detector_original_filename} -> expecting ONNX: {default_detector_onnx}")
            logging.info(f"EasyOCR.Reader loaded PyTorch recognizer: {recognizer_original_filename} -> expecting ONNX: {default_recognizer_onnx}")

        except Exception as e:
            logging.warning(f"Could not determine ONNX model names from EasyOCR.Reader paths: {e}. Using fallback ONNX names.")
            default_detector_onnx = "craft_mlt_25k.onnx" # Fallback
            # Recognizer name can be language-dependent. Example for ru/en:
            if 'ru' in languages and ('en' in languages or len(languages) == 1):
                 default_recognizer_onnx = "cyrillic_g2.onnx" # Common for Russian or mixed Ru/En
            elif 'en' in languages:
                 default_recognizer_onnx = "english_g2.onnx" # Common for English
            else: # Fallback for other languages or if primary is not 'ru' or 'en'
                 default_recognizer_onnx = f"{languages[0]}_g2.onnx" # Generic guess
            logging.info(f"Fallback ONNX detector: {default_detector_onnx}, Fallback ONNX recognizer: {default_recognizer_onnx}")


        # --- ONNX Model Loading Logic ---
        # This 'gpu' flag here is for ONNX Runtime provider selection
        onnx_use_gpu = use_gpu # Separate variable to clarify its for ONNX, can be same as EasyOCR Reader's use_gpu

        try:
            import onnxruntime as ort # Already imported by onnx_detector/recognizer, but good for clarity here too
            from .converter import download_models_from_hf

            onnx_model_dir_base = download_dir if download_dir else os.path.join(get_project_base_directory(), "rag/res/easyocr")
            onnx_model_dir = os.path.join(onnx_model_dir_base, "onnx")
            os.makedirs(onnx_model_dir, exist_ok=True)

            det_path_local = os.path.join(onnx_model_dir, default_detector_onnx)
            rec_path_local = os.path.join(onnx_model_dir, default_recognizer_onnx)
            vocab_path_local = os.path.join(onnx_model_dir, "vocab.txt") # vocab.txt is standard

            models_exist_locally = os.path.exists(det_path_local) and \
                                   os.path.exists(rec_path_local) and \
                                   os.path.exists(vocab_path_local)

            if not models_exist_locally:
                logging.info(f"Expected ONNX models or vocab not found locally in {onnx_model_dir}. Attempting download from HF: {self.ONNX_REPO_ID}")
                try:
                    # download_models_from_hf downloads to a subdir of local_dir if repo has structure, or flat.
                    # Ensure it returns the effective directory.
                    effective_onnx_dir = download_models_from_hf(self.ONNX_REPO_ID, local_dir=onnx_model_dir)
                    # Re-check paths in the potentially new directory structure from HF download
                    det_path_local = os.path.join(effective_onnx_dir, default_detector_onnx)
                    rec_path_local = os.path.join(effective_onnx_dir, default_recognizer_onnx)
                    vocab_path_local = os.path.join(effective_onnx_dir, "vocab.txt") # Standard name

                    models_exist_locally = os.path.exists(det_path_local) and \
                                           os.path.exists(rec_path_local) and \
                                           os.path.exists(vocab_path_local)
                    if models_exist_locally:
                         logging.info(f"ONNX models and vocab found after download in {effective_onnx_dir}.")
                    else:
                         logging.warning(f"Expected ONNX models ({default_detector_onnx}, {default_recognizer_onnx}) or vocab.txt still not found after download attempt from {self.ONNX_REPO_ID} into {effective_onnx_dir}.")
                         # load_onnx_model will try auto-detection next if specific files aren't found.
                except Exception as e:
                    logging.error(f"Failed to download ONNX models from Hugging Face: {e}. Will proceed to attempt loading with any existing ONNX files or fail.")

            # Attempt to load ONNX models (either specific or auto-detected by load_onnx_model)
            try:
                # If specific models were found (locally or downloaded), load_onnx_model will use them.
                # If not, load_onnx_model's auto-detection for recognizer_name will kick in.
                self.onnx_detector, self.onnx_recognizer = load_onnx_model(
                    onnx_model_dir, # Directory to search for models
                    use_gpu=onnx_use_gpu, # GPU flag for ONNX Runtime
                    detector_name=default_detector_onnx, # Provide expected detector
                    recognizer_name=default_recognizer_onnx if os.path.exists(rec_path_local) else None # Provide expected recognizer if it exists, else let load_onnx_model auto-detect
                )
                self.use_onnx = True
                # To know exactly which models were loaded if auto-detection was used, ONNXDetector/Recognizer should log their input paths.
                logging.info(f"Successfully initialized ONNX models. Detector: {self.onnx_detector.input_name if self.onnx_detector else 'Failed'}, Recognizer: {self.onnx_recognizer.input_name if self.onnx_recognizer else 'Failed'}.")
            except FileNotFoundError as fnf_e:
                 logging.warning(f"ONNX FileNotFoundError during load_onnx_model: {fnf_e}. ONNX setup failed. Falling back to PyTorch.")
                 self.use_onnx = False
            except Exception as e_load:
                logging.error(f"Failed to load ONNX models: {e_load}. Trace: {e_load.with_traceback(None)}. ONNX setup failed. Falling back to PyTorch.")
                self.use_onnx = False

        except ImportError:
            logging.warning("ONNX Runtime not available. EasyOCR will use PyTorch models (slower).")
            self.use_onnx = False
        except Exception as e_outer_onnx:
            logging.error(f"Outer error during ONNX setup: {e_outer_onnx}. Falling back to PyTorch.")
            self.use_onnx = False

        if not self.use_onnx:
            logging.info("Using PyTorch backend for EasyOCR.")
        else:
            logging.info("Using ONNX backend for EasyOCR.")

        self.drop_score = 0.5
        from collections import OrderedDict
        from threading import RLock
        self._max_cache_size = max_cache_size
        self._cache_lock = RLock()
        self._cache = OrderedDict()
        self.crop_image_res_index = 0

    def _cache_put(self, key, value):
        with self._cache_lock:
            if len(self._cache) >= self._max_cache_size:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def _cache_get(self, key):
        with self._cache_lock:
            if key in self._cache:
                value = self._cache.pop(key)
                self._cache[key] = value
                return value
            return None

    def _generate_cache_key(self, img):
        if img is None: return None
        shape = img.shape
        sample_size = min(1000, img.size // 10)
        step = max(1, img.size // sample_size if sample_size > 0 else img.size + 1) # Avoid step=0 if img.size is small
        samples = img.ravel()[::step]
        m = hashlib.md5()
        m.update(str(shape).encode())
        m.update(samples.tobytes())
        return m.hexdigest()

    def _ensure_rgb(self, img):
        if img is None: return None
        if img.size == 0:
            logging.warning("Empty image provided to _ensure_rgb")
            return None # Or an empty image of expected type
        if len(img.shape) == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # If grayscale and PyTorch reader needs 3 channels, it usually handles it.
        # If grayscale and ONNX model needs 3 channels, it should be handled before ONNX model.
        # ONNXDetector handles replication if needed. ONNXRecognizer expects grayscale.
        return img # Return as is; specific converters handle channel needs.

    def sorted_boxes(self, dt_boxes):
        if not dt_boxes: # Handles None or empty list
            return []
        try:
            # Ensure all elements are list/array-like for consistent access
            # And that they have at least one point with two coordinates
            if not all(isinstance(b, (list, np.ndarray)) and len(b) > 0 and isinstance(b[0], (list, np.ndarray)) and len(b[0]) == 2 for b in dt_boxes):
                logging.warning(f"Inconsistent structure in dt_boxes for sorting. Attempting to filter valid boxes. Original: {dt_boxes}")
                # Filter out malformed boxes before sorting
                valid_boxes = []
                for b in dt_boxes:
                    if isinstance(b, (list, np.ndarray)) and len(b) > 0 and isinstance(b[0], (list, np.ndarray)) and len(b[0]) == 2:
                         # Further check if all 4 points exist if that's the expectation for x[0][1]
                         if len(b) == 4 and all(isinstance(p, (list, np.ndarray)) and len(p) == 2 for p in b):
                            valid_boxes.append(b)
                         elif len(b) > 0 : # if it's just one point, sorting key might still work if it's [[x,y]]
                            valid_boxes.append(b) # This might be too lenient for the key x[0][1]
                    else:
                        logging.debug(f"Skipping malformed box during sort preprocessing: {b}")
                dt_boxes = valid_boxes
                if not dt_boxes: return []

            _boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0])) # Initial sort
            num_boxes = len(_boxes)

            for i in range(num_boxes - 1):
                for j in range(i, -1, -1): # Iterate backwards from i
                    if j + 1 < num_boxes:
                        # Condition from original logic
                        if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                           (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                            # Swap
                            tmp = _boxes[j]
                            _boxes[j] = _boxes[j + 1]
                            _boxes[j + 1] = tmp
                        else:
                            # Inner loop condition not met, break to next i
                            break
            return _boxes # Ensure _boxes is returned after the loops complete
        except Exception as e:
            logging.error(f"Error during box sorting: {e}. Input boxes: {dt_boxes}")
            return dt_boxes # Return original on error to avoid crash

    def detect(self, img, device_id=None):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}
        if img is None:
            return zip([], []), time_dict

        cache_key = self._generate_cache_key(img)
        if cache_key:
            cached_result = self._cache_get(f"detect_{cache_key}")
            if cached_result:
                boxes, timing = cached_result
                time_dict['det'] = timing
                time_dict['all'] = timing
                logging.debug(f"Cache hit for detect: {cache_key}") # Restored debug
                return zip(boxes, [("", 0.0) for _ in range(len(boxes))]), time_dict

        start_time = time.time()
        img_processed = self._ensure_rgb(img) # Ensure RGB for PyTorch, ONNX detector handles its needs.
        if img_processed is None:
            return zip([], []), time_dict

        boxes = []
        if self.use_onnx and self.onnx_detector:
            boxes = self.onnx_detector.detect(img_processed) # ONNXDetector expects RGB
        else:
            try:
                # EasyOCR Reader's `detect` method has a complex return: (horizontal_list, free_list), score
                # Using `readtext` with detail=0 or extracting boxes from detail=1 is more straightforward
                # for getting a list of boxes.
                results_pytorch = self.reader.readtext(img_processed, detail=0, paragraph=False) # Returns list of boxes
                boxes = results_pytorch # Ensure this is a list of box coordinates
            except Exception as e:
                logging.error(f"PyTorch EasyOCR.Reader.readtext (for detection) failed: {e}")
                boxes = []

        detection_time = time.time() - start_time
        time_dict['det'] = detection_time
        time_dict['all'] = detection_time

        sorted_boxes_list = self.sorted_boxes(boxes if boxes else [])

        if cache_key:
            logging.debug(f"Caching result for detect: {cache_key}") # Restored debug
            self._cache_put(f"detect_{cache_key}", (sorted_boxes_list, detection_time))

        return zip(sorted_boxes_list, [("", 0.0) for _ in range(len(sorted_boxes_list))]), time_dict

    def get_rotate_crop_image(self, img, points):
        if img is None or points is None:
            raise ValueError("Image or points cannot be None for get_rotate_crop_image")

        if not isinstance(points, np.ndarray):
            points_arr = np.array(points, dtype=np.float32)
        else:
            points_arr = points.astype(np.float32)

        if points_arr.shape != (4, 2):
            if points_arr.size == 8: # Attempt to reshape if flat list of 8 coordinates
                try: points_arr = points_arr.reshape((4,2))
                except ValueError: raise ValueError(f"Points array size 8 but cannot reshape to (4,2). Original shape: {points_arr.shape}")
            else: raise ValueError(f"Shape of points must be 4x2. Got {points_arr.shape}")

        w_crop = int(max(np.linalg.norm(points_arr[0] - points_arr[1]), np.linalg.norm(points_arr[2] - points_arr[3])))
        h_crop = int(max(np.linalg.norm(points_arr[0] - points_arr[3]), np.linalg.norm(points_arr[1] - points_arr[2])))

        w_crop = max(1, w_crop) # Ensure at least 1 pixel
        h_crop = max(1, h_crop)

        pts_std = np.float32([[0,0],[w_crop,0],[w_crop,h_crop],[0,h_crop]])

        try:
            M = cv2.getPerspectiveTransform(points_arr, pts_std)
        except cv2.error as e: # Catch OpenCV specific errors
            logging.error(f"cv2.getPerspectiveTransform failed: {e}. Points: {points_arr}, Target: {pts_std}")
            # Return a minimal black image of the target size
            num_channels = img.shape[2] if len(img.shape) == 3 else 1
            return np.zeros((h_crop, w_crop, num_channels) if num_channels > 1 else (h_crop, w_crop), dtype=img.dtype)

        dst_img = cv2.warpPerspective(img, M, (w_crop, h_crop), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)

        dst_h, dst_w = dst_img.shape[0:2]
        if dst_h * 1.0 / max(1, dst_w) >= 1.5: # Avoid division by zero
            dst_img = np.rot90(dst_img)
        return dst_img

    def recognize(self, ori_im, box, device_id=None):
        if ori_im is None or box is None: return None

        img_key = self._generate_cache_key(ori_im)
        box_str = "_".join(map(str, np.array(box).astype(int).flatten())) # Consistent box string for key
        cache_key = None
        if img_key:
            cache_key = f"recognize_{img_key}_{box_str}"
            cached_val = self._cache_get(cache_key)
            if cached_val: return cached_val

        try:
            img_crop = self.get_rotate_crop_image(ori_im, box)
            logging.debug(f"Recognize: Cropped image shape for {box_str}: {img_crop.shape if img_crop is not None else 'None'}") # Restored debug
        except ValueError as e: # Catch error from get_rotate_crop_image
            logging.error(f"Failed to crop image for recognition: {e}")
            return None

        if img_crop is None or img_crop.size == 0:
            logging.warning("Cropped image for recognition is empty.")
            return None

        # _ensure_rgb for PyTorch. ONNXRecognizer handles its own grayscale conversion.
        # PyTorch Reader's recognizer also handles grayscale/RGB input.
        img_crop_processed = self._ensure_rgb(img_crop)

        text, confidence = "", 0.0
        if self.use_onnx and self.onnx_recognizer:
            text = self.onnx_recognizer.recognize(img_crop_processed) # ONNXRecognizer expects RGB/Grayscale, handles internally
            confidence = 1.0 if text else 0.0 # Basic confidence for ONNX
        else:
            try:
                # reader.recognize is complex; reader.readtext on crop is simpler
                pt_results = self.reader.readtext(img_crop_processed, detail=1, paragraph=False)
                if pt_results:
                    text, confidence = pt_results[0][1], pt_results[0][2] # Get text and conf from first result
            except Exception as e:
                logging.error(f"PyTorch EasyOCR.Reader.readtext (for recognition) failed: {e}")

        result_tuple = (text, confidence) if confidence >= self.drop_score else None

        if cache_key: self._cache_put(cache_key, result_tuple)
        return result_tuple

    def recognize_batch(self, img_list, device_id=None):
        if not img_list: return []

        batch_output = []
        for img_crop_item in img_list:
            if img_crop_item is None or img_crop_item.size == 0:
                logging.debug("Recognize_batch: Skipping None or empty crop item.") # Restored debug
                batch_output.append(None)
                continue

            # Individual caching for each crop in the batch context
            # `recognize` method has its own internal caching, but this is for `recognize_batch` context
            # if we were to bypass full `recognize` method for some reason.
            # For simplicity, let's call the single `recognize` method for each, which handles caching.
            # This means we don't need a separate cache layer here IF `recognize` is robust.
            # However, the original `recognize_batch` in `EasyOCR` class did not call `self.recognize`.
            # It had its own logic. Replicating that slightly:

            key_crop_batch = self._generate_cache_key(img_crop_item)
            cached_crop_res = None
            if key_crop_batch:
                cached_crop_res = self._cache_get(f"rec_batch_crop_{key_crop_batch}")
                if cached_crop_res: logging.debug(f"Recognize_batch: Cache hit for crop {key_crop_batch[:10]}...") # Restored debug

            if cached_crop_res:
                batch_output.append(cached_crop_res)
                continue

            # Processed for recognizer (RGB/Grayscale handled by downstream)
            img_crop_proc = self._ensure_rgb(img_crop_item)

            current_text, current_conf = "", 0.0
            if self.use_onnx and self.onnx_recognizer:
                current_text = self.onnx_recognizer.recognize(img_crop_proc)
                current_conf = 1.0 if current_text else 0.0
            else:
                try:
                    pt_rec_results = self.reader.readtext(img_crop_proc, detail=1, paragraph=False)
                    if pt_rec_results:
                        current_text, current_conf = pt_rec_results[0][1], pt_rec_results[0][2]
                except Exception as e_rec_batch:
                    logging.error(f"PyTorch recognition in batch failed for a crop: {e_rec_batch}")

            final_crop_result = (current_text, current_conf) if current_conf >= self.drop_score else None

            if key_crop_batch:
                logging.debug(f"Recognize_batch: Caching result for crop {key_crop_batch[:10]}...") # Restored debug
                self._cache_put(f"rec_batch_crop_{key_crop_batch}", final_crop_result)
            batch_output.append(final_crop_result)

        return batch_output

    def __call__(self, img, device_id=0, cls=True): # cls is unused
        time_dict = {'det': 0, 'rec': 0, 'all': 0}
        if img is None:
            return None, None, time_dict # Maintain (results, time_dict) structure for return

        full_cache_key = self._generate_cache_key(img)
        if full_cache_key:
            cached_data = self._cache_get(f"call_{full_cache_key}")
            if cached_data:
                # Expect cached_data to be (list_of_results, all_time_float)
                res_list, all_t = cached_data
                time_dict['all'] = all_t
                # Approximate det/rec times if not stored, or if complex to store them from cache
                time_dict['det'] = all_t * 0.5 # Generic approximation
                time_dict['rec'] = all_t * 0.5
                logging.debug(f"Full __call__ cache hit for {full_cache_key}") # Restored debug
                return res_list, time_dict


        overall_start_time = time.time()

        # Detection part
        # `detect` method returns (zip_iterator, time_dict_det)
        # We need the boxes from the zip_iterator.
        det_results_iter, time_dict_det = self.detect(img) # device_id passed for API compat, not used by self.detect
        time_dict['det'] = time_dict_det['det'] # Get actual detection time

        detected_boxes = [item[0] for item in det_results_iter] # Extract boxes

        if not detected_boxes:
            time_dict['all'] = time.time() - overall_start_time
            return None, None, time_dict

        # Recognition part (on sorted boxes from `detect` which internally sorts)
        # `detect` method already sorted the boxes.
        ocr_final_results = []
        recognition_start_time = time.time()

        # Pass original BGR image to `recognize` as it handles `_ensure_rgb` and `get_rotate_crop_image`
        for box_item in detected_boxes: # `detected_boxes` from `self.detect` are already sorted
            rec_tuple = self.recognize(img, box_item) # `img` is original BGR
            if rec_tuple: # If (text, confidence) is not None
                # Defensive check for the structure of rec_tuple
                if not isinstance(rec_tuple, tuple) or len(rec_tuple) != 2:
                    logging.error(f"EasyOCR.__call__: UNEXPECTED structure for recognition_tuple: {rec_tuple}, type: {type(rec_tuple)}. Box: {box_item}. Skipping this result.")
                    continue # Skip this problematic result

                # Further check element types if desired, e.g., text is str, confidence is float
                # For now, the main check is that it's a 2-element tuple.

                ocr_final_results.append((box_item, rec_tuple))

        time_dict['rec'] = time.time() - recognition_start_time
        time_dict['all'] = time.time() - overall_start_time
        logging.debug(f"__call__ execution times: Det: {time_dict['det']:.4f}s, Rec: {time_dict['rec']:.4f}s, All: {time_dict['all']:.4f}s") # Restored debug

        if full_cache_key:
            self._cache_put(f"call_{full_cache_key}", (ocr_final_results, time_dict['all']))
            logging.debug(f"Full __call__ result cached for {full_cache_key}") # Restored debug

        if not ocr_final_results:
            return None, None, time_dict

        return ocr_final_results, time_dict


    def clear_cache(self):
        with self._cache_lock:
            self._cache.clear()
            logging.info("EasyOCR cache cleared")

    def _convert_to_onnx_models(self, output_dir=None, upload_to_hf=False, hf_repo_id=None, hf_token=None):
        if self.use_onnx:
            logging.warning("Already configured to use ONNX, or ONNX conversion was already attempted. Conversion not re-initiated.")
            return None # Or path to existing ONNX dir if known

        from .converter import convert_to_onnx
        languages_for_conversion = self.reader.lang_list

        # Determine GPU for conversion process (can be different from runtime use_gpu)
        conversion_use_gpu = True # Or make this a parameter

        # Default output directory for conversion if not provided
        if output_dir is None:
            try:
                proj_base = get_project_base_directory()
                output_dir = os.path.join(proj_base, "rag/res/easyocr/onnx_converted") # Specific subdir for converted
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e_conv_dir:
                logging.error(f"Could not create default ONNX conversion output_dir: {e_conv_dir}. Conversion may fail or use current dir.")
                output_dir = "." # Fallback to current directory

        logging.info(f"Starting ONNX model conversion for languages: {languages_for_conversion}. Output to: {output_dir}")

        try:
            # Assuming convert_to_onnx returns: output_dir, detector_name, recognizer_name
            converted_output_path, det_name, rec_name = convert_to_onnx(
                languages=languages_for_conversion,
            output_dir=output_dir,
                use_gpu=conversion_use_gpu, # For conversion process
            upload_to_hf=upload_to_hf,
            hf_repo_id=hf_repo_id,
            hf_token=hf_token
        )
            logging.info(f"ONNX Models converted: Detector='{det_name}', Recognizer='{rec_name}'. Saved in '{converted_output_path}'.")
            # Optionally, re-initialize this EasyOCR instance to use these newly converted models
            # For example, by setting self.use_onnx = True and loading them.
            # This would require knowing the exact names and having a robust loading mechanism.
            # Or, inform user to re-initialize.
            logging.info("Please re-initialize EasyOCR instance to use newly converted ONNX models from the specified output directory.")
            return converted_output_path
        except Exception as e_conversion:
            logging.error(f"ONNX model conversion failed: {e_conversion}")
            return None

# Potential alias for backward compatibility if other parts of the project import OCR from here.
# OCR = EasyOCR
# This should be handled by the importing module to decide which OCR class to use.