#
# Based on work from InfiniFlow project
# This file was created by Alexander Fedotov as an extension to the original project
# Licensed under the Apache License, Version 2.0
#

import cv2
import numpy as np
import logging
from skimage.measure import label
import onnxruntime as ort # Keep top-level for clarity, though original had it in __init__

class ONNXDetector:
    """
    ONNX Implementation of EasyOCR detector
    """
    def __init__(self, model_path, use_gpu=True):
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
                logging.info("Using CUDAExecutionProvider for ONNX detector.")
            # Temporarily disable CoreML to default to CPU if CUDA is not available
            # elif 'CoreMLExecutionProvider' in ort.get_available_providers():
            # providers.append('CoreMLExecutionProvider')
            # logging.info("Using CoreMLExecutionProvider for ONNX detector.")
            else:
                logging.warning("CUDAExecutionProvider not available. Falling back to CPUExecutionProvider for ONNX detector.")
                providers.append('CPUExecutionProvider')
        else:
            providers.append('CPUExecutionProvider')
            logging.info("Using CPUExecutionProvider for ONNX detector (GPU explicitly disabled).")

        # Load the model
        self.session = ort.InferenceSession(model_path, options=options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        # Define input and output shapes
        logging.info(f"Detector input shape: {self.input_shape}")
        self.model_input_height = 640
        self.model_input_width = 640

        # Configuration for detection based on EasyOCR defaults
        # EXPERIMENTING with thresholds again due to model output characteristics
        self.min_size = 10         # From easyocr.utils.getDetBoxes_core (default for min_size argument)
        self.text_threshold = 0.65 # EXPERIMENTAL: Default 0.7, slightly lowered with dilation
        self.low_text = 0.6        # Reverting to a previously better value for initial segmentation
        self.link_threshold = 0.55 # Reverting to a previously better value for initial segmentation

    def detect(self, image):
        """
        Detect text regions in an image

        Args:
            image: Input image as numpy array (RGB)

        Returns:
            List of bounding boxes in format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        # logging.error("############################################################")
        # logging.error("####### CALLING ONNXDetector.detect METHOD #######")
        # logging.error("############################################################")

        # Prepare image for detector
        h, w = image.shape[:2]

        # Scale image to model input shape (using determined model_input_height/width)
        # Note: target_height/width are for resizing the input to the model.
        # The ratios for scaling output boxes back should be based on the actual feature map dimensions.
        target_height, target_width = self.model_input_height, self.model_input_width
        # ratio_h = h / target_height if target_height > 0 else 1 # Old ratio calculation
        # ratio_w = w / target_width if target_width > 0 else 1 # Old ratio calculation

        img_resized = cv2.resize(image, (target_width, target_height))

        # Normalize image
        img_norm = img_resized.astype(np.float32) / 255.0

        # ImageNet Normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # Ensure img_norm has 3 channels for subtraction/division if it's not grayscale
        if img_norm.ndim == 3 and img_norm.shape[2] == 3:
            img_norm = (img_norm - mean) / std
        elif img_norm.ndim == 2: # Grayscale image, needs different handling or check if model supports it
            # For now, let's assume the model expects 3 channels if mean/std are for 3 channels.
            # This case might need specific model or handling if grayscale is intended.
            logging.warning("ONNXDetector: Applying ImageNet normalization to a 2D image by replicating channels. This might be incorrect if model expects grayscale.")
            img_norm = np.stack((img_norm,)*3, axis=-1) # Convert to 3-channel grayscale
            img_norm = (img_norm - mean) / std
        else:
            logging.warning(f"ONNXDetector: ImageNet normalization not applied. Unexpected image dimensions: {img_norm.shape}")

        img_norm = img_norm.transpose(2, 0, 1)  # HWC -> CHW
        img_norm = np.expand_dims(img_norm, axis=0)  # Add batch dimension

        # Run inference
        outputs = self.session.run(None, {self.input_name: img_norm})

        # Process model output
        # Specific logic for processing detector outputs
        boxes = []
        # outputs[0] is expected to be (batch_size, height, width, num_channels) or (batch_size, num_channels, height, width)
        # For CRAFT, num_channels is 2 (region score, affinity score)
        # We need the region score map. Let's inspect the shape of outputs[0].
        raw_output = outputs[0] # Full output for the first (and only) batch item.
        logging.debug(f"ONNXDetector: raw_output shape: {raw_output.shape}")

        # Determine if channels are last or second dimension.
        # Common CRAFT output shapes: (1, H, W, 2) or (1, 2, H, W)
        # We'll take the first element of the batch.
        output_tensor_for_image = raw_output[0] # Shape (H,W,C) or (C,H,W)

        region_score_map = None
        link_score_map = None

        if output_tensor_for_image.ndim == 3:
            if output_tensor_for_image.shape[0] == 2: # Shape (C, H, W) -> (2, H, W)
                region_score_map = output_tensor_for_image[0, :, :]
                link_score_map = output_tensor_for_image[1, :, :]
                logging.debug(f"Extracted region_score_map (shape {region_score_map.shape}) and link_score_map (shape {link_score_map.shape}) from (C,H,W) output.")
            elif output_tensor_for_image.shape[2] == 2: # Shape (H, W, C) -> (H, W, 2)
                region_score_map = output_tensor_for_image[:, :, 0]
                link_score_map = output_tensor_for_image[:, :, 1]
                logging.debug(f"Extracted region_score_map (shape {region_score_map.shape}) and link_score_map (shape {link_score_map.shape}) from (H,W,C) output.")
            else:
                logging.warning(f"Unexpected 3D output shape for detector: {output_tensor_for_image.shape}. Cannot separate region and link maps.")
                return []
        elif output_tensor_for_image.ndim == 2: # Assuming this is only region_score_map if it's 2D
            region_score_map = output_tensor_for_image
            logging.warning(f"Output tensor is 2D (shape {region_score_map.shape}). Assuming it's region_score_map. Link map processing will be skipped.")
            # link_score_map will remain None
        else:
            logging.error(f"CRITICAL_ERROR_UNEXPECTED_DIM: Unexpected output dimension: {output_tensor_for_image.ndim}, shape {output_tensor_for_image.shape}.")
            return []

        # Apply sigmoid to convert logits to probabilities if they aren't already
        for map_name, score_map_ref in [("region_score_map", "region_score_map"), ("link_score_map", "link_score_map")]:
            current_map = locals()[score_map_ref]
            if current_map is not None and current_map.size > 0:
                logging.info(f"ONNXDetector: Applying sigmoid to {map_name}.")
                current_map_processed = 1 / (1 + np.exp(-current_map))
                if score_map_ref == "region_score_map":
                    region_score_map = current_map_processed
                elif score_map_ref == "link_score_map":
                    link_score_map = current_map_processed

            elif current_map is None and map_name == "link_score_map":
                logging.info("ONNXDetector: link_score_map is None, skipping sigmoid for it.")
            else: # Should not happen for region_score_map if previous checks passed
                logging.warning(f"ONNXDetector: {map_name} is None or empty before sigmoid check.")
                if map_name == "region_score_map": return [] # Critical if region map is bad

        # Log statistics AFTER potential sigmoid
        if region_score_map is not None and region_score_map.size > 0:
            logging.info(f"Region map stats: Min={region_score_map.min():.2f}, Max={region_score_map.max():.2f}, Mean={region_score_map.mean():.2f}") # More concise info
        else:
            logging.error("CRITICAL_ERROR_REGION_MAP_EMPTY: region_score_map is None or empty before binarization.")
            return []

        if link_score_map is not None and link_score_map.size > 0:
            logging.info(f"Link map stats: Min={link_score_map.min():.2f}, Max={link_score_map.max():.2f}, Mean={link_score_map.mean():.2f}") # More concise info
        elif link_score_map is None:
             logging.info("LINK_MAP_STATS: link_score_map is None. Proceeding without it.")
        else: # link_score_map exists but is empty
            logging.warning("LINK_MAP_WARNING: link_score_map is empty. Proceeding without it for combination.")

        # ---- Corrected ratio calculation ----
        # h, w are original image dimensions
        # region_score_map.shape gives the dimensions of the map from which coordinates are derived.
        map_h, map_w = region_score_map.shape
        logging.debug(f"ONNXDetector: Original image dims: H={h}, W={w}")
        logging.debug(f"ONNXDetector: Feature map dims for ratio: map_H={map_h}, map_W={map_w}")

        ratio_h = h / map_h if map_h > 0 else 1.0
        ratio_w = w / map_w if map_w > 0 else 1.0
        logging.info(f"ONNXDetector: Calculated coordinate scaling ratios: ratio_H={ratio_h:.4f}, ratio_W={ratio_w:.4f}")
        # ---- End Corrected ratio calculation ----

        # --- Logic based on EasyOCR's craft_utils.getDetBoxes / utils.getDetBoxes_core ---

        # 1. Binarize text map (region_score_map) by low_text (for initial segmentation)
        logging.debug(f"ONNXDetector: Binarizing region_score_map with low_text_threshold = {self.low_text} for segmap")
        binary_text_map_for_seg = region_score_map > self.low_text
        logging.debug(f"ONNXDetector: binary_text_map_for_seg non-zero count: {np.count_nonzero(binary_text_map_for_seg)}")

        # 2. Binarize link map (link_score_map) by link_threshold (if available)
        if link_score_map is not None and link_score_map.shape == region_score_map.shape:
            logging.debug(f"ONNXDetector: Binarizing link_score_map with link_threshold = {self.link_threshold}")
            binary_link_map = link_score_map > self.link_threshold
            logging.debug(f"ONNXDetector: binary_link_map non-zero count: {np.count_nonzero(binary_link_map)}")
            # Combine the binarized maps to create segmap
            logging.debug("ONNXDetector: Creating segmap from binarized region_map and binarized link_map.")
            segmap = np.clip(binary_text_map_for_seg.astype(np.float32) + binary_link_map.astype(np.float32), 0, 1).astype(np.uint8)
        else:
            logging.warning("ONNXDetector: link_score_map not available or mismatched shape. Using only binarized region_score_map for segmap.")
            segmap = binary_text_map_for_seg.astype(np.uint8)

        logging.debug(f"ONNXDetector: segmap non-zero count: {np.count_nonzero(segmap)}")

        # Optional: Dilate segmap (as done in EasyOCR before finding contours)
        kernel = np.ones((3,3), np.uint8)
        segmap_dilated = cv2.dilate(segmap, kernel, iterations=1) # iter_num=1 is common in EasyOCR
        logging.debug(f"ONNXDetector: Applied dilation to segmap. Dilated segmap non-zero count: {np.count_nonzero(segmap_dilated)}")

        # 3. Find connected components (labels) on this segmap
        # skimage.measure.label expects an integer array. Background is 0.
        labels, num_labels = label(segmap_dilated, connectivity=2, return_num=True) # Using DILATED segmap
        # labels, num_labels = label(segmap, connectivity=2, return_num=True) # Using non-dilated segmap for now
        logging.debug(f"ONNXDetector: Found {num_labels} components on segmap before confidence filtering.")

        boxes = []
        # Iterate through components found by skimage.measure.label
        # For skimage, labels run from 1 to num_labels. label 0 is background.
        for i in range(1, num_labels + 1):
            component_mask = (labels == i)
            # Calculate area (size)
            size = np.sum(component_mask)

            # 4a. Size filtering (same as before, using self.min_size)
            if size < self.min_size:
                logging.debug(f"ONNXDetector: Skipping component {i} due to small size: {size} < {self.min_size}")
                continue

            # 4b. Confidence filtering: Check MAX score in original region_score_map for this component against text_threshold
            # This uses the original region_score_map (after sigmoid), and self.text_threshold (0.7)
            # This aligns with EasyOCR's getDetBoxes_core which filters by: if np.max(textmap_original[labels==k]) < text_threshold
            max_text_val_in_component = np.max(region_score_map[component_mask])
            if max_text_val_in_component < self.text_threshold:
                logging.debug(f"ONNXDetector: Skipping component {i} due to low MAX text value in region_score_map: {max_text_val_in_component:.4f} < {self.text_threshold}")
                continue

            logging.debug(f"ONNXDetector: Component {i} PASSED filters. Size: {size}, MAX Text Val in region_score_map: {max_text_val_in_component:.4f}")

            # Get bounding box for this component using min/max row/col (axis-aligned)
            rows, cols = np.where(component_mask)
            min_row, max_row = np.min(rows), np.max(rows)
            min_col, max_col = np.min(cols), np.max(cols)

            # Scaled coordinates using the correctly defined ratio_w and ratio_h
            s_min_col = min_col * ratio_w
            s_max_col = max_col * ratio_w
            s_min_row = min_row * ratio_h
            s_max_row = max_row * ratio_h

            # Box points in [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] format
            # (top-left, top-right, bottom-right, bottom-left)
            current_box = [[s_min_col, s_min_row], [s_max_col, s_min_row], [s_max_col, s_max_row], [s_min_col, s_max_row]]
            boxes.append(current_box)

        logging.info(f"ONNXDetector: Detected {len(boxes)} boxes after all filters.")
        # Log the actual boxes for detailed inspection (using ERROR level as requested)
        # if boxes:
        #     logging.error(f"ONNXDetector: Final {len(boxes)} detected boxes:")
        #     for idx, box in enumerate(boxes):
        #         logging.error(f"ONNXDetector: Box {idx}: {box}")
        # else:
        #     logging.error("ONNXDetector: No boxes detected after all filters.")

        return boxes