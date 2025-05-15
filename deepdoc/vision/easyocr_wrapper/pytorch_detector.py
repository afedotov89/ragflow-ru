#
# Based on work from InfiniFlow project
# This file was created by Alexander Fedotov as an extension to the original project
# Licensed under the Apache License, Version 2.0
#

import logging
import numpy as np
import easyocr # For type hinting reader, if necessary

class PyTorchDetector:
    def __init__(self, reader: easyocr.Reader):
        self.reader = reader
        logging.info("Initialized PyTorchDetector.")

    def detect(self, img_processed_rgb):
        """
        Detects text in an image using the PyTorch backend of easyocr.Reader.
        Args:
            img_processed_rgb: RGB image (numpy array) preprocessed for detection.
        Returns:
            List of bounding boxes, where each box is a list of 4 points [[x,y], [x,y], [x,y], [x,y]].
        """
        boxes = []
        try:
            returned_horizontal_list, returned_free_list = self.reader.detect(img_processed_rgb, optimal_num_chars=None)

            logging.debug(f"PyTorchDetector.detect - returned_horizontal_list (count: {len(returned_horizontal_list if returned_horizontal_list else [])}): {returned_horizontal_list}")
            logging.debug(f"PyTorchDetector.detect - returned_free_list (count: {len(returned_free_list if returned_free_list else [])}): {returned_free_list}")

            actual_horizontal_boxes = []
            if returned_horizontal_list and isinstance(returned_horizontal_list, list) and len(returned_horizontal_list) > 0:
                if isinstance(returned_horizontal_list[0], list):
                    actual_horizontal_boxes = returned_horizontal_list[0]
                else:
                    logging.warning(f"PyTorchDetector: Unexpected structure in returned_horizontal_list: {returned_horizontal_list}. Attempting to use as is if list of lists.")
                    if isinstance(returned_horizontal_list, list) and all(isinstance(i, list) for i in returned_horizontal_list):
                         actual_horizontal_boxes = returned_horizontal_list

            actual_free_boxes = []
            if returned_free_list and isinstance(returned_free_list, list) and len(returned_free_list) > 0:
                if isinstance(returned_free_list[0], list):
                    actual_free_boxes = returned_free_list[0]
                else:
                    logging.warning(f"PyTorchDetector: Unexpected structure in returned_free_list: {returned_free_list}. Attempting to use as is if list of lists.")
                    if isinstance(returned_free_list, list) and all(isinstance(i, list) for i in returned_free_list):
                        actual_free_boxes = returned_free_list

            processed_boxes = []
            if actual_horizontal_boxes:
                for i, hbox in enumerate(actual_horizontal_boxes):
                    if isinstance(hbox, (list, np.ndarray)) and len(hbox) == 4:
                        try:
                            x_min, x_max, y_min, y_max = map(int, hbox)
                            four_points = [
                                [x_min, y_min], [x_max, y_min],
                                [x_max, y_max], [x_min, y_max]
                            ]
                            processed_boxes.append(four_points)
                        except (ValueError, TypeError) as ve_unpack:
                            logging.error(f"PyTorchDetector.detect - Error processing horizontal_box {hbox}: {ve_unpack}. Skipping.")
                            continue
                    else:
                        logging.warning(f"PyTorchDetector.detect - Invalid horizontal_box format: {hbox}. Skipping.")

            if actual_free_boxes:
                for i, fbox in enumerate(actual_free_boxes):
                    if isinstance(fbox, (list, np.ndarray)) and len(fbox) == 4 and \
                       all(isinstance(pt, (list, np.ndarray, tuple)) and len(pt) == 2 for pt in fbox):
                        try:
                            int_fbox = [[int(p[0]), int(p[1])] for p in fbox]
                            processed_boxes.append(int_fbox)
                        except (ValueError, TypeError) as e_coord:
                            logging.error(f"PyTorchDetector.detect - Error converting coordinates for free_box {fbox}: {e_coord}. Skipping.")
                            continue
                    else:
                        logging.warning(f"PyTorchDetector.detect - Invalid free_box format: {fbox}. Skipping.")

            boxes = processed_boxes
        except Exception as e:
            logging.error(f"PyTorchDetector.detect failed: {e}", exc_info=True)
            boxes = []
        return boxes