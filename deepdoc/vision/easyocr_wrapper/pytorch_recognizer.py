#
# Based on work from InfiniFlow project
# This file was created by Alexander Fedotov as an extension to the original project
# Licensed under the Apache License, Version 2.0
#

import logging
import easyocr # For type hinting reader, if necessary

class PyTorchRecognizer:
    def __init__(self, reader: easyocr.Reader):
        self.reader = reader
        logging.info("Initialized PyTorchRecognizer.")

    def recognize(self, img_crop_processed_rgb):
        """
        Recognizes text in a cropped image using the PyTorch backend of easyocr.Reader.
        Args:
            img_crop_processed_rgb: Cropped RGB image (numpy array) preprocessed for recognition.
        Returns:
            Tuple (text, confidence) or None if recognition fails or confidence is too low.
            The drop_score logic is handled by the caller (EasyOCR class).
        """
        text, confidence = "", 0.0
        if img_crop_processed_rgb is None or img_crop_processed_rgb.size == 0:
            logging.warning("PyTorchRecognizer.recognize: Input image crop is None or empty.")
            return "", 0.0

        try:
            pt_results = self.reader.readtext(img_crop_processed_rgb, detail=1, paragraph=False)
            if pt_results:
                text, confidence = pt_results[0][1], pt_results[0][2]
                logging.debug(f"PyTorchRecognizer.recognize: Text='{text}', Confidence={confidence:.4f}")
            else:
                logging.debug("PyTorchRecognizer.recognize: No text recognized by reader.readtext.")
        except Exception as e:
            logging.error(f"PyTorchRecognizer.recognize failed: {e}", exc_info=True)

        return text, confidence