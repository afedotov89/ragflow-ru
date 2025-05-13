#
# Based on work from InfiniFlow project
# This file was created by Alexander Fedotov as an extension to the original project
# Licensed under the Apache License, Version 2.0
#

from .ocr import EasyOCR
from .converter import convert_to_onnx

__all__ = ['EasyOCR', 'convert_to_onnx']