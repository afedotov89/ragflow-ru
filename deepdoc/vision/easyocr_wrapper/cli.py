#
# Based on work from InfiniFlow project
# This file was created by Alexander Fedotov as an extension to the original project
# Licensed under the Apache License, Version 2.0
#

import argparse
import os
import time
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from api.utils.file_utils import get_project_base_directory

from deepdoc.vision.easyocr_wrapper.ocr import EasyOCR
from deepdoc.vision.easyocr_wrapper.converter import convert_to_onnx, download_models_from_hf

def setup_logging(level=logging.INFO):
    """Set up logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def draw_boxes(image, boxes_info, output_path=None):
    """
    Draw bounding boxes on the image

    Args:
        image: Input image
        boxes_info: List of (box, (text, confidence)) tuples
        output_path: Path to save the image with boxes
    """
    img_boxes = image.copy()

    for box, (text, confidence) in boxes_info:
        box = np.array(box, dtype=np.int32)
        cv2.polylines(img_boxes, [box], True, (0, 255, 0), 2)

        x, y = box[0]
        cv2.putText(img_boxes, f"{text} ({confidence:.2f})", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(output_path, img_boxes)
        logging.info(f"Image with boxes saved to {output_path}")
    else:
        plt.figure(figsize=(15, 15))
        plt.imshow(cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB))
        plt.title("OCR Results")
        plt.axis("off")
        plt.show()

def benchmark(ocr, image_paths, repeat=5):
    """
    Benchmark OCR performance

    Args:
        ocr: OCR instance
        image_paths: List of image paths
        repeat: Number of times to repeat the benchmark
    """
    total_time = 0
    total_images = 0

    for image_path in image_paths:
        logging.info(f"Processing image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f"Failed to load image: {image_path}")
            continue

        times = []
        for i in range(repeat):
            start_time = time.time()
            results = ocr(img)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        logging.info(f"Image: {os.path.basename(image_path)}")
        logging.info(f"Average time: {avg_time:.4f} seconds")
        logging.info(f"Min time: {min(times):.4f} seconds")
        logging.info(f"Max time: {max(times):.4f} seconds")

        total_time += avg_time
        total_images += 1

        if repeat > 1:
            avg_inference_time = sum(times[1:]) / len(times[1:])
            logging.info(f"Average inference time (excluding first run): {avg_inference_time:.4f} seconds")

    if total_images > 0:
        logging.info(f"Overall average time per image: {total_time / total_images:.4f} seconds")
    else:
        logging.warning("No images were processed successfully")

def main():
    parser = argparse.ArgumentParser(description="EasyOCR Tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert EasyOCR models to ONNX")
    convert_parser.add_argument("--languages", nargs="+", default=["en", "ru"],
                               help="Languages to include in the model")
    convert_parser.add_argument("--output-dir", help="Directory to save ONNX models")
    convert_parser.add_argument("--use-gpu", action="store_true", help="Use GPU for conversion")
    convert_parser.add_argument("--upload-to-hf", action="store_true",
                              help="Upload models to Hugging Face")
    convert_parser.add_argument("--hf-repo-id", help="Hugging Face repository ID")
    convert_parser.add_argument("--hf-token", help="Hugging Face API token")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download ONNX models from Hugging Face")
    download_parser.add_argument("--repo-id", required=True, help="Hugging Face repository ID")
    download_parser.add_argument("--output-dir", help="Directory to save models")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test OCR on images")
    test_parser.add_argument("--image", required=True, help="Path to image")
    test_parser.add_argument("--output", help="Path to save output image")
    test_parser.add_argument("--languages", nargs="+", default=["en", "ru"],
                            help="Languages to use")
    test_parser.add_argument("--use-gpu", action="store_true", help="Use GPU for inference")
    test_parser.add_argument("--use-onnx", action="store_true", help="Use ONNX models")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark OCR performance")
    benchmark_parser.add_argument("--images-dir", required=True, help="Directory containing images")
    benchmark_parser.add_argument("--repeat", type=int, default=5,
                                help="Number of times to repeat the benchmark")
    benchmark_parser.add_argument("--languages", nargs="+", default=["en", "ru"],
                                help="Languages to use")
    benchmark_parser.add_argument("--use-gpu", action="store_true", help="Use GPU for inference")
    benchmark_parser.add_argument("--use-onnx", action="store_true", help="Use ONNX models")

    args = parser.parse_args()
    setup_logging()

    if args.command == "convert":
        output_dir = convert_to_onnx(
            languages=args.languages,
            output_dir=args.output_dir,
            use_gpu=args.use_gpu,
            upload_to_hf=args.upload_to_hf,
            hf_repo_id=args.hf_repo_id,
            hf_token=args.hf_token
        )
        logging.info(f"Models converted and saved to {output_dir}")

    elif args.command == "download":
        output_dir = download_models_from_hf(
            repo_id=args.repo_id,
            local_dir=args.output_dir
        )
        logging.info(f"Models downloaded to {output_dir}")

    elif args.command == "test":
        ocr = EasyOCR(
            languages=args.languages,
            use_gpu=args.use_gpu,
            use_onnx=args.use_onnx
        )

        img = cv2.imread(args.image)
        if img is None:
            logging.error(f"Failed to load image: {args.image}")
            return

        start_time = time.time()
        results = ocr(img)
        end_time = time.time()

        logging.info(f"OCR completed in {end_time - start_time:.4f} seconds")
        logging.info(f"Found {len(results)} text regions:")

        for i, (box, (text, confidence)) in enumerate(results):
            logging.info(f"{i+1}. Text: {text}, Confidence: {confidence:.4f}, Box: {box}")

        draw_boxes(img, results, args.output)

    elif args.command == "benchmark":
        ocr = EasyOCR(
            languages=args.languages,
            use_gpu=args.use_gpu,
            use_onnx=args.use_onnx
        )

        image_paths = []
        for root, _, files in os.walk(args.images_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(root, file))

        if not image_paths:
            logging.error(f"No images found in {args.images_dir}")
            return

        logging.info(f"Found {len(image_paths)} images")
        benchmark(ocr, image_paths, args.repeat)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()