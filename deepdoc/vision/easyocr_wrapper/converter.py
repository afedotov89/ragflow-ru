#
# Based on work from InfiniFlow project
# This file was created by Alexander Fedotov as an extension to the original project
# Licensed under the Apache License, Version 2.0
#

import os
import logging
import time
from huggingface_hub import snapshot_download, upload_file

def convert_to_onnx(languages=['en', 'ru'], output_dir=None, use_gpu=True, upload_to_hf=False, hf_repo_id=None, hf_token=None):
    """
    Convert EasyOCR models to ONNX format

    Args:
        languages: List of languages
        output_dir: Directory to save ONNX models
        use_gpu: Whether to use GPU for conversion
        upload_to_hf: Whether to upload models to Hugging Face
        hf_repo_id: Hugging Face repository ID
        hf_token: Hugging Face API token

    Returns:
        Tuple of (output_dir, detector_onnx_name, recognizer_onnx_name)
    """
    try:
        import torch
        import easyocr
    except ImportError:
        raise ImportError("PyTorch and EasyOCR must be installed for conversion")

    if output_dir is None:
        # Attempt to import dynamically to avoid circular dependency if this file is imported elsewhere
        try:
            from api.utils.file_utils import get_project_base_directory
            output_dir = os.path.join(get_project_base_directory(), "rag/res/easyocr/onnx")
        except ImportError:
            # Fallback if the above import fails (e.g. running standalone)
            logging.warning("Could not import get_project_base_directory, using default relative path for ONNX models.")
            output_dir = os.path.join("rag/res/easyocr/onnx")


    os.makedirs(output_dir, exist_ok=True)

    # Initialize EasyOCR
    # It's important that model_storage_directory points to where EasyOCR can find its .pth models
    # For conversion, we usually rely on EasyOCR downloading them if not present.
    reader = easyocr.Reader(languages, gpu=use_gpu, download_enabled=True)

    # Get original model names from the reader's attributes
    # These paths are where EasyOCR stores/loaded its .pth models from
    detector_name = os.path.basename(reader.detector_path)
    recognizer_name = os.path.basename(reader.recognizer_path)

    # Create ONNX model names based on original names
    detector_onnx_name = os.path.splitext(detector_name)[0] + ".onnx"
    recognizer_onnx_name = os.path.splitext(recognizer_name)[0] + ".onnx"

    logging.info(f"Converting detector: {detector_name} -> {detector_onnx_name}")
    logging.info(f"Converting recognizer: {recognizer_name} -> {recognizer_onnx_name}")

    # Export detector model
    # The detector and recognizer objects are part of the reader instance
    detector_model = reader.detector
    detector_model.eval()

    # Create example input for detector
    # Common input size for CRAFT detector, adjust if using a different detector architecture
    dummy_input_detector = torch.randn(1, 3, 640, 640)
    if use_gpu and torch.cuda.is_available():
        dummy_input_detector = dummy_input_detector.cuda()
        detector_model = detector_model.cuda()

    det_output_path = os.path.join(output_dir, detector_onnx_name)

    # Export detector to ONNX
    torch.onnx.export(
        detector_model,
        dummy_input_detector,
        det_output_path,
        export_params=True,
        opset_version=12, # EasyOCR usually works well with opset 11 or 12
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'], # Or specific output names if known, e.g., ['y', 'feature'] for CRAFT
        dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                     'output': {0: 'batch_size'} # Adjust if output shapes are more complex
                    }
    )

    logging.info(f"Detector exported to {det_output_path}")

    # Export recognizer model
    recognizer_model = reader.recognizer
    recognizer_model.eval()

    # Create example input for recognizer
    # Common input size, e.g. (batch, channel, height, width)
    # Height is often fixed (e.g., 32 or 48), width is variable for text recognition.
    dummy_input_recognizer = torch.randn(1, 1, 48, 320) # Assuming grayscale input (channel=1) for some recognizers
    # If your recognizer model expects 3 channels (RGB), use: torch.randn(1, 3, 48, 320)
    # Check the specific recognizer model architecture for correct input shape
    # For example, if reader.recognizer.module.img_channel == 1 for CRNN
    if hasattr(reader.recognizer, 'module') and hasattr(reader.recognizer.module, 'img_channel') and reader.recognizer.module.img_channel == 1:
        dummy_input_recognizer = torch.randn(1, 1, reader.recognizer.module.imgH, 320)
    elif hasattr(reader.recognizer, 'module') and hasattr(reader.recognizer.module, 'imgH'): # For other models, assume 3 channels if not specified
        dummy_input_recognizer = torch.randn(1, 3, reader.recognizer.module.imgH, 320)
    else: # Fallback if imgH or img_channel not found
        dummy_input_recognizer = torch.randn(1, 3, 48, 320)
        logging.warning("Could not determine exact recognizer input shape, using default (1,3,48,320).")

    if use_gpu and torch.cuda.is_available():
        dummy_input_recognizer = dummy_input_recognizer.cuda()
        recognizer_model = recognizer_model.cuda()

    rec_output_path = os.path.join(output_dir, recognizer_onnx_name)

    # Export recognizer to ONNX
    torch.onnx.export(
        recognizer_model,
        dummy_input_recognizer,
        rec_output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size', 3: 'width'}, # height is usually fixed for recognizer
                     'output': {0: 'batch_size', 1: 'sequence_length'} # output shape depends on model (CTC, Attention)
                    }
    )

    logging.info(f"Recognizer exported to {rec_output_path}")

    # Save vocabulary (character list for the recognizer)
    # This is crucial for the recognizer to decode predictions correctly.
    vocab = reader.recognizer.character
    vocab_path = os.path.join(output_dir, "vocab.txt")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab))

    logging.info(f"Vocabulary saved to {vocab_path}")

    # Create a metadata file to remember which models were converted
    metadata_path = os.path.join(output_dir, "models_info.txt")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write(f"Detector: {detector_name} -> {detector_onnx_name}\n")
        f.write(f"Recognizer: {recognizer_name} -> {recognizer_onnx_name}\n")
        f.write(f"Languages: {','.join(languages)}\n")
        f.write(f"Conversion date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Upload models to Hugging Face if requested
    if upload_to_hf:
        if not hf_repo_id:
            raise ValueError("Hugging Face repository ID must be provided for upload")

        # hf_token can be optional if user is already logged in via huggingface-cli
        # However, explicitly passing it is safer for scripts.

        logging.info(f"Uploading ONNX models to Hugging Face repository: {hf_repo_id}")
        files_to_upload = [
            (det_output_path, detector_onnx_name),
            (rec_output_path, recognizer_onnx_name),
            (vocab_path, "vocab.txt"),
            (metadata_path, "models_info.txt")
        ]

        for file_path, path_in_repo in files_to_upload:
            try:
                upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=path_in_repo,
                    repo_id=hf_repo_id,
                    token=hf_token # Pass token if provided
                )
                logging.info(f"Successfully uploaded {path_in_repo} to {hf_repo_id}")
            except Exception as e:
                logging.error(f"Failed to upload {path_in_repo} to {hf_repo_id}: {e}")

    return output_dir, detector_onnx_name, recognizer_onnx_name

def download_models_from_hf(repo_id, local_dir=None, hf_token=None):
    """
    Download models from a Hugging Face repository.

    Args:
        repo_id: Hugging Face repository ID (e.g., "InfiniFlow/easyocr-onnx-models")
        local_dir: Local directory to save models.
                   Defaults to "rag/res/easyocr/onnx" relative to project base.
        hf_token: Optional Hugging Face API token for private repositories.

    Returns:
        Path to the local directory where models are downloaded.
    """
    if local_dir is None:
        try:
            from api.utils.file_utils import get_project_base_directory
            local_dir = os.path.join(get_project_base_directory(), "rag/res/easyocr/onnx")
        except ImportError:
            logging.warning("Could not import get_project_base_directory, using default relative path for ONNX models download.")
            local_dir = os.path.join("rag/res/easyocr/onnx")

    os.makedirs(local_dir, exist_ok=True)

    logging.info(f"Downloading ONNX models from Hugging Face repository: {repo_id} to {local_dir}")

    try:
        # snapshot_download will download all files from the repo_id to local_dir
        model_dir_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=hf_token, # Pass token if provided, for private repos
            local_dir_use_symlinks=False # Recommended to avoid issues with symlinks
        )
        logging.info(f"Models successfully downloaded to {model_dir_path}")
        return model_dir_path
    except Exception as e:
        logging.error(f"Failed to download models from {repo_id}: {e}")
        # Depending on desired behavior, you might re-raise the exception
        # or return None, or the initially intended local_dir if some files might exist
        raise # Re-raise the exception to make the caller aware of the failure