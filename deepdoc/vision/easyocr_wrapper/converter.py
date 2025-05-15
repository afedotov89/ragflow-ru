#
# Based on work from InfiniFlow project
# This file was created by Alexander Fedotov as an extension to the original project
# Licensed under the Apache License, Version 2.0
#

import os
import logging
import time
from huggingface_hub import snapshot_download, upload_file
from rag.settings import EASYOCR_ONNX_REPO_ID
from .ocr import EasyOCR  # Import the EasyOCR class to access model name constants

def convert_to_onnx(languages=None, output_dir=None, use_gpu=True, upload_to_hf=False, hf_repo_id=None, hf_token=None):
    """
    Convert EasyOCR models to ONNX format

    Args:
        languages: List of languages to include in the model
        output_dir: Directory to save ONNX models
        use_gpu: Whether to use GPU for conversion
        upload_to_hf: Whether to upload models to Hugging Face
        hf_repo_id: Hugging Face repository ID
        hf_token: Hugging Face API token

    Returns:
        Tuple of (output_dir, detector_onnx_name, recognizer_onnx_name)
    """
    languages = EasyOCR.LANGUAGES if languages is None else languages

    # Use the default repo ID from settings if not provided
    if upload_to_hf and not hf_repo_id:
        hf_repo_id = EASYOCR_ONNX_REPO_ID
        logging.info(f"Using default Hugging Face repository ID from settings: {hf_repo_id}")

    # Force CPU for conversion to avoid device mismatch issues, especially with MPS.
    use_gpu = False
    logging.info("Forcing CPU for ONNX conversion process to ensure stability.")

    try:
        import torch
        import easyocr
    except ImportError:
        raise ImportError("PyTorch and EasyOCR must be installed for conversion")

    if output_dir is None:
        try:
            from api.utils.file_utils import get_project_base_directory
            output_dir = os.path.join(get_project_base_directory(), "rag/res/easyocr/onnx")
        except ImportError:
            logging.warning("Could not import get_project_base_directory, using default relative path for ONNX models.")
            output_dir = os.path.join("rag/res/easyocr/onnx")

    os.makedirs(output_dir, exist_ok=True)

    # Initialize EasyOCR Reader
    logging.info(f"Initializing EasyOCR Reader with languages: {languages}, gpu: {use_gpu}, model_storage_directory: {output_dir}")
    try:
        pth_model_dir = os.path.join(output_dir, ".easyocr_pth_models")
        os.makedirs(pth_model_dir, exist_ok=True)
        logging.info(f"EasyOCR .pth models will be stored/looked for in: {pth_model_dir}")

        reader = easyocr.Reader(
            lang_list=languages,
            gpu=use_gpu,
            model_storage_directory=pth_model_dir,
            download_enabled=True,
            detector=True,
            recognizer=True,
            verbose=True
        )
        logging.info("EasyOCR Reader initialized successfully.")

        if reader.detector is None:
            logging.error("EasyOCR reader.detector is None after initialization. Cannot proceed with ONNX conversion for detector.")
            raise ValueError("EasyOCR detector model failed to load.")
        if reader.recognizer is None:
            logging.error("EasyOCR reader.recognizer is None after initialization. Cannot proceed with ONNX conversion for recognizer.")
            raise ValueError("EasyOCR recognizer model failed to load.")

        logging.info(f"Successfully loaded EasyOCR PyTorch models: detector={type(reader.detector)}, recognizer={type(reader.recognizer)}")

    except Exception as e:
        logging.error(f"CRITICAL: Failed to initialize EasyOCR Reader or its internal models. Error: {e}")
        logging.error("Check if EasyOCR can download models (network access, permissions) or if models exist in the specified model_storage_directory.")
        raise

    # Define ONNX model names using the standardized constants from EasyOCR class
    actual_detector_model_key = EasyOCR.DETECTOR_MODEL
    actual_recognizer_model_key = EasyOCR.RECOGNIZER_MODEL

    detector_onnx_name = f"{actual_detector_model_key}.onnx"
    recognizer_onnx_name = f"{actual_recognizer_model_key}.onnx"

    logging.info(f"Target ONNX Detector: {detector_onnx_name}")
    logging.info(f"Target ONNX Recognizer: {recognizer_onnx_name}")

    # Export detector model
    detector_model_to_export = reader.detector
    if isinstance(detector_model_to_export, torch.nn.DataParallel):
        logging.info("Detector model is wrapped in DataParallel, using .module for ONNX export.")
        detector_model_to_export = detector_model_to_export.module
    detector_model_to_export.eval()

    # Create example input for detector
    dummy_input_detector = torch.randn(1, 3, 640, 640)
    if use_gpu and torch.cuda.is_available():
        dummy_input_detector = dummy_input_detector.cuda()
        detector_model_to_export = detector_model_to_export.cuda()

    det_output_path = os.path.join(output_dir, detector_onnx_name)

    # Export detector to ONNX
    torch.onnx.export(
        detector_model_to_export,
        dummy_input_detector,
        det_output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                     'output': {0: 'batch_size'}}
    )

    logging.info(f"Detector exported to {det_output_path}")

    # Export recognizer model
    recognizer_model_to_export = reader.recognizer
    if isinstance(recognizer_model_to_export, torch.nn.DataParallel):
        logging.info("Recognizer model is wrapped in DataParallel, using .module for ONNX export.")
        recognizer_model_to_export = recognizer_model_to_export.module
    recognizer_model_to_export.eval()

    # Create example input for recognizer
    dummy_input_recognizer_image = torch.randn(1, 1, 48, 320)

    # Try to get more precise input shape based on loaded recognizer model
    if hasattr(recognizer_model_to_export, 'module') and hasattr(recognizer_model_to_export.module, 'img_channel') and hasattr(recognizer_model_to_export.module, 'imgH'):
        img_channel = recognizer_model_to_export.module.img_channel
        imgH = recognizer_model_to_export.module.imgH
        dummy_input_recognizer_image = torch.randn(1, img_channel, imgH, 320)
        logging.info(f"Using more precise recognizer input shape: (1, {img_channel}, {imgH}, 320)")
    elif hasattr(recognizer_model_to_export, 'img_channel') and hasattr(recognizer_model_to_export, 'imgH'):
        img_channel = recognizer_model_to_export.img_channel
        imgH = recognizer_model_to_export.imgH
        dummy_input_recognizer_image = torch.randn(1, img_channel, imgH, 320)
        logging.info(f"Using more precise recognizer input shape (no module): (1, {img_channel}, {imgH}, 320)")
    else:
        logging.warning("Could not determine exact recognizer input shape from model attributes, using default (1,1,48,320). This might require adjustment.")
        img_channel = 1
        imgH = 48

    batch_max_length = 25
    dummy_input_recognizer_text = torch.zeros((1, batch_max_length), dtype=torch.long)

    recognizer_input_args = (dummy_input_recognizer_image, dummy_input_recognizer_text)
    recognizer_input_names = ['image', 'text_input']
    recognizer_dynamic_axes = {
        'image': {0: 'batch_size', 3: 'width'},
        'text_input': {0: 'batch_size', 1: 'sequence_length'}
    }

    rec_output_path = os.path.join(output_dir, recognizer_onnx_name)

    # Export recognizer to ONNX
    logging.info(f"Exporting recognizer model to {rec_output_path} with inputs: image ({dummy_input_recognizer_image.shape}), text_input ({dummy_input_recognizer_text.shape})")
    torch.onnx.export(
        recognizer_model_to_export,
        recognizer_input_args,
        rec_output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=recognizer_input_names,
        output_names=['output'],
        dynamic_axes=recognizer_dynamic_axes
    )

    logging.info(f"Recognizer exported to {rec_output_path}")

    # Save vocabulary (character list for the recognizer)
    vocab_path = os.path.join(output_dir, "vocab.txt")
    if not hasattr(reader, 'character'):
        raise RuntimeError("EasyOCR Reader does not have a 'character' attribute. Please use a compatible version of EasyOCR.")
    vocab = list(reader.character)
    vocab_with_blank = ['_'] + vocab  # Add blank symbol as the first line
    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab_with_blank))
    logging.info(f"Vocabulary saved to {vocab_path}")

    # Create a metadata file to remember which models were converted
    metadata_path = os.path.join(output_dir, "models_info.txt")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write(f"Detector: {actual_detector_model_key} -> {detector_onnx_name}\n")
        f.write(f"Recognizer: {actual_recognizer_model_key} -> {recognizer_onnx_name}\n")
        f.write(f"Languages: {','.join(languages)}\n")
        f.write(f"Conversion date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Upload models to Hugging Face if requested
    if upload_to_hf:
        if not hf_repo_id:
            raise ValueError("Hugging Face repository ID must be provided for upload")

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
                    token=hf_token
                )
                logging.info(f"Successfully uploaded {path_in_repo} to {hf_repo_id}")
            except Exception as e:
                logging.error(f"Failed to upload {path_in_repo} to {hf_repo_id}: {e}")

    return output_dir, detector_onnx_name, recognizer_onnx_name

def download_models_from_hf(repo_id=None, local_dir=None, hf_token=None):
    """
    Download models from a Hugging Face repository.

    Args:
        repo_id: Hugging Face repository ID (e.g., "afedotov/easyocr-onnx")
                If None, uses the default from settings.
        local_dir: Local directory to save models.
                   Defaults to "rag/res/easyocr/onnx" relative to project base.
        hf_token: Optional Hugging Face API token for private repositories.

    Returns:
        Path to the local directory where models are downloaded.
    """
    # Use the default repo ID from settings if not provided
    if repo_id is None:
        repo_id = EASYOCR_ONNX_REPO_ID
        logging.info(f"Using default Hugging Face repository ID from settings: {repo_id}")

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
        model_dir_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=hf_token,
            local_dir_use_symlinks=False
        )
        logging.info(f"Models successfully downloaded to {model_dir_path}")
        return model_dir_path
    except Exception as e:
        logging.error(f"Failed to download models from {repo_id}: {e}")
        raise