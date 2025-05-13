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
    # Force CPU for conversion to avoid device mismatch issues, especially with MPS.
    use_gpu = False
    logging.info("Forcing CPU for ONNX conversion process to ensure stability.")

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

    # Initialize EasyOCR Reader
    # This step is crucial as it will download .pth models if not present
    # and initialize reader.detector and reader.recognizer (PyTorch models)
    logging.info(f"Initializing EasyOCR Reader with languages: {languages}, gpu: {use_gpu}, model_storage_directory: {output_dir}")
    try:
        # We point model_storage_directory to our intended ONNX output_dir's parent,
        # or a dedicated ".EasyOCR/model" subdir within it, so .pth files are managed predictably.
        # For conversion, EasyOCR needs to load its .pth models first.
        # It will download them to `model_storage_directory` if `download_enabled` is True.
        pth_model_dir = os.path.join(output_dir, ".easyocr_pth_models") # Store .pth in a subdir of ONNX output
        os.makedirs(pth_model_dir, exist_ok=True)
        logging.info(f"EasyOCR .pth models will be stored/looked for in: {pth_model_dir}")

        reader = easyocr.Reader(
            lang_list=languages,
            gpu=use_gpu,
            model_storage_directory=pth_model_dir,
            download_enabled=True,
            detector=True,  # Ensure detector model is loaded
            recognizer=True, # Ensure recognizer model is loaded
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

    # Define ONNX model names
    # Based on common EasyOCR model names. Detector is usually CRAFT.
    # Recognizer names depend on the language(s).
    # For simplicity, we'll use generic names or derive from the first language if specific mapping isn't easily available.
    # Ideally, EasyOCR might provide the exact loaded model names, but we're working around that.

    detector_base_name = "craft_mlt_25k" # Common detector
    # For recognizer, we might need a more robust way to get the name if multiple languages are used,
    # or if the recognizer model name isn't fixed per language.
    # For now, let's assume the first language gives a hint or use a generic name.
    # This part might need refinement based on how EasyOCR names its multi-language models or selected recognizer.

    # Attempt to get specific model names used by the reader if available (new internal API might exist)
    # This is an attempt to get the actual model name chosen by EasyOCR
    actual_detector_model_key = getattr(reader, 'detector_model', detector_base_name) # e.g., 'craft_mlt_25k'
    actual_recognizer_model_key = getattr(reader, 'recog_network_name', None) # Newer attribute? or derive from lang_list

    if actual_recognizer_model_key is None:
        # Fallback: Construct a plausible recognizer name from the language list
        # This is a heuristic. EasyOCR's internal choice for multi-language models can be complex.
        # We take the first language as a primary hint.
        lang_prefix = languages[0]
        # Try to find a recognizer model file that EasyOCR might have downloaded for this language.
        # This requires inspecting pth_model_dir or using a known naming convention from EasyOCR's model hub.
        # Example: Russian might be 'russian_g2.pth', English 'english_g2.pth'
        # For this example, let's assume a generic naming if not directly found.
        # A more robust solution would be to check `reader.recognizer_model_filenames` if such an attribute exists
        # or parse `reader.recog_network` if it gives a direct hint.

        # Let's look for common patterns in the pth_model_dir
        best_rec_candidate = None
        if os.path.exists(pth_model_dir):
            for lang_code in languages: # Check all specified languages
                # Common patterns: <lang>_g<version>.pth or just <lang>.pth
                # Example: 'english_g2.pth', 'russian_g2.pth'
                # We are looking for the .pth file to derive the base name.
                potential_rec_names = [f"{lang_code}.pth", f"{lang_code}_g1.pth", f"{lang_code}_g2.pth", f"{lang_code}_g3.pth"]
                for fname_candidate in potential_rec_names:
                    if os.path.exists(os.path.join(pth_model_dir, fname_candidate)):
                        best_rec_candidate = os.path.splitext(fname_candidate)[0]
                        logging.info(f"Found potential recognizer base name from downloaded files: {best_rec_candidate}")
                        break
                if best_rec_candidate:
                    break

        if best_rec_candidate:
             actual_recognizer_model_key = best_rec_candidate
        else:
            # Fallback if no specific recognizer model name could be determined
            actual_recognizer_model_key = f"{languages[0]}_recognizer" # Default if no better name found
            logging.warning(f"Could not determine specific recognizer model name, using generic: {actual_recognizer_model_key}")


    detector_onnx_name = f"{actual_detector_model_key}.onnx"
    recognizer_onnx_name = f"{actual_recognizer_model_key}.onnx"

    logging.info(f"Target ONNX Detector: {detector_onnx_name}")
    logging.info(f"Target ONNX Recognizer: {recognizer_onnx_name}")

    # Export detector model
    # The detector and recognizer objects are part of the reader instance
    detector_model_to_export = reader.detector
    if isinstance(detector_model_to_export, torch.nn.DataParallel):
        logging.info("Detector model is wrapped in DataParallel, using .module for ONNX export.")
        detector_model_to_export = detector_model_to_export.module
    detector_model_to_export.eval()

    # Create example input for detector
    # Common input size for CRAFT detector, adjust if using a different detector architecture
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
    recognizer_model_to_export = reader.recognizer
    if isinstance(recognizer_model_to_export, torch.nn.DataParallel):
        logging.info("Recognizer model is wrapped in DataParallel, using .module for ONNX export.")
        recognizer_model_to_export = recognizer_model_to_export.module
    recognizer_model_to_export.eval()

    # Create example input for recognizer
    # Common input size, e.g. (batch, channel, height, width)
    # Height is often fixed (e.g., 32 or 48), width is variable for text recognition.
    dummy_input_recognizer_image = torch.randn(1, 1, 48, 320) # Assuming grayscale input
    # Try to get more precise input shape based on loaded recognizer model
    if hasattr(recognizer_model_to_export, 'module') and hasattr(recognizer_model_to_export.module, 'img_channel') and hasattr(recognizer_model_to_export.module, 'imgH'):
        img_channel = recognizer_model_to_export.module.img_channel
        imgH = recognizer_model_to_export.module.imgH
        dummy_input_recognizer_image = torch.randn(1, img_channel, imgH, 320) # Width 320 is a common example
        logging.info(f"Using more precise recognizer input shape: (1, {img_channel}, {imgH}, 320)")
    elif hasattr(recognizer_model_to_export, 'img_channel') and hasattr(recognizer_model_to_export, 'imgH'): # For models not wrapped in DataParallel.module
        img_channel = recognizer_model_to_export.img_channel
        imgH = recognizer_model_to_export.imgH
        dummy_input_recognizer_image = torch.randn(1, img_channel, imgH, 320)
        logging.info(f"Using more precise recognizer input shape (no module): (1, {img_channel}, {imgH}, 320)")
    else:
        logging.warning("Could not determine exact recognizer input shape from model attributes, using default (1,1,48,320). This might require adjustment.")
        img_channel = 1 # Fallback
        imgH = 48      # Fallback

    # For the second argument `text` that the recognizer's forward method expects.
    # Based on common CTC model inputs, this is usually a tensor of target lengths or indices.
    # For ONNX export tracing, its content might not be critical, but its presence and type are.
    # Let's assume batch_size=1 and a max_text_length for the dummy input.
    # Kromtar used torch.rand for this, but LongTensor seems more appropriate if it represents indices or lengths.
    # However, to match Kromtar's guide which reported success, let's start with their approach for the second input structure if it was indeed float.
    # If model expects integer indices for text input (common for CTC loss calculation or some architectures):
    # max_text_length = 25 # A typical max length for dummy text input
    # dummy_input_recognizer_text = torch.randint(0, reader.character_len, (1, max_text_length), dtype=torch.long)
    # Based on Kromtar's guide, it seems they used a 2D float tensor for the second input as well for tracing.
    # Example: batch_size_2_1 = 50, in_shape_2=[1, batch_size_2_1], dummy_input_2 = torch.rand(in_shape_2)
    # This implies the second input might be related to something other than direct character indices for some models or during specific forward passes.
    # Let's use a shape that reflects a sequence length for a batch size of 1.
    # The actual `text_for_pred` used in EasyOCR's `recognition.py` is `torch.LongTensor(batch_size, self.batch_max_length).fill_(0)`. It is then used for teacher forcing if enabled.
    # For export, we need to provide *something*. Let's use a LongTensor as it is closer to the actual usage.
    batch_max_length = 25 # Corresponds to self.batch_max_length in EasyOCR, a common value.
    dummy_input_recognizer_text = torch.zeros((1, batch_max_length), dtype=torch.long) # Batch size 1

    recognizer_input_args = (dummy_input_recognizer_image, dummy_input_recognizer_text)
    recognizer_input_names = ['image', 'text_input']
    recognizer_dynamic_axes = {
        'image': {0: 'batch_size', 3: 'width'},
        'text_input': {0: 'batch_size', 1: 'sequence_length'}
    }

    # Forcing CPU for conversion, so ensure dummy inputs are on CPU
    # dummy_input_recognizer_image = dummy_input_recognizer_image.cpu()
    # dummy_input_recognizer_text = dummy_input_recognizer_text.cpu()
    # recognizer_model_to_export = recognizer_model_to_export.cpu() # model already on CPU due to global use_gpu=False

    rec_output_path = os.path.join(output_dir, recognizer_onnx_name)

    # Export recognizer to ONNX
    logging.info(f"Exporting recognizer model to {rec_output_path} with inputs: image ({dummy_input_recognizer_image.shape}), text_input ({dummy_input_recognizer_text.shape})")
    torch.onnx.export(
        recognizer_model_to_export,
        recognizer_input_args, # Tuple of inputs
        rec_output_path,
        export_params=True,
        opset_version=12, # Kromtar used 11, but 12 is also common
        do_constant_folding=True,
        input_names=recognizer_input_names,
        output_names=['output'],
        dynamic_axes=recognizer_dynamic_axes
    )

    logging.info(f"Recognizer exported to {rec_output_path}")

    # Save vocabulary (character list for the recognizer)
    # This is crucial for the recognizer to decode predictions correctly.
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
        f.write(f"Detector: {detector_base_name} -> {detector_onnx_name}\n")
        f.write(f"Recognizer: {actual_recognizer_model_key} -> {recognizer_onnx_name}\n")
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
        repo_id: Hugging Face repository ID (e.g., "afedotov/easyocr-onnx")
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