#
# Based on work from InfiniFlow project
# This file was created by Alexander Fedotov as an extension to the original project
# Licensed under the Apache License, Version 2.0
#

import os
import sys
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            '../../')))

from deepdoc.vision.easyocr_wrapper.ocr import EasyOCR
from deepdoc.vision import init_in_out
import argparse
import numpy as np
import trio

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2' #2 gpus, uncontinuous
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #1 gpu
# os.environ['CUDA_VISIBLE_DEVICES'] = '' #cpu


def main(args):
    import torch.cuda

    cuda_devices = torch.cuda.device_count()
    limiter = [trio.CapacityLimiter(1) for _ in range(cuda_devices)] if cuda_devices > 1 else None
    easyocr = EasyOCR()
    images, outputs = init_in_out(args)

    def __ocr_easy(i, img):
        print("Task {} start".format(i))

        # Convert PIL Image to numpy array
        img_array = np.array(img)

        # Use recognize_text method for direct text recognition
        # Pass paragraph parameter from command line arguments
        text = easyocr.recognize_text(img_array, paragraph=args.paragraph)

        # Display recognition mode
        mode = "Paragraph mode" if args.paragraph else "Separate lines mode"

        # Print recognized text to console with header
        print("\n" + "="*40)
        print(f"RECOGNIZED TEXT (image {i+1}) - {mode}:")
        print("="*40)
        print(text)
        print("="*40 + "\n")

        # Save text to a separate file
        output_path = outputs[i]
        with open(output_path + ".txt", "w+", encoding='utf-8') as f:
            f.write(text)

        print("Task {} done - text saved to {}".format(i, output_path + ".txt"))

    async def __ocr_thread(i, img, limiter=None):
        if limiter:
            async with limiter:
                print("Task {} use device {}".format(i, i % cuda_devices))
                await trio.to_thread.run_sync(lambda: __ocr_easy(i, img))
        else:
            __ocr_easy(i, img)

    async def __ocr_launcher():
        if cuda_devices > 1:
            async with trio.open_nursery() as nursery:
                for i, img in enumerate(images):
                    nursery.start_soon(__ocr_thread, i, img, limiter[i % cuda_devices])
                    await trio.sleep(0.1)
        else:
            for i, img in enumerate(images):
                await __ocr_thread(i, img)

    trio.run(__ocr_launcher)

    print("EasyOCR tasks are all done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',
                        help="Directory where to store images or PDFs, or a file path to a single image or PDF",
                        required=True)
    parser.add_argument('--output_dir',
                        help="Directory where to store the output images. Default: './easyocr_outputs'",
                        default="./easyocr_outputs")
    parser.add_argument('--paragraph',
                        help="Merge recognized text into paragraphs",
                        action='store_true')
    args = parser.parse_args()

    main(args)