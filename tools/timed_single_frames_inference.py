import argparse
import os
import logging
import time  # <-- For measuring inference times

import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2  # Loads your SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor


def save_grayscale_mask(
    path: str, 
    mask_array: np.ndarray, 
    palette: bytes = None
):
    """
    Save a single-channel mask (H,W) as a PNG file, optionally with a palette.
    `mask_array` should be dtype=uint8 or bool. 0=background, 1=foreground.
    """
    out_img = Image.fromarray(mask_array.astype(np.uint8))
    if palette is not None:
        out_img.putpalette(palette)
    out_img.save(path)


def single_frame_inference(
    image_predictor: SAM2ImagePredictor,
    image_path: str,
    output_mask_path: str,
    score_thresh: float = 0.0,
    use_multimask_output: bool = False,
):
    """
    Single-image inference with 3 bottom points. Returns the total time spent on
    set_image + predict (in seconds).
    """

    # Start timing
    start_time = time.time()

    # 1) Load the single image
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size

    # 2) Put it in the predictor. (Computes the embedding.)
    image_predictor.reset_predictor()
    image_predictor.set_image(pil_image)

    # 3) Create the 3 bottom points (x,y) in pixel coordinates
    bottom_left  = (0,       height - 1)
    bottom_right = (width-1, height - 1)
    bottom_mid   = ((width-1)//2, height - 1)
    point_coords = np.array([bottom_left, bottom_right, bottom_mid], dtype=np.float32)
    point_labels = np.ones(len(point_coords), dtype=np.int32)  # 1=foreground

    # 4) Run predictor.predict(...) with those points
    masks, iou_predictions, low_res_logits = image_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=use_multimask_output,
        return_logits=True,  # We'll get the raw logits
        normalize_coords=False
    )

    # Force GPU sync so we measure real time
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # End timing
    end_time = time.time() - start_time

    # Pick the best mask if multi-mask
    if use_multimask_output:
        best_mask_idx = int(iou_predictions.argmax())
    else:
        best_mask_idx = 0
    best_mask_logit = masks[best_mask_idx]  # shape (H, W)

    # 5) Threshold the mask by `score_thresh`
    final_mask_bin = (best_mask_logit > score_thresh).astype(np.uint8)

    # Save to disk
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    save_grayscale_mask(output_mask_path, final_mask_bin, palette=None)
    logging.info(f"Saved single-object mask to {output_mask_path}")

    return end_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="./checkpoints/sam2.1_hiera_base_plus.pt",
        help="path to the SAM 2 model checkpoint",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="directory containing subfolders, each with exactly ONE image",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        required=True,
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold to convert the mask logits to binary (default: 0.0)",
    )
    parser.add_argument(
        "--multimask_output",
        action="store_true",
        help="whether to generate multiple masks from the prompt (and pick best)."
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=0,
        help="Number of dummy images to process for GPU warmup. (Optional)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Building SAM2 model...")
    # 1) Build the SAM2 model from your config + checkpoint
    sam2_model = build_sam2(args.sam2_cfg, args.sam2_checkpoint)
    logging.info("Model built.")

    # 2) Create an image predictor
    image_predictor = SAM2ImagePredictor(
        sam_model=sam2_model,
        mask_threshold=args.score_thresh,
    )

    # Optionally run warm-up
    if args.warmup_iters > 0:
        logging.info(f"Running {args.warmup_iters} warmup iterations...")
        dummy_image = Image.new("RGB", (512, 512), color=(0, 0, 0))
        for _ in range(args.warmup_iters):
            image_predictor.reset_predictor()
            image_predictor.set_image(dummy_image)
            # Make up a dummy point
            coords = np.array([[0, 511]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            _ = image_predictor.predict(
                point_coords=coords,
                point_labels=labels,
                multimask_output=False,
                return_logits=True,
                normalize_coords=False
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        logging.info("Warmup completed.\n")

    # 3) For each subfolder in base_dir, there's exactly 1 image
    folder_names = [
        name for name in os.listdir(args.base_dir)
        if os.path.isdir(os.path.join(args.base_dir, name))
    ]
    logging.info(f"Found {len(folder_names)} subfolders in {args.base_dir}.")

    # We'll collect times for all images
    all_times = []

    for folder_name in folder_names:
        folder_path = os.path.join(args.base_dir, folder_name)
        images = [
            p for p in os.listdir(folder_path)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
        ]
        if len(images) == 0:
            logging.warning(f"No images found in {folder_path}, skipping.")
            continue
        if len(images) > 1:
            logging.warning(f"Found multiple images in {folder_path}, using first only: {images}")

        image_name = images[0]
        image_path = os.path.join(folder_path, image_name)

        # 4) Output path => e.g. <output_mask_dir>/<folder_name>/<image_name>.png
        base_name = os.path.splitext(image_name)[0]
        output_mask_path = os.path.join(
            args.output_mask_dir,
            folder_name,
            f"{base_name}.png"
        )

        # 5) Do single-frame inference & measure time
        elapsed_time = single_frame_inference(
            image_predictor=image_predictor,
            image_path=image_path,
            output_mask_path=output_mask_path,
            score_thresh=args.score_thresh,
            use_multimask_output=args.multimask_output,
        )
        all_times.append(elapsed_time)
        logging.info(
            f"Inference time for {folder_name}/{image_name}: {elapsed_time:.4f} sec"
        )

    if len(all_times) > 0:
        avg_time = sum(all_times) / len(all_times)
        min_time = min(all_times)
        max_time = max(all_times)
        logging.info(f"\nInference completed on {len(all_times)} images.")
        logging.info(f"Avg time: {avg_time:.4f} sec | Min: {min_time:.4f} | Max: {max_time:.4f}")
    else:
        logging.info("No images processed. Nothing to report.")

    logging.info(f"Done. Masks saved to: {args.output_mask_dir}")


if __name__ == "__main__":
    main()
