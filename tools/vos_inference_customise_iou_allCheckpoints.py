import argparse
import os
import logging
import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2  # Adjust import to match your actual codebase
from sam2.sam2_image_predictor import SAM2ImagePredictor

DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00..."  # truncated for brevity

def save_grayscale_mask(path: str, mask_array: np.ndarray, palette: bytes = None):
    """
    Save a single-channel mask (H,W) as a PNG file, optionally with a palette.
    `mask_array` should be dtype=uint8 or bool. 0=background, 1=foreground.
    """
    out_img = Image.fromarray(mask_array.astype(np.uint8))
    if palette is not None:
        out_img.putpalette(palette)
    out_img.save(path)

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes Intersection over Union between two binary masks.
    Both masks are assumed to be boolean (True=foreground, False=background).
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        # Both masks are empty => IoU = 1 if both are indeed empty,
        # otherwise 0. Typically intersection=0 as well, so IoU=1
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def single_frame_inference(
    image_predictor: SAM2ImagePredictor,
    image_path: str,
    output_mask_path: str,
    score_thresh: float = 0.0,
    use_multimask_output: bool = False,
):
    """
    1) Loads the single image from `image_path`
    2) Sets it in the SAM2ImagePredictor
    3) Creates 3 bottom points as prompt
    4) Runs `predict`
    5) Thresholds and saves the mask to `output_mask_path`.
    Returns the binary mask array (0 or 1).
    """
    # 1) Load the single image
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size

    # 2) Put it in the predictor
    image_predictor.reset_predictor()
    image_predictor.set_image(pil_image)  # This will do the embedding.

    # 3) Create the 3 bottom points (x,y) in pixel coordinates
    bottom_left  = (0,         height - 1)
    bottom_right = (width - 1, height - 1)
    bottom_mid   = ((width - 1)//2, height - 1)
    point_coords = np.array([bottom_left, bottom_right, bottom_mid], dtype=np.float32)
    point_labels = np.ones(len(point_coords), dtype=np.int32)  # 1=foreground

    # 4) Run predictor.predict(...)
    masks, iou_predictions, low_res_logits = image_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=use_multimask_output,
        return_logits=True,
        normalize_coords=False
    )
    # 'masks.shape' => (C, origH, origW); iou_predictions.shape => (C,)

    best_mask_idx = int(iou_predictions.argmax()) if use_multimask_output else 0
    best_mask_logit = masks[best_mask_idx]  # shape (H, W)

    # 5) Threshold the mask by `score_thresh`
    final_mask_bin = (best_mask_logit > score_thresh).astype(np.uint8)

    # Save to disk
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    save_grayscale_mask(output_mask_path, final_mask_bin, palette=None)
    logging.info(f"Saved single-object mask to {output_mask_path}")

    return final_mask_bin  # 0/1 mask as np.uint8

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM 2 model configuration file",
    )
    # You can keep this if you want, but we won't use it in multi-checkpoint mode.
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default=None,
        help="Path to a single model checkpoint (unused if --sam2_checkpoint_dir is given)",
    )
    parser.add_argument(
        "--sam2_checkpoint_dir",
        type=str,
        help="Directory containing multiple .pt checkpoints to evaluate",
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
        "--gt_mask_dir",
        type=str,
        required=True,
        help="directory containing subfolders of ground truth masks (same structure)."
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Gather the list of checkpoints. If sam2_checkpoint_dir is given, we use all
    # checkpoint files in that dir. Otherwise, fall back to single checkpoint usage.
    # -------------------------------------------------------------------------
    if args.sam2_checkpoint_dir:
        ckpt_files = [
            f for f in os.listdir(args.sam2_checkpoint_dir)
            if (f.endswith(".pt") or f.endswith(".pth") or f.endswith(".ckpt"))
        ]
        ckpt_files.sort()
        ckpt_paths = [os.path.join(args.sam2_checkpoint_dir, f) for f in ckpt_files]
        logging.info(f"Found {len(ckpt_paths)} checkpoints in {args.sam2_checkpoint_dir}")
    else:
        # Fallback: single checkpoint scenario
        if not args.sam2_checkpoint:
            raise ValueError("Either --sam2_checkpoint_dir or --sam2_checkpoint must be provided!")
        ckpt_paths = [args.sam2_checkpoint]
        logging.info(f"Using a single checkpoint: {ckpt_paths[0]}")

    # List to store the (checkpoint_name, mean_iou) for each checkpoint
    results_per_ckpt = []

    # -------------------------------------------------------------------------
    # For each checkpoint, build the model, run inference on all images, compute IoU
    # -------------------------------------------------------------------------
    for ckpt_path in ckpt_paths:
        logging.info(f"=============================================")
        logging.info(f" Evaluating checkpoint: {ckpt_path}")
        logging.info(f"=============================================")

        # 1) Build the SAM2 model from your config + checkpoint
        sam2_model = build_sam2(args.sam2_cfg, ckpt_path)

        # 2) Create an image predictor
        image_predictor = SAM2ImagePredictor(
            sam_model=sam2_model,
            mask_threshold=args.score_thresh
        )

        # 3) For each subfolder in base_dir, there's exactly 1 image
        folder_names = [
            name for name in os.listdir(args.base_dir)
            if os.path.isdir(os.path.join(args.base_dir, name))
        ]
        logging.info(f"Found {len(folder_names)} subfolders in {args.base_dir}.")

        iou_scores = []
        processed_names = []

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
                logging.warning(
                    f"Multiple images found in {folder_path}, using the first only: {images}"
                )

            image_name = images[0]
            image_path = os.path.join(folder_path, image_name)

            # Output path => place results in a subfolder named after checkpoint
            # e.g.  <output_mask_dir>/<CHECKPOINT_BASENAME>/<folder_name>/<image_name>.png
            ckpt_base = os.path.splitext(os.path.basename(ckpt_path))[0]  # e.g. "checkpoint_2"
            base_name = os.path.splitext(image_name)[0]
            output_mask_path = os.path.join(
                args.output_mask_dir,
                ckpt_base,
                folder_name,
                f"{base_name}.png"
            )

            # 5) Run inference and get predicted mask
            pred_mask_bin = single_frame_inference(
                image_predictor=image_predictor,
                image_path=image_path,
                output_mask_path=output_mask_path,
                score_thresh=args.score_thresh,
                use_multimask_output=args.multimask_output,
            )

            ####################################################################
            # Load the GT mask and compute IoU
            ####################################################################
            gt_folder_path = os.path.join(args.gt_mask_dir, folder_name)
            if not os.path.isdir(gt_folder_path):
                logging.warning(f"Ground truth folder not found for {folder_name}, skipping.")
                continue

            gt_masks = [
                p for p in os.listdir(gt_folder_path)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
            ]
            if len(gt_masks) == 0:
                logging.warning(f"No GT mask found in {gt_folder_path}, skipping IoU.")
                continue
            if len(gt_masks) > 1:
                logging.warning(
                    f"Multiple GT masks found in {gt_folder_path}, using first only: {gt_masks}"
                )
            gt_mask_name = gt_masks[0]
            gt_mask_path = os.path.join(gt_folder_path, gt_mask_name)

            gt_img = Image.open(gt_mask_path)
            gt_array = np.array(gt_img)

            # If the mask is RGB, sum across channels; otherwise use grayscale
            if gt_array.ndim == 3:
                gt_bin = (gt_array.sum(axis=-1) != 0)
            else:
                gt_bin = (gt_array != 0)

            pred_bin = (pred_mask_bin != 0)  # convert to boolean
            iou_val = compute_iou(pred_bin, gt_bin)
            iou_scores.append(iou_val)
            processed_names.append(folder_name)
            logging.info(f"IoU for '{folder_name}': {iou_val:.4f}")

        # Summarize for this checkpoint
        if len(iou_scores) > 0:
            mean_iou = sum(iou_scores) / len(iou_scores)
            logging.info(f"======== IOU SUMMARY for {ckpt_path} ========")
            for name, val in zip(processed_names, iou_scores):
                logging.info(f"{name:>20s}: {val:.4f}")
            logging.info(f"Mean IoU across {len(iou_scores)} images: {mean_iou:.4f}")
            results_per_ckpt.append((ckpt_path, mean_iou))
        else:
            logging.info("No IoU was computed (no valid GT or predictions).")
            results_per_ckpt.append((ckpt_path, 0.0))

    # -------------------------------------------------------------------------
    # Print final results across *all* checkpoints
    # -------------------------------------------------------------------------
    logging.info("========== Final IoU Results Across All Checkpoints ==========")
    for ckpt_path, mean_iou in results_per_ckpt:
        logging.info(f"{os.path.basename(ckpt_path):>30s}: {mean_iou:.4f}")
    logging.info("==============================================================")

    logging.info("Done.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
