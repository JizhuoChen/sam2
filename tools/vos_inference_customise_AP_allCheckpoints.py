import argparse
import os
import logging
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# For soft-score metrics
from sklearn.metrics import average_precision_score

def save_grayscale_mask(path: str, mask_array: np.ndarray, palette: bytes = None):
    """
    Save a single-channel mask (H,W) as a PNG file, optionally with a palette.
    """
    out_img = Image.fromarray(mask_array.astype(np.uint8))
    if palette is not None:
        out_img.putpalette(palette)
    out_img.save(path)

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes Intersection over Union between two binary masks.
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def compute_average_precision(pred_score: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute pixel-level Average Precision between predicted scores and binary GT.
    pred_score: H,W  in [0,1] or raw logits (apply sigmoid if so).
    gt_mask:    H,W  in {0,1}.
    """
    # Flatten
    pred_score_flat = pred_score.flatten().astype(np.float32)
    gt_mask_flat = gt_mask.flatten().astype(np.uint8)

    # If pred_score is actually raw logits, apply sigmoid:
    # pred_score_flat = 1.0 / (1.0 + np.exp(-pred_score_flat))

    return average_precision_score(gt_mask_flat, pred_score_flat)

def single_frame_inference(
    image_predictor: SAM2ImagePredictor,
    image_path: str,
    output_mask_path: str,
    score_thresh: float = 0.0,
    use_multimask_output: bool = False,
):
    """
    Returns:
      final_mask_bin -> thresholded binary mask (uint8)
      pred_prob      -> the continuous [0,1] predicted mask (or raw logits).
    """
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    image_predictor.reset_predictor()
    image_predictor.set_image(pil_image)

    bottom_left  = (0, height - 1)
    bottom_right = (width - 1, height - 1)
    bottom_mid   = ((width - 1)//2, height - 1)
    point_coords = np.array([bottom_left, bottom_right, bottom_mid], dtype=np.float32)
    point_labels = np.ones(len(point_coords), dtype=np.int32)

    masks, iou_predictions, low_res_logits = image_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=use_multimask_output,
        return_logits=True,
        normalize_coords=False
    )

    best_mask_idx = int(iou_predictions.argmax()) if use_multimask_output else 0
    pred_prob = masks[best_mask_idx]  # shape (H, W), presumably in [0,1]

    # Binary threshold
    final_mask_bin = (pred_prob > score_thresh).astype(np.uint8)
    
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    save_grayscale_mask(output_mask_path, final_mask_bin, palette=None)
    logging.info(f"Saved single-object mask to {output_mask_path}")
    
    return final_mask_bin, pred_prob

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

    if args.sam2_checkpoint_dir:
        ckpt_files = [f for f in os.listdir(args.sam2_checkpoint_dir)
                      if (f.endswith(".pt") or f.endswith(".pth") or f.endswith(".ckpt"))]
        ckpt_files.sort()
        ckpt_paths = [os.path.join(args.sam2_checkpoint_dir, f) for f in ckpt_files]
    else:
        if not args.sam2_checkpoint:
            raise ValueError("Either --sam2_checkpoint_dir or --sam2_checkpoint must be provided!")
        ckpt_paths = [args.sam2_checkpoint]

    results_per_ckpt = []

    for ckpt_path in ckpt_paths:
        logging.info(f"=============================================")
        logging.info(f" Evaluating checkpoint: {ckpt_path}")
        logging.info(f"=============================================")

        sam2_model = build_sam2(args.sam2_cfg, ckpt_path)
        image_predictor = SAM2ImagePredictor(
            sam_model=sam2_model,
            mask_threshold=args.score_thresh
        )

        folder_names = [
            name for name in os.listdir(args.base_dir)
            if os.path.isdir(os.path.join(args.base_dir, name))
        ]

        iou_scores = []
        ap_scores = []    # <--- We'll store pixel-level Average Precision here
        processed_names = []

        for folder_name in folder_names:
            folder_path = os.path.join(args.base_dir, folder_name)
            images = [
                p for p in os.listdir(folder_path)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
            ]
            if len(images) == 0:
                continue
            image_name = images[0]
            image_path = os.path.join(folder_path, image_name)

            ckpt_base = os.path.splitext(os.path.basename(ckpt_path))[0]
            base_name = os.path.splitext(image_name)[0]
            output_mask_path = os.path.join(
                args.output_mask_dir,
                ckpt_base,
                folder_name,
                f"{base_name}.png"
            )

            # Get thresholded mask + continuous predictions
            pred_mask_bin, pred_prob = single_frame_inference(
                image_predictor=image_predictor,
                image_path=image_path,
                output_mask_path=output_mask_path,
                score_thresh=args.score_thresh,
                use_multimask_output=args.multimask_output,
            )

            # Load GT
            gt_folder_path = os.path.join(args.gt_mask_dir, folder_name)
            if not os.path.isdir(gt_folder_path):
                continue
            gt_masks = [
                p for p in os.listdir(gt_folder_path)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
            ]
            if len(gt_masks) == 0:
                continue
            gt_mask_name = gt_masks[0]
            gt_mask_path = os.path.join(gt_folder_path, gt_mask_name)

            gt_img = Image.open(gt_mask_path)
            gt_array = np.array(gt_img)
            if gt_array.ndim == 3:
                gt_bin = (gt_array.sum(axis=-1) != 0)
            else:
                gt_bin = (gt_array != 0)

            # IoU (thresholded)
            pred_bin = (pred_mask_bin != 0)  # boolean
            iou_val = compute_iou(pred_bin, gt_bin)
            iou_scores.append(iou_val)

            # Pixel-Level Average Precision (soft scores)
            # pred_prob is assumed in [0,1], while gt_bin is boolean
            ap_val = compute_average_precision(pred_prob, gt_bin)
            ap_scores.append(ap_val)

            processed_names.append(folder_name)
            logging.info(f"IoU for '{folder_name}': {iou_val:.4f}, AP: {ap_val:.4f}")

        if len(iou_scores) > 0:
            mean_iou = sum(iou_scores) / len(iou_scores)
            mean_ap  = sum(ap_scores) / len(ap_scores)

            logging.info(f"======== METRIC SUMMARY for {ckpt_path} ========")
            for name, iou_val, ap_val in zip(processed_names, iou_scores, ap_scores):
                logging.info(f"{name:>20s}: IoU={iou_val:.4f}, AP={ap_val:.4f}")
            logging.info(f"Mean IoU: {mean_iou:.4f},  Mean AP: {mean_ap:.4f}")

            results_per_ckpt.append((ckpt_path, mean_iou, mean_ap))
        else:
            logging.info("No IoU or AP computed (no valid GT or predictions).")
            results_per_ckpt.append((ckpt_path, 0.0, 0.0))

    logging.info("========== Final Results Across All Checkpoints ==========")
    for ckpt_path, mean_iou, mean_ap in results_per_ckpt:
        ckpt_file = os.path.basename(ckpt_path)
        logging.info(f"{ckpt_file:>30s}: IoU={mean_iou:.4f}, AP={mean_ap:.4f}")
    logging.info("==========================================================")
    logging.info("Done.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
