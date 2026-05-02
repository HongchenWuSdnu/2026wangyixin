import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as TF
from transformers import ChineseCLIPProcessor


class FasterRCNNRegionExtractor:
    def __init__(self, args):
        self.args = args
        self.args.device = self.resolve_device(args.device)
        if args.hf_endpoint:
            os.environ["HF_ENDPOINT"] = args.hf_endpoint
        if args.local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.data_dir = Path(args.data_dir)
        self.processed_dir = Path(args.processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir = self.processed_dir / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.clip_processor = None
        if args.export_clip_npy:
            self.clip_processor = ChineseCLIPProcessor.from_pretrained(
                args.clip_model_name,
                local_files_only=args.local_files_only,
            )
        self.detector = None
        if not args.use_fast_crop_fallback:
            try:
                weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                self.detector = fasterrcnn_resnet50_fpn(weights=weights)
            except Exception:
                self.detector = fasterrcnn_resnet50_fpn(weights=None)
            self.detector.to(self.args.device)
            self.detector.eval()

    @staticmethod
    def resolve_device(device_name: str) -> str:
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device_name

    def load_all_posts(self) -> pd.DataFrame:
        frames = []
        for split in ["train", "valid", "test"]:
            split_path = self.processed_dir / f"{split}_EANN_frozen.npy"
            if not split_path.exists():
                raise FileNotFoundError(f"Missing split file: {split_path}")
            split_df = pd.DataFrame(
                np.load(split_path, allow_pickle=True),
                columns=["original_post", "label", "image_id", "post_id"],
            )
            frames.append(split_df)
        merged = pd.concat(frames, axis=0, ignore_index=True)
        return merged.drop_duplicates(subset=["image_id"]).reset_index(drop=True)

    def resolve_image_path(self, image_id: str, label: int) -> Path:
        root_name = "rumor_images" if int(label) == 1 else "nonrumor_images"
        candidate_dir = self.data_dir / root_name
        for extension in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = candidate_dir / f"{image_id}{extension}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Image file for {image_id} not found in {candidate_dir}")

    def detect_regions(self, image: Image.Image) -> Tuple[List[List[float]], List[float], List[int]]:
        image_tensor = TF.to_tensor(image).to(self.args.device)
        with torch.no_grad():
            prediction = self.detector([image_tensor])[0]

        boxes = prediction["boxes"].detach().cpu()
        scores = prediction["scores"].detach().cpu()
        labels = prediction["labels"].detach().cpu()

        keep_mask = scores >= self.args.score_threshold
        boxes = boxes[keep_mask][: self.args.max_regions]
        scores = scores[keep_mask][: self.args.max_regions]
        labels = labels[keep_mask][: self.args.max_regions]
        return boxes.tolist(), scores.tolist(), labels.tolist()

    def fast_crop_regions(self, image: Image.Image) -> Tuple[List[List[float]], List[float], List[int]]:
        width, height = image.size
        half_w, half_h = width // 2, height // 2
        boxes = [
            [0, 0, width, height],
            [0, 0, half_w, half_h],
            [half_w, 0, width, half_h],
            [0, half_h, half_w, height],
            [half_w, half_h, width, height],
            [width // 4, height // 4, width * 3 // 4, height * 3 // 4],
        ]
        boxes = boxes[1 : self.args.max_regions + 1]
        scores = [1.0] * len(boxes)
        labels = [0] * len(boxes)
        return boxes, scores, labels

    def crop_regions(self, image: Image.Image, boxes: List[List[float]]) -> List[Image.Image]:
        width, height = image.size
        regions = [image]
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = int(max(0, min(x1, width - 1)))
            y1 = int(max(0, min(y1, height - 1)))
            x2 = int(max(x1 + 1, min(x2, width)))
            y2 = int(max(y1 + 1, min(y2, height)))
            regions.append(image.crop((x1, y1, x2, y2)))
        return regions[: self.args.crop_num]

    def prepare_region_for_clip(self, region: Image.Image) -> Image.Image:
        resized = region.copy()
        resized.thumbnail((self.args.resize_size, self.args.resize_size), Image.Resampling.BICUBIC)
        if resized.size != (self.args.resize_size, self.args.resize_size):
            canvas = Image.new("RGB", (self.args.resize_size, self.args.resize_size), color=(0, 0, 0))
            offset_x = (self.args.resize_size - resized.size[0]) // 2
            offset_y = (self.args.resize_size - resized.size[1]) // 2
            canvas.paste(resized, (offset_x, offset_y))
            return canvas
        return resized

    def encode_regions(self, regions: List[Image.Image]) -> torch.Tensor:
        pixel_values = []
        for region in regions:
            prepared_region = self.prepare_region_for_clip(region)
            encoded = self.clip_processor(images=prepared_region, return_tensors="pt")
            pixel_values.append(encoded["pixel_values"].squeeze(0))
        while len(pixel_values) < self.args.crop_num:
            pixel_values.append(torch.zeros(3, 224, 224))
        return torch.stack(pixel_values[: self.args.crop_num])

    def build_empty_region_tensor(self) -> torch.Tensor:
        return torch.zeros(self.args.crop_num, 3, 224, 224)

    def save_region_images(self, image_id: str, regions: List[Image.Image]):
        target_dir = self.crops_dir / image_id
        target_dir.mkdir(parents=True, exist_ok=True)
        for index, region in enumerate(regions):
            suffix = "full" if index == 0 else f"object_{index:02d}"
            region.save(target_dir / f"{suffix}.jpg")

    def run(self):
        image_df = self.load_all_posts()
        region_metadata = {}
        progress_desc = "Fast crop preprocess" if self.detector is None else "Faster R-CNN detect"
        missing_image_ids = []
        clip_inputs: Dict[str, np.ndarray] = {} if self.args.export_clip_npy else None

        for _, row in tqdm(image_df.iterrows(), total=len(image_df), desc=progress_desc):
            image_id = str(row["image_id"])
            try:
                image_path = self.resolve_image_path(image_id, int(row["label"]))
                image = Image.open(image_path).convert("RGB")
            except (FileNotFoundError, OSError):
                if clip_inputs is not None:
                    clip_inputs[image_id] = self.build_empty_region_tensor().numpy().astype(np.float16)
                region_metadata[image_id] = {
                    "image_path": None,
                    "boxes": [],
                    "scores": [],
                    "labels": [],
                    "region_count": 0,
                    "missing_image": True,
                }
                missing_image_ids.append(image_id)
                continue

            if self.detector is None:
                boxes, scores, labels = self.fast_crop_regions(image)
            else:
                boxes, scores, labels = self.detect_regions(image)
            regions = self.crop_regions(image, boxes)
            self.save_region_images(image_id, regions)
            if clip_inputs is not None:
                clip_inputs[image_id] = self.encode_regions(regions).numpy().astype(np.float16)
            region_metadata[image_id] = {
                "image_path": str(image_path),
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "region_count": len(regions),
            }

        if clip_inputs is not None:
            np.save(self.processed_dir / "clip_image_preprocess.npy", clip_inputs)
        with open(self.processed_dir / "faster_rcnn_regions.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "regions": region_metadata,
                    "missing_image_ids": missing_image_ids,
                    "missing_count": len(missing_image_ids),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end Faster R-CNN image preprocessing for Weibo fake news detection.")
    # ========== 修改点：默认路径与 process_text_weibo.py 保持一致 ==========
    parser.add_argument("--data_dir", type=str, default="data/weibo", help="Directory containing rumor_images and nonrumor_images")
    parser.add_argument("--processed_dir", type=str, default="data/weibo/processed", help="Directory containing split files and output crops")
    # =========================================================================
    parser.add_argument("--clip_model_name", type=str, default="OFA-Sys/chinese-clip-vit-base-patch16")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--crop_num", type=int, default=6)
    parser.add_argument("--max_regions", type=int, default=5)
    parser.add_argument("--score_threshold", type=float, default=0.7)
    parser.add_argument("--use_fast_crop_fallback", action="store_true")
    parser.add_argument("--export_clip_npy", action="store_true", help="Precompute CLIP image features for training")
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    FasterRCNNRegionExtractor(parse_args()).run()