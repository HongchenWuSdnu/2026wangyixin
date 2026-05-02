import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from transformers import ChineseCLIPProcessor


class FakeNewsDataset(Dataset):
    def __init__(
        self,
        data_df,
        crop_num,
        st_num,
        dataset,
        entity_tokens,
        crop_input,
        text_input,
        max_entities=15,
        processed_dir=None,
        clip_model_name="OFA-Sys/chinese-clip-vit-base-patch16",
        local_files_only=False,
    ):
        self.data_df = data_df
        self.crop_num = crop_num
        self.st_num = st_num
        self.dataset = dataset
        self.entity_tokens = entity_tokens
        self.crop_input = crop_input
        self.text_input = text_input
        self.max_entities = max_entities
        self.processed_dir = Path(processed_dir) if processed_dir is not None else None
        self.crops_dir = self.processed_dir / "crops" if self.processed_dir is not None else None
        self.clip_processor = None
        if self.crop_input is None:
            # 当不再预存全部图像张量时，按需读取裁剪图并做 Chinese-CLIP 预处理
            self.clip_processor = ChineseCLIPProcessor.from_pretrained(
                clip_model_name,
                local_files_only=local_files_only,
            )

    def __len__(self):
        return self.data_df.shape[0]

    def _ensure_tensor(self, value, dtype):
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype)
        return torch.tensor(value, dtype=dtype)

    def _pad_entities(self, entity_tokens):
        entity_tokens = self._ensure_tensor(entity_tokens, torch.long)
        cur_num = entity_tokens.shape[0]
        if cur_num < self.max_entities:
            pad = torch.zeros(self.max_entities - cur_num, entity_tokens.shape[1], dtype=entity_tokens.dtype)
            entity_tokens = torch.cat([entity_tokens, pad], dim=0)
        else:
            entity_tokens = entity_tokens[:self.max_entities]
        return entity_tokens

    def _pad_regions(self, crop_input):
        crop_input = self._ensure_tensor(crop_input, torch.float32)
        cur_num = crop_input.shape[0]
        if cur_num < self.crop_num:
            pad = torch.zeros(self.crop_num - cur_num, *crop_input.shape[1:], dtype=crop_input.dtype)
            crop_input = torch.cat([crop_input, pad], dim=0)
        else:
            crop_input = crop_input[:self.crop_num]
        return crop_input

    def _load_region_images(self, image_id):
        if self.crops_dir is None:
            raise ValueError("processed_dir is required when crop_input is not preloaded.")
        image_crop_dir = self.crops_dir / str(image_id)
        if not image_crop_dir.exists():
            return torch.zeros(self.crop_num, 3, 224, 224, dtype=torch.float32)

        region_tensors = []
        for index in range(self.crop_num):
            file_name = "full.jpg" if index == 0 else f"object_{index:02d}.jpg"
            region_path = image_crop_dir / file_name
            if not region_path.exists():
                break
            region = Image.open(region_path).convert("RGB")
            encoded = self.clip_processor(images=region, return_tensors="pt")
            region_tensors.append(encoded["pixel_values"].squeeze(0))

        if not region_tensors:
            return torch.zeros(self.crop_num, 3, 224, 224, dtype=torch.float32)
        return self._pad_regions(torch.stack(region_tensors))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        post_id = self.data_df["post_id"][idx]
        image_id = self.data_df["image_id"][idx]
        label = torch.tensor(self.data_df["label"][idx], dtype=torch.long)

        entity_tokens = self._pad_entities(self.entity_tokens[post_id])
        if self.crop_input is None:
            crop_input = self._load_region_images(image_id)
        else:
            crop_input = self._pad_regions(self.crop_input[image_id])
        text_input = self._ensure_tensor(self.text_input[post_id], torch.long)

        return {
            "post_id": post_id,
            "label": label,
            "crop_input": crop_input,
            "entity_tokens": entity_tokens,
            "text_input": text_input,
        }
