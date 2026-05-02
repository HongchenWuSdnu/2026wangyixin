import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, ChineseCLIPProcessor

ENTITY_LABELS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
ENTITY_TYPES = {"PER", "LOC", "ORG"}


class WeakChineseNERDataset(Dataset):
    def __init__(self, texts: Sequence[str], tokenizer, max_length: int, weak_patterns: Dict[str, Sequence[str]]):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {label: idx for idx, label in enumerate(ENTITY_LABELS)}
        self.weak_patterns = {
            entity_type: [re.compile(pattern) for pattern in patterns]
            for entity_type, patterns in weak_patterns.items()
        }

    def __len__(self):
        return len(self.texts)

    def _create_char_labels(self, text: str) -> List[str]:
        labels = ["O"] * len(text)
        for entity_type, patterns in self.weak_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    start, end = match.span()
                    if start >= end:
                        continue
                    labels[start] = f"B-{entity_type}"
                    for inner_idx in range(start + 1, end):
                        labels[inner_idx] = f"I-{entity_type}"
        return labels

    def __getitem__(self, idx):
        text = self.texts[idx]
        char_labels = self._create_char_labels(text)
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offsets = tokens.pop("offset_mapping").squeeze(0)
        label_ids = []
        for start, end in offsets.tolist():
            if start == end:
                label_ids.append(-100)
                continue
            label_ids.append(self.label2id[char_labels[start]])

        item = {key: value.squeeze(0) for key, value in tokens.items()}
        item["labels"] = torch.tensor(label_ids, dtype=torch.long)
        return item


class TextPreprocessor:
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
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name, local_files_only=args.local_files_only)
        self.clip_processor = ChineseCLIPProcessor.from_pretrained(
            args.clip_model_name,
            local_files_only=args.local_files_only,
        )
        self.label2id = {label: idx for idx, label in enumerate(ENTITY_LABELS)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    @staticmethod
    def resolve_device(device_name: str) -> str:
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device_name

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"(@.*?)[\s]", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&nbsp;", "", text)
        text = re.sub(r"&quot", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _iter_weibo_records(self, file_path: Path, label: int) -> Iterable[Tuple[str, str, str, int]]:
        with open(file_path, "rb") as handle:
            lines = handle.readlines()
        for idx in range(0, len(lines), 3):
            if idx + 2 >= len(lines):
                break
            post_id = lines[idx].decode("utf-8", errors="ignore").split("|")[0].strip()
            image_id = lines[idx + 1].decode("utf-8", errors="ignore").strip()
            original_post = self.clean_text(lines[idx + 2].decode("utf-8", errors="ignore"))
            yield post_id, image_id, original_post, label

    def load_raw_dataframe(self) -> pd.DataFrame:
        tweet_dir = self.data_dir / "tweets"
        file_specs = [
            (tweet_dir / "train_nonrumor.txt", 0),
            (tweet_dir / "train_rumor.txt", 1),
            (tweet_dir / "test_nonrumor.txt", 0),
            (tweet_dir / "test_rumor.txt", 1),
        ]
        rows = []
        for file_path, label in file_specs:
            if not file_path.exists():
                raise FileNotFoundError(f"Missing raw text file: {file_path}")
            rows.extend(self._iter_weibo_records(file_path, label))
        data_df = pd.DataFrame(rows, columns=["post_id", "image_id", "original_post", "label"])
        data_df.to_csv(self.processed_dir / "raw_weibo_posts.csv", index=False)
        return data_df

    def build_paired_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        image_roots = [self.data_dir / "nonrumor_images", self.data_dir / "rumor_images"]
        available_images = {}
        for root in image_roots:
            if not root.exists():
                continue
            for image_path in root.iterdir():
                if image_path.is_file():
                    available_images[image_path.stem] = image_path

        paired_rows = []
        for _, row in raw_df.iterrows():
            image_candidates = str(row["image_id"]).split("|")
            for image_ref in image_candidates:
                image_name = Path(image_ref).stem
                image_path = available_images.get(image_name)
                if image_path is not None:
                    paired_rows.append(
                        {
                            "post_id": row["post_id"],
                            "image_id": image_name,
                            "original_post": row["original_post"],
                            "label": int(row["label"]),
                            "image_path": str(image_path),
                        }
                    )
                    break
        paired_df = pd.DataFrame(paired_rows)
        if paired_df.empty:
            raise RuntimeError("No paired text-image samples were found. Check data_dir and raw Weibo files.")
        paired_df.to_csv(self.processed_dir / "paired_weibo_posts.csv", index=False)
        return paired_df

    def split_dataframe(self, paired_df: pd.DataFrame):
        if self.args.split_mode == "official":
            split_files = {
                "train": self.data_dir / "train_id.pickle",
                "valid": self.data_dir / "validate_id.pickle",
                "test": self.data_dir / "test_id.pickle",
            }
            if all(path.exists() for path in split_files.values()):
                split_frames = {}
                post_ids = paired_df["post_id"].astype(str)
                for split_name, split_path in split_files.items():
                    split_ids = {str(item) for item in pickle.load(open(split_path, "rb"))}
                    split_frames[split_name] = paired_df[post_ids.isin(split_ids)].copy().reset_index(drop=True)
                train_df = split_frames["train"]
                valid_df = split_frames["valid"]
                test_df = split_frames["test"]
            else:
                print("Official split files are missing. Fallback to random split.")
                self.args.split_mode = "random"

        if self.args.split_mode == "random":
            train_df, temp_df = train_test_split(
                paired_df,
                test_size=self.args.valid_ratio + self.args.test_ratio,
                stratify=paired_df["label"],
                random_state=self.args.seed,
            )
            relative_test_ratio = self.args.test_ratio / (self.args.valid_ratio + self.args.test_ratio)
            valid_df, test_df = train_test_split(
                temp_df,
                test_size=relative_test_ratio,
                stratify=temp_df["label"],
                random_state=self.args.seed,
            )
        for split_name, split_df in {"train": train_df, "valid": valid_df, "test": test_df}.items():
            split_df = split_df[["original_post", "label", "image_id", "post_id"]].reset_index(drop=True)
            np.save(self.processed_dir / f"{split_name}_EANN_frozen.npy", split_df.to_numpy(dtype=object))
            split_df.to_csv(self.processed_dir / f"{split_name}_EANN_frozen.csv", index=False)
        return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def build_weak_patterns(self) -> Dict[str, Sequence[str]]:
        return {
            "PER": [
                r"[\u4e00-\u9fa5]{2,4}(先生|女士|教授|书记|主席|总理|导演|记者)",
                r"[\u4e00-\u9fa5]{2,3}(称|表示|指出)",
            ],
            "LOC": [
                r"在[\u4e00-\u9fa5]{2,8}(市|省|县|区|镇|村)",
                r"[\u4e00-\u9fa5]{2,8}(机场|车站|广场|大学|医院)",
            ],
            "ORG": [
                r"[\u4e00-\u9fa5A-Za-z]{2,20}(公司|集团|大学|医院|银行|政府|委员会|部门|学院)",
                r"[\u4e00-\u9fa5A-Za-z]{2,20}(警方|检方|法院|媒体)",
            ],
        }

    def train_ner_model(self, train_texts: Sequence[str], valid_texts: Sequence[str]) -> Path:
        weak_patterns = self.build_weak_patterns()
        train_dataset = WeakChineseNERDataset(train_texts, self.tokenizer, self.args.max_text_length, weak_patterns)
        valid_dataset = WeakChineseNERDataset(valid_texts, self.tokenizer, self.args.max_text_length, weak_patterns)
        train_loader = DataLoader(train_dataset, batch_size=self.args.ner_batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.args.ner_batch_size, shuffle=False)

        model = AutoModelForTokenClassification.from_pretrained(
            self.args.bert_model_name,
            num_labels=len(ENTITY_LABELS),
            id2label=self.id2label,
            label2id=self.label2id,
            local_files_only=self.args.local_files_only,
        ).to(self.args.device)
        optimizer = AdamW(model.parameters(), lr=self.args.ner_lr)

        best_loss = float("inf")
        save_dir = self.processed_dir / "ner_model"
        save_dir.mkdir(exist_ok=True)

        for epoch in range(self.args.ner_epochs):
            model.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, desc=f"NER Train {epoch + 1}"):
                batch = {key: value.to(self.args.device) for key, value in batch.items()}
                output = model(**batch)
                optimizer.zero_grad()
                output.loss.backward()
                optimizer.step()
                train_loss += output.loss.item()

            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(valid_loader, desc=f"NER Valid {epoch + 1}"):
                    batch = {key: value.to(self.args.device) for key, value in batch.items()}
                    output = model(**batch)
                    valid_loss += output.loss.item()

            avg_valid_loss = valid_loss / max(len(valid_loader), 1)
            avg_train_loss = train_loss / max(len(train_loader), 1)
            print(f"NER epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, valid_loss={avg_valid_loss:.4f}")
            if avg_valid_loss <= best_loss:
                best_loss = avg_valid_loss
                model.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)
        return save_dir

    def decode_entities(self, text: str, model, tokenizer) -> List[str]:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=self.args.max_text_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = encoded.pop("offset_mapping").squeeze(0)
        encoded = {key: value.to(self.args.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits.squeeze(0)
        pred_ids = logits.argmax(dim=-1).cpu().tolist()

        entities = []
        current_tokens = []
        current_type = None
        for pred_id, (start, end) in zip(pred_ids, offset_mapping.tolist()):
            if start == end:
                continue
            label = self.id2label[pred_id]
            token_text = text[start:end]
            if label == "O":
                if current_tokens and current_type in ENTITY_TYPES:
                    entities.append("".join(current_tokens))
                current_tokens = []
                current_type = None
                continue
            prefix, entity_type = label.split("-", 1)
            if prefix == "B" or entity_type != current_type:
                if current_tokens and current_type in ENTITY_TYPES:
                    entities.append("".join(current_tokens))
                current_tokens = [token_text]
                current_type = entity_type
            else:
                current_tokens.append(token_text)
        if current_tokens and current_type in ENTITY_TYPES:
            entities.append("".join(current_tokens))

        deduped = []
        seen = set()
        for entity in entities:
            entity = entity.strip()
            if entity and entity not in seen and len(entity) <= self.args.max_entity_chars:
                deduped.append(entity)
                seen.add(entity)
        return deduped[: self.args.max_entities]

    def export_multimodal_text_inputs(self, all_df: pd.DataFrame, ner_model_dir: Path):
        ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_dir, local_files_only=True).to(self.args.device)
        ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_dir, local_files_only=True)
        ner_model.eval()

        entity_tokens_dic = {}
        text_input_dic = {}
        entity_texts = {}

        for _, row in tqdm(all_df.iterrows(), total=len(all_df), desc="Export text inputs"):
            post_id = row["post_id"]
            text = self.clean_text(str(row["original_post"]))
            entities = self.decode_entities(text, ner_model, ner_tokenizer)

            clip_text_inputs = self.clip_processor(
                text=[text],
                padding="max_length",
                truncation=True,
                max_length=self.args.max_text_length,
                return_tensors="pt",
            )
            text_input_dic[post_id] = clip_text_inputs["input_ids"].squeeze(0)

            if entities:
                entity_inputs = self.clip_processor(
                    text=entities,
                    padding="max_length",
                    truncation=True,
                    max_length=self.args.max_entity_length,
                    return_tensors="pt",
                )
                entity_tokens_dic[post_id] = entity_inputs["input_ids"]
            else:
                entity_tokens_dic[post_id] = torch.zeros((0, self.args.max_entity_length), dtype=torch.long)
            entity_texts[post_id] = entities

        np.save(self.processed_dir / "word_clipinputs.npy", text_input_dic)
        np.save(self.processed_dir / "entity_tokens.npy", entity_tokens_dic)
        with open(self.processed_dir / "entity_texts.json", "w", encoding="utf-8") as handle:
            json.dump(entity_texts, handle, ensure_ascii=False, indent=2)

    def run(self):
        raw_df = self.load_raw_dataframe()
        paired_df = self.build_paired_dataframe(raw_df)
        train_df, valid_df, test_df = self.split_dataframe(paired_df)
        ner_model_dir = self.train_ner_model(train_df["original_post"].tolist(), valid_df["original_post"].tolist())
        all_df = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=True)
        self.export_multimodal_text_inputs(all_df, ner_model_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end text preprocessing for Weibo fake news detection.")
    # ========== 修改点：默认路径改为 data/weibo 和 data/weibo/processed ==========
    parser.add_argument("--data_dir", type=str, default="data/weibo", help="Directory containing tweets/ and images")
    parser.add_argument("--processed_dir", type=str, default="data/weibo/processed", help="Output directory for processed files")
    # ===========================================================================
    parser.add_argument("--bert_model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--clip_model_name", type=str, default="OFA-Sys/chinese-clip-vit-base-patch16")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--split_mode", type=str, default="official", choices=["official", "random"])
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--max_text_length", type=int, default=200)
    parser.add_argument("--max_entity_length", type=int, default=20)
    parser.add_argument("--max_entities", type=int, default=15)
    parser.add_argument("--max_entity_chars", type=int, default=20)
    parser.add_argument("--ner_batch_size", type=int, default=8)
    parser.add_argument("--ner_epochs", type=int, default=3)
    parser.add_argument("--ner_lr", type=float, default=3e-5)
    parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    TextPreprocessor(args).run()