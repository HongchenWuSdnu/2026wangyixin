import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from modular_models import MultimodalFakeNewsDetector
from utils.modular_data_loader import FakeNewsDataset
from utils.train_eval_helper import set_random_seed


class CachedFeatureDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Modular multimodal fake news detection with BERT, NER, Faster R-CNN regions and CLIP alignment."
    )
    parser.add_argument("--dataset", type=str, default="weibo")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--conv_out", type=int, default=64)
    parser.add_argument("--crop_num", type=int, default=6)
    parser.add_argument("--st_num", type=int, default=16)
    parser.add_argument("--max_entities", type=int, default=15)
    parser.add_argument("--dropout_p", type=float, default=0.3)
    parser.add_argument("--layer_num", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--conv_kernel", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--name", type=str, default="modular_mm_fake_news")
    # ========== 修改点：默认路径与预处理输出一致 ==========
    parser.add_argument("--processed_dir", type=str, default="data/weibo/processed")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    # ====================================================
    parser.add_argument("--bert_model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--clip_model_name", type=str, default="OFA-Sys/chinese-clip-vit-base-patch16")
    parser.add_argument("--shared_feature_dim", type=int, default=256)
    parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--precompute_backbone", action="store_true")
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--tune_threshold", action="store_true")
    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--freeze_clip_text", action="store_true")
    parser.add_argument("--freeze_clip_image", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--lr_scheduler", action="store_true")
    # ========== Ablation Study Parameters ==========
    parser.add_argument("--ablation_mode", type=str, default="full",
        choices=["full", "no_entity", "no_cross_modal", "no_conflict", "lite"],
        help="Ablation mode: full (baseline), no_entity, no_cross_modal, no_conflict, lite")
    # ====================================================
    return parser.parse_args()


def resolve_device(device_name: str) -> str:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device_name


def load_split_dataframe(processed_dir: Path, split: str) -> pd.DataFrame:
    split_path = processed_dir / f"{split}_EANN_frozen.npy"
    df_columns = ["original_post", "label", "image_id", "post_id"]
    data = np.load(split_path, allow_pickle=True)
    return pd.DataFrame(data, columns=df_columns)


def load_data_objects(processed_dir: Path):
    train_df = load_split_dataframe(processed_dir, "train")
    valid_df = load_split_dataframe(processed_dir, "valid")
    test_df = load_split_dataframe(processed_dir, "test")
    entity_tokens = np.load(processed_dir / "entity_tokens.npy", allow_pickle=True).item()
    crop_input = None  # 动态加载，不预存
    text_input = np.load(processed_dir / "word_clipinputs.npy", allow_pickle=True).item()
    return train_df, valid_df, test_df, entity_tokens, crop_input, text_input


def build_loader(data_df, args, entity_tokens, crop_input, text_input, shuffle):
    dataset = FakeNewsDataset(
        data_df=data_df,
        crop_num=args.crop_num,
        st_num=args.st_num,
        dataset=args.dataset,
        entity_tokens=entity_tokens,
        crop_input=crop_input,
        text_input=text_input,
        max_entities=args.max_entities,
        processed_dir=args.processed_dir,
        clip_model_name=args.clip_model_name,
        local_files_only=args.local_files_only,
    )
    sampler = None
    if shuffle and args.use_weighted_sampler:
        label_counts = data_df["label"].value_counts().to_dict()
        sample_weights = data_df["label"].map(lambda label: 1.0 / label_counts[label]).tolist()
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, num_workers=args.num_workers)


def build_cached_loader(samples, args, shuffle):
    return DataLoader(CachedFeatureDataset(samples), batch_size=args.batch_size, shuffle=shuffle, num_workers=0)


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def precompute_split_features(model, loader, device, split_name):
    cached_samples = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Cache {split_name}"):
            batch = move_batch_to_device(batch, device)
            global_text_feature, entity_features, image_features = model.encode_modalities(batch)
            labels = batch["label"]
            for idx in range(labels.size(0)):
                cached_samples.append(
                    {
                        "global_text_feature": global_text_feature[idx].cpu(),
                        "entity_features": entity_features[idx].cpu(),
                        "image_features": image_features[idx].cpu(),
                        "label": labels[idx].cpu(),
                    }
                )
    return cached_samples


def build_loss_fn(train_df, args, device):
    if not args.use_class_weights:
        return None
    weights = torch.tensor([1.5, 1.0], dtype=torch.float32, device=device)
    print(f"Using custom class weights: Real=1.5, Fake=1.0")
    return weights


def predict_from_output(output, threshold=0.5):
    positive_prob = output[:, 1].exp()
    return (positive_prob >= threshold).long(), positive_prob


def run_epoch(model, loader, device, optimizer=None, use_cached_backbone=False, loss_weights=None, threshold=0.5):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in tqdm(loader):
            batch = move_batch_to_device(batch, device)
            if use_cached_backbone:
                output = model.forward_from_encoded(
                    batch["global_text_feature"],
                    batch["entity_features"],
                    batch["image_features"],
                )
            else:
                output = model(batch)
            loss = F.nll_loss(output, batch["label"], weight=loss_weights)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds, probs = predict_from_output(output, threshold=threshold)
            total_loss += loss.item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    # ========== 计算AUC ==========
    all_probs_np = np.array(all_probs)
    all_labels_np = np.array(all_labels)
    try:
        auc = roc_auc_score(all_labels_np, all_probs_np)
    except:
        auc = 0.0
    # ============================

    return avg_loss, acc, macro_f1, auc, all_labels, all_preds, all_probs


def tune_best_threshold(model, loader, device, use_cached_backbone=False):
    _, _, _, _, labels, _, probs = run_epoch(
        model,
        loader,
        device,
        optimizer=None,
        use_cached_backbone=use_cached_backbone,
        threshold=0.5,
    )
    labels = np.array(labels)
    probs = np.array(probs)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.1, 0.91, 0.05):
        preds = (probs >= threshold).astype(int)
        score = f1_score(labels, preds, average="macro")
        if score > best_f1:
            best_f1 = score
            best_threshold = float(round(threshold, 2))
    return best_threshold, best_f1


def evaluate_and_report(model, loader, device, split_name, use_cached_backbone=False, threshold=0.5):
    loss, acc, macro_f1, labels, preds, _ = run_epoch(
        model,
        loader,
        device,
        optimizer=None,
        use_cached_backbone=use_cached_backbone,
        threshold=threshold,
    )
    print(f"{split_name} Loss: {loss:.4f} | Acc: {acc:.4f} | Macro-F1: {macro_f1:.4f}")
    print(classification_report(labels, preds, digits=4, zero_division=0))
    return acc


def main():
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.device = resolve_device(args.device)
    set_random_seed(args.seed)

    processed_dir = Path(args.processed_dir)
    save_dir = Path(args.save_dir) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # ========== Log Ablation Mode ==========
    print(f"{'='*60}")
    print(f"Ablation Mode: {args.ablation_mode}")
    print(f"Model Name: {args.name}")
    print(f"{'='*60}")
    # ====================================================

    train_df, valid_df, test_df, entity_tokens, crop_input, text_input = load_data_objects(processed_dir)
    train_loader = build_loader(train_df, args, entity_tokens, crop_input, text_input, shuffle=True)
    valid_loader = build_loader(valid_df, args, entity_tokens, crop_input, text_input, shuffle=False)
    test_loader = build_loader(test_df, args, entity_tokens, crop_input, text_input, shuffle=False)

    model = MultimodalFakeNewsDetector(args).to(args.device)
    use_cached_backbone = args.precompute_backbone

    if use_cached_backbone:
        for parameter in model.text_encoder.parameters():
            parameter.requires_grad = False
        for parameter in model.visual_encoder.parameters():
            parameter.requires_grad = False
        train_loader = build_cached_loader(precompute_split_features(model, train_loader, args.device, "train"), args, shuffle=True)
        valid_loader = build_cached_loader(precompute_split_features(model, valid_loader, args.device, "valid"), args, shuffle=False)
        test_loader = build_cached_loader(precompute_split_features(model, test_loader, args.device, "test"), args, shuffle=False)

    optimizer = optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5, verbose=True
        )

    loss_weights = build_loss_fn(train_df, args, args.device)

    best_val_acc = 0.0
    best_model_path = save_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1, _, _, _ = run_epoch(
            model,
            train_loader,
            args.device,
            optimizer=optimizer,
            use_cached_backbone=use_cached_backbone,
            loss_weights=loss_weights,
        )
        val_loss, val_acc, val_f1, _, _, _ = run_epoch(
            model,
            valid_loader,
            args.device,
            optimizer=None,
            use_cached_backbone=use_cached_backbone,
        )
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
        )

        if scheduler is not None:
            scheduler.step(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print(f"Best validation accuracy: {best_val_acc:.4f}")

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=args.device))
    best_threshold = 0.5
    if args.tune_threshold:
        best_threshold, best_threshold_f1 = tune_best_threshold(
            model,
            valid_loader,
            args.device,
            use_cached_backbone=use_cached_backbone,
        )
        print(f"Best validation threshold: {best_threshold:.2f} | Macro-F1: {best_threshold_f1:.4f}")
    evaluate_and_report(
        model,
        test_loader,
        args.device,
        "Test",
        use_cached_backbone=use_cached_backbone,
        threshold=best_threshold,
    )


if __name__ == "__main__":
    main()