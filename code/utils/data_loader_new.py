import torch
from torch.utils.data import Dataset

class FakeNewsDataset(Dataset):
    def __init__(self, data_df, crop_num, st_num, dataset, entity_tokens, crop_input, text_input):
        self.data_df = data_df
        self.crop_num = crop_num
        self.st_num = st_num
        self.dataset = dataset
        self.entity_tokens = entity_tokens
        self.crop_input = crop_input
        self.text_input = text_input

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        post_id = self.data_df['post_id'][idx]
        image_id = self.data_df['image_id'][idx]
        label = self.data_df['label'][idx]
        label = torch.tensor(label, dtype=torch.long)

        entity_tokens = self.entity_tokens[post_id]          # shape: [num_entities, 20]
        # 填充到固定长度（最多15个实体）
        MAX_ENTITIES = 15
        cur_num = entity_tokens.shape[0]
        if cur_num < MAX_ENTITIES:
            pad = torch.zeros(MAX_ENTITIES - cur_num, entity_tokens.shape[1], dtype=entity_tokens.dtype)
            entity_tokens = torch.cat([entity_tokens, pad], dim=0)
        else:
            entity_tokens = entity_tokens[:MAX_ENTITIES]

        crop_input = self.crop_input[image_id]               # shape: [crop_num, 3, 224, 224]
        text_input = self.text_input[post_id]                # shape: [max_len]

        sample = {
            'post_id': post_id,
            'label': label,
            'crop_input': crop_input,
            'entity_tokens': entity_tokens,
            'text_input': text_input,
        }
        return sample