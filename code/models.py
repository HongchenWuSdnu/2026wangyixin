import torch
import torch.nn as nn
from transformers import ChineseCLIPModel
import torch.nn.functional as F

CNNSim2_param = {
    'st_fc1_in': 512,
    'st_fc1_out': 256,
    'st_fc2_out': 128,
    'st_fc3_out': 64,
    'consis_fc1_out': 128,
    'consis_fc2_out': 64,
    'fusion_fc1_out': 64,
    'fusion_fc2_out': 32
}

class TransformerEncoder(nn.Module):
    """ general attention for tgt & src from different modality
    """
    def __init__(self, model_dim, layer_num, head, tgt_seq, src_seq):
        super(TransformerEncoder, self).__init__()
        self.layer_num = layer_num
        self.multihead_attns_t = nn.ModuleList([nn.MultiheadAttention(model_dim, head) for _ in range(self.layer_num)])
        self.multihead_attns_s = nn.ModuleList([nn.MultiheadAttention(model_dim, head) for _ in range(self.layer_num)])
        self.LN_ts1 = nn.ModuleList([nn.LayerNorm([tgt_seq, model_dim]) for _ in range(self.layer_num)])
        self.LN_ss1 = nn.ModuleList([nn.LayerNorm([src_seq, model_dim]) for _ in range(self.layer_num)])
        self.LN_ts2 = nn.ModuleList([nn.LayerNorm([tgt_seq, model_dim]) for _ in range(self.layer_num)])
        self.LN_ss2 = nn.ModuleList([nn.LayerNorm([src_seq, model_dim]) for _ in range(self.layer_num)])
        FF_K = 4
        self.ff_ts = nn.ModuleList([nn.Sequential(nn.Linear(model_dim, FF_K * model_dim),
                                                  nn.ReLU(),
                                                  nn.Linear(FF_K * model_dim, model_dim)) for _ in range(self.layer_num)])
        self.ff_ss = nn.ModuleList([nn.Sequential(nn.Linear(model_dim, FF_K * model_dim),
                                                  nn.ReLU(),
                                                  nn.Linear(FF_K * model_dim, model_dim)) for _ in range(self.layer_num)])

    def forward(self, tgt, src):
        for multihead_attn_t, multihead_attn_s, ff_t, ff_s, LN_t1, LN_s1, LN_t2, LN_s2 in zip(self.multihead_attns_t, self.multihead_attns_s, self.ff_ts, self.ff_ss, self.LN_ts1, self.LN_ss1, self.LN_ts2, self.LN_ss2):
            res_t = tgt  # [B, Seq, D]
            res_s = src
            tgt = tgt.permute(1, 0, 2)  # [Seq, B, D]
            src = src.permute(1, 0, 2)  # [Seq, B, D]
            tgt_new, _ = multihead_attn_t(tgt, src, src)
            src_new, _ = multihead_attn_s(src, tgt, tgt)
            tgt_new = tgt_new.permute(1, 0, 2)  # [B, Seq, D]
            src_new = src_new.permute(1, 0, 2)  # [B, Seq, D]
            tgt_new = LN_t1(tgt_new + res_t)
            src_new = LN_s1(src_new + res_s)
            res_t = tgt_new
            res_s = src_new
            tgt_new = ff_t(tgt_new)
            src_new = ff_s(src_new)
            tgt = LN_t2(tgt_new + res_t)
            src = LN_s2(src_new + res_s)
        return tgt, src


class C3N(nn.Module):
    def __init__(self, args):
        super(C3N, self).__init__()
        self.is_weibo = (args.dataset == 'weibo')
        if self.is_weibo:
            self.clip_model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
            self.clip_model = self.clip_model.to(args.device)
            self.device = args.device

        Ks_word = args.conv_kernel
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=args.conv_out,
                                              kernel_size=(K, args.crop_num)) for K in Ks_word])
        self.logit_scale = 100
        Conv_out = len(Ks_word) * args.conv_out
        self.transformer = TransformerEncoder(model_dim=512, layer_num=args.layer_num, head=8,
                                              tgt_seq=args.st_num, src_seq=args.crop_num)
        self.fc_st1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(),
                                    nn.Linear(256, 128), nn.ReLU(),
                                    nn.Linear(128, 64), nn.ReLU())
        self.fc_consis1 = nn.Sequential(nn.Linear(Conv_out, 128), nn.ReLU(),
                                        nn.Linear(128, 64), nn.ReLU())
        self.fc_ob1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(),
                                    nn.Linear(256, 128), nn.ReLU(),
                                    nn.Linear(128, 64), nn.ReLU())
        # 冲突量化层
        self.fc_conflict = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Dropout(args.dropout_p))
        fusion_input_dim = 64 + 64*2 + 32   # consis + st + ob + conflict
        self.fusion = nn.Sequential(nn.Linear(fusion_input_dim, 128), nn.ReLU(),
                                    nn.Linear(128, 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU())
        self.fc = nn.Linear(32, 2)
        self.dropout = nn.Dropout(args.dropout_p)

    def similarity_weight(self, txt_fea, img_fea):
        txt_fea_ = txt_fea / txt_fea.norm(dim=-1, keepdim=True)
        img_fea_ = img_fea / img_fea.norm(dim=-1, keepdim=True)
        img_fea_T = torch.transpose(img_fea_, 1, 2)
        sim = torch.matmul(txt_fea_, img_fea_T)
        sim = (self.logit_scale * sim).unsqueeze(1)
        fea_maps = [F.relu(conv(sim)).squeeze(3) for conv in self.convs]
        consis_fea_avg = [F.avg_pool1d(i, i.shape[2]).squeeze(2) for i in fea_maps]
        consis_fea_avg = torch.cat(consis_fea_avg, 1)

        sim_raw = sim.squeeze(1)
        max_sim_per_entity = torch.max(sim_raw, dim=2)[0]
        mean_sim = sim_raw.mean(dim=(1,2))
        missing_ratio = (max_sim_per_entity < 0.3).float().mean(dim=1)
        var_max_sim = torch.var(max_sim_per_entity, dim=1)
        max_sim_overall = torch.max(max_sim_per_entity, dim=1)[0]
        conflict_stats = torch.stack([mean_sim, missing_ratio, var_max_sim, max_sim_overall], dim=1)

        st_embed_pos = txt_fea[:, 0, :]
        ob_embed_pos = img_fea[:, 0, :]
        return st_embed_pos, ob_embed_pos, consis_fea_avg, conflict_stats

    def clip_encode(self, text_input, crop_input, entity_tokens):
        # text_input: [B, L]
        text_attn = (text_input != 0).long()
        sentence_features = self.clip_model.get_text_features(input_ids=text_input, attention_mask=text_attn)
        B, num_entities, seq_len = entity_tokens.shape
        entity_tokens_flat = entity_tokens.view(-1, seq_len)
        entity_attn = (entity_tokens_flat != 0).long()
        word_features = self.clip_model.get_text_features(input_ids=entity_tokens_flat, attention_mask=entity_attn)
        word_features = word_features.view(B, num_entities, -1)
        word_features = torch.cat([sentence_features.unsqueeze(1), word_features], dim=1)

        num_crops = crop_input.shape[1]
        crop_input_flat = crop_input.view(-1, 3, 224, 224)
        crop_features = self.clip_model.get_image_features(pixel_values=crop_input_flat)
        crop_features = crop_features.view(B, num_crops, -1)
        return word_features, crop_features

    def forward(self, data):
        text_input = data['text_input']          # [B, L]
        crop_input = data['crop_input']          # [B, num_crops, 3, 224, 224]
        if self.is_weibo:
            word_features, crop_features = self.clip_encode(text_input, crop_input, data['entity_tokens'])
        else:
            # twitter 部分暂不实现，此处仅作占位
            word_features = torch.cat([text_input.unsqueeze(1), data['n_word_input']], dim=1)
            crop_features = crop_input
        wi_fea, iw_fea = self.transformer(word_features, crop_features)
        st, ob, consis, conflict_stats = self.similarity_weight(wi_fea, iw_fea)

        st = self.dropout(self.fc_st1(st))
        ob = self.dropout(self.fc_ob1(ob))
        consis = self.dropout(self.fc_consis1(consis))
        conflict_feat = self.dropout(self.fc_conflict(conflict_stats))
        combined = torch.cat([st, ob, consis, conflict_feat], dim=-1)
        fused = self.dropout(self.fusion(combined))
        logit = F.log_softmax(self.fc(fused), dim=-1)
        return logit