基于多层次语义一致性建模的多模态虚假新闻检测

---
项目简介：
本项目是一个基于多层次语义一致性建模的多模态虚假新闻检测模型。
在社交媒体时代，谣言通常以文本+图像的多模态形式传播。仅依赖单一模态（文本或图像）容易被误导。本项目通过深度学习技术，同时分析文本和图像内容，利用两种模态之间的内容相关性来提高虚假新闻检测的准确性。
---
主要工作和改进点：
本文提出一种基于多层次语义一致性建模的多模态虚假新闻检测方法，将图文语义一致性建模分解为三个递进层次：
第一层次——特征级对齐：图像侧利用Faster R-CNN提取显著目标区域，使用Chinese-CLIP图像编码器获得区域级视觉语义特征；文本侧通过线性投影将BERT文本特征映射到与视觉特征相同维度的表示空间；后续跨模态匹配关系由跨模态交互模块和训练过程共同学习。
第二层次——交互级对齐：通过双向跨模态Transformer实现文本与图像的深层双向语义交互，捕捉模态间的复杂关联和语义对应关系
第三层次——判别级量化：构建实体-对象相似度矩阵，提取一致性特征与冲突统计特征；将语义不一致显式转化为可量化的判别信号；融合多维特征输入分类器完成判别。

代码组织结构
```
code/
├── main.py                    # 主训练脚本（入口）
├── models.py                  # 核心模型定义
├── modular_models.py          # 模块化模型实现
├── process_text_weibo.py      # Weibo文本预处理
├── process_image_weibo.py     # Weibo图像预处理
└── utils/
    ├── data_loader_new.py           # 数据加载器
    ├── modular_data_loader.py       # 模块化数据加载器
    └── train_eval_helper.py         # 训练/评估助手函数
data/
└── weibo/                     # Weibo 数据集相关文件
├── stop_words.txt         # 中文停用词表
├── processed/             # 处理后的文本数据与数据划分文件
│   ├── train.csv          # 训练集
│   ├── test.csv           # 测试集
│   └── valid.csv          # 验证集
├── nonrumor_images_sample/ # 真实新闻图片示例
└── rumor_images_sample/    # 虚假新闻图片示例
scripts/
└── analyze_ablation.py         # 消融实验分析脚本
```
各模块说明：
- models.py：定义模型架构（包含文本编码器、图像编码器、跨模态Transformer等）
- modular_models.py：便于消融实验的模块化实现
- utils/train_eval_helper.py：包含训练循环、评估函数、损失函数定义
- process_*.py：数据预处理，生成可直接用于训练的格式
- 由于处理后的完整图片数据文件体积较大，且原始 Weibo 多模态虚假新闻数据集可能存在再分发限制，本仓库未上传完整图片数据集。完整实验中使用的图片数据需要根据原始 Weibo 多模态虚假新闻数据集来源获取，并按照本仓库提供的预处理脚本进行处理。
---

使用方法：

第1步：环境配置

硬件要求
- 操作系统：Ubuntu 22.04（或其他Linux/Windows）
- GPU：NVIDIA RTX 4090（或其他CUDA计算能力≥7.0的GPU）
- CUDA版本：12.1
- cuDNN：8.x 及以上

软件环境安装

1. 克隆仓库
   ```bash
   git clone https://github.com/your-username/2026wangxinyi.git
   cd 2026wangxinyi
   ```

2. 创建Python环境
   ```bash
   conda create -n mmfnd python=3.8
   conda activate mmfnd
   ```

3. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

依赖包版本：
```
torch==2.1.0
torchvision==0.16.0
torchtext==0.16.0
transformers==4.25.1
clip==1.0
cn_clip==1.5.1
opencv_python==4.7.0.68
pandas~=3.0.2
Pillow==9.5.0
scikit-learn~=1.8.0
```

第2步：数据预处理

在运行模型前，需要预处理Weibo数据：

```bash
cd code/

# 第一步：文本预处理
python process_text_weibo.py --input_dir ../data/weibo --output_dir ../data/weibo/processed

# 第二步：图像预处理
python process_image_weibo.py --input_dir ../data/weibo --output_dir ../data/weibo/processed
```

预处理输出：清理后的文本特征、预处理后的图像特征、生成训练/验证/测试集划分

第3步：模型训练

```bash
cd code/

# 运行完整模型训练
python main.py \
    --data_dir ../data/weibo/processed \
    --checkpoint_dir ../checkpoints \
    --epochs 8 \
    --batch_size 20 \
    --learning_rate 2e-5
```

第4步：消融实验（可选）

```bash
# 修改 main.py 中的 ABLATION_MODE 参数
# "full" - 完整模型
# "no_entity" - 移除实体编码
# "no_cross_modal" - 移除跨模态交互
# "no_conflict" - 移除冲突编码
# "lite" - 简化融合

python main.py --ablation_mode no_entity
python main.py --ablation_mode no_cross_modal
# ...
```

第5步：分析结果

```bash
cd ../scripts/

# 自动生成消融实验对比分析
python analyze_ablation.py --checkpoints_dir ../checkpoints --output_dir ../results/ablation
```
---

数据集：本项目使用 Weibo虚假新闻数据集：(https://doi.org/10.6084/m9.figshare.28516655)

---

Weibo数据集结果

| 模型 | 准确率 (Accuracy) | 宏平均F1 (Macro-F1) | 损失 (Loss) | 说明 |
|------|:---:|:---:|:---:|---|
| Baseline (Full) | 0.9312 | 0.9253 | 0.2395 | 完整模型，最优 |
| No Cross-Modal | 0.9329 | 0.9257 | 0.3406 | 移除跨模态交互 |
| No Conflict | 0.9312 | 0.9237 | 0.2350 | 移除冲突编码 |
| No Entity | 0.9191 | 0.9094 | 0.3031 | 移除实体编码 |
| Lite (Simple) | 0.9165 | 0.9080 | 0.4683 | 简化融合 |

关键发现

整体性能优异
- 在Weibo数据集上达到93.12%的准确率和92.53%的宏平均F1
- 对虚假新闻检测的泛化能力强

---

更多说明

项目引用
```bibtex
@article{Qiao2025MMFND,
  title={Improving multimodal fake news detection by leveraging cross-modal content correlation},
  author={Jiao Qiao, Xianghua Li, Chao Gao, Lianwei Wu, Junwei Feng, Zhen Wang},
  journal={Information Processing and Management},
  year={2025},
  volume={62},
  issue={5},
  pages={104120}
}
```

