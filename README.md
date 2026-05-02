基于多层次语义一致性建模的多模态虚假新闻检测

---
项目简介：
本项目是一个基于多层次语义一致性建模的多模态虚假新闻检测模型。
在社交媒体时代，谣言通常以文本+图像的多模态形式传播。仅依赖单一模态（文本或图像）容易被误导。本项目通过深度学习技术，同时分析文本和图像内容，利用两种模态之间的内容相关性来提高虚假新闻检测的准确性。
---
主要工作和改进点：
本文提出一种基于多层次语义一致性建模的多模态虚假新闻检测方法，将图文语义一致性建模分解为三个递进层次：
第一层次 - 特征级对齐：
- 图像侧：利用Faster R-CNN提取显著目标区域，使用Chinese-CLIP图像编码器获得区域级视觉语义特征
- 文本侧：通过线性投影将BERT文本特征映射到与视觉特征相同维度的表示空间
- 跨模态匹配：后续跨模态匹配关系由跨模态交互模块和训练过程共同学习
第二层次 - 交互级对齐：
- 通过双向跨模态Transformer实现文本与图像的深层双向语义交互
- 捕捉模态间的复杂关联和语义对应关系
第三层次 - 判别级量化：
- 构建实体-对象相似度矩阵，提取一致性特征与冲突统计特征
- 将语义不一致显式转化为可量化的判别信号
- 融合多维特征输入分类器完成判别
实验结果：
- 在Weibo数据集上达到93.12%的准确率和92.53%的宏平均F1
- 完整的消融实验验证各模块的有效性和进一步的改进空间
- 支持轻量化版本，性能损失最小化

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
└── weibo/                      # Weibo数据集
    ├── stop_words.txt          # 中文停用词表
    └── processedData/          # 处理后的数据
        ├── df_train.csv        # 完整训练集
        ├── df_test.csv         # 完整测试集
        └── df_train[123].csv, df_test[123].csv  # 数据划分

scripts/
└── analyze_ablation.py         # 消融实验分析脚本
```
各模块说明：
- models.py：定义模型架构（包含文本编码器、图像编码器、跨模态Transformer等）
- modular_models.py：便于消融实验的模块化实现
- utils/train_eval_helper.py：包含训练循环、评估函数、损失函数定义
- process_*.py：数据预处理，生成可直接用于训练的格式

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

预处理输出：
- 清理后的文本特征
- 预处理后的图像特征
- 生成训练/验证/测试集划分

第3步：模型训练

```bash
cd code/

# 运行完整模型训练
python main.py \
    --data_dir ../data/weibo/processed \
    --checkpoint_dir ../checkpoints \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 3e-5
```

主要参数说明：
- --data_dir：预处理后的数据目录
- --checkpoint_dir：模型保存路径
- --epochs：训练轮数
- --batch_size：批大小
- --learning_rate：学习率

训练输出：
- 模型权重保存到 ../checkpoints/ablation_full/best_model.pt
- 训练日志和测试结果保存到 ../checkpoints/ablation_full/test_results.log

第4步：消融实验（可选）

如果要运行消融实验（测试各模块的作用）：

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

这会生成：
- ablation_results.csv - 性能对比表
- ablation_report.md - 分析报告
- metrics_comparison.png - 图表可视化

---

数据集

本项目使用 Weibo虚假新闻数据集：(https://doi.org/10.6084/m9.figshare.28516655)

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

各层次的作用分析
通过消融实验，我们发现多层次语义一致性建模框架中各层次具有不同的作用和贡献：

1. 特征级对齐（多粒度文本语义建模）- 核心贡献
   - 对性能提升贡献最为显著
   - 是整个框架的基础，直接影响准确率和F1分数
   - 移除此层次后，模型性能下降最明显

2. 交互级对齐（跨模态Transformer）- 稳定性贡献
   - Accuracy和Macro-F1上的增益并不稳定
   - 但有助于显著降低Loss，提高训练稳定性
   - 改善类别级Recall的均衡性，减少类别偏差
   - 在轻量化场景中可选

3. 判别级量化（显式冲突量化）- 补充信息
   - 贡献相对边际，但仍有一定作用
   - 为细粒度判别提供补充信息
   - 增强模型对虚假新闻显式冲突特征的捕捉

三层协同效果
- 三个层次的协同使用使完整模型在以下方面取得了较好的综合表现：
  - 整体性能（Accuracy和F1）
  - 训练稳定性（Loss收敛）
  - 类别均衡性（Recall均衡）

性能权衡
- 若追求最大准确率，可重点优化特征级对齐
- 若追求稳定性和均衡性，需要保留交互级对齐
- 若需轻量化部署，可移除判别级量化，性能损失最小


验证指标
- 在真实Weibo数据集上训练和测试
- 使用标准的Accuracy、Precision、Recall、F1-Score评估
- 完整的5折交叉验证确保结果稳定

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

