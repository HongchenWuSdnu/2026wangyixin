#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3N 消融实验结果分析脚本
=====================

功能：
1. 从checkpoints目录收集所有消融实验的结果
2. 生成JSON格式的结果数据库
3. 创建可视化对比表（CSV + PNG图表）
4. 生成Markdown格式的分析报告

使用方法：
    python scripts/analyze_ablation.py
    python scripts/analyze_ablation.py --results_dir results/ablation
    python scripts/analyze_ablation.py --resume  # 只生成表格和图表，不重新提取
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# ============================================================================
# 配置
# ============================================================================

ABLATION_MODES = {
    "ablation_full": {
        "name": "Baseline (Full)",
        "description": "完整模型（所有模块启用）",
        "order": 0,
    },
    "ablation_no_entity": {
        "name": "No Entity",
        "description": "移除实体编码",
        "order": 1,
    },
    "ablation_no_cross_modal": {
        "name": "No Cross-Modal",
        "description": "移除跨模态Transformer交互",
        "order": 2,
    },
    "ablation_no_conflict": {
        "name": "No Conflict",
        "description": "移除冲突编码器",
        "order": 3,
    },
    "ablation_lite": {
        "name": "Lite (Simple)",
        "description": "简化融合（无实体+无跨模态+无冲突）",
        "order": 4,
    },
}

# ============================================================================
# 工具函数
# ============================================================================

def extract_metrics_from_log(log_file: Path) -> Dict:
    """
    从训练日志中提取最终的测试性能指标

    日志格式:
        Test Loss: 0.3421 | Acc: 0.8234 | Macro-F1: 0.8156
                 precision    recall  f1-score   support

        Fake       0.8190    0.8123    0.8156      1234
        True       0.8268    0.8345    0.8307      1345
    """
    metrics = {}

    if not log_file.exists():
        return metrics

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找Test Loss行
        for line in content.split('\n'):
            if 'Test Loss:' in line:
                parts = line.split('|')
                for part in parts:
                    if 'Loss:' in part:
                        metrics['loss'] = float(part.split(':')[1].strip())
                    elif 'Acc:' in part:
                        metrics['accuracy'] = float(part.split(':')[1].strip())
                    elif 'Macro-F1:' in part:
                        metrics['f1_macro'] = float(part.split(':')[1].strip())

            # 提取Fake类指标
            if line.strip().startswith('Fake'):
                parts = line.split()
                if len(parts) >= 5:
                    metrics['precision_fake'] = float(parts[1])
                    metrics['recall_fake'] = float(parts[2])
                    metrics['f1_fake'] = float(parts[3])

            # 提取True类指标
            if line.strip().startswith('True'):
                parts = line.split()
                if len(parts) >= 5:
                    metrics['precision_true'] = float(parts[1])
                    metrics['recall_true'] = float(parts[2])
                    metrics['f1_true'] = float(parts[3])

    except Exception as e:
        print(f"警告：无法解析 {log_file}: {e}")

    return metrics


def collect_ablation_results(checkpoints_dir: Path) -> Dict[str, Dict]:
    """
    收集所有消融实验的结果
    """
    results = {}

    for ablation_name, config in ABLATION_MODES.items():
        model_dir = checkpoints_dir / ablation_name
        log_file = model_dir / "test_results.log"

        print(f"查找 {ablation_name}...", end=" ")

        if not model_dir.exists():
            print("[ERROR] 目录不存在")
            results[ablation_name] = {
                "status": "not_found",
                "metrics": {}
            }
            continue

        metrics = extract_metrics_from_log(log_file)

        if metrics:
            print("[OK] 找到结果")
            results[ablation_name] = {
                "status": "found",
                "metrics": metrics
            }
        else:
            print("[WARN] 未找到日志文件")
            results[ablation_name] = {
                "status": "no_log",
                "metrics": {}
            }

    return results


def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    创建对比表DataFrame
    """
    data = []

    for ablation_name in sorted(ABLATION_MODES.keys(),
                               key=lambda x: ABLATION_MODES[x]['order']):
        if ablation_name not in results:
            continue

        config = ABLATION_MODES[ablation_name]
        result = results[ablation_name]
        metrics = result.get('metrics', {})

        row = {
            'Model': config['name'],
            'Description': config['description'],
            'Accuracy': metrics.get('accuracy', np.nan),
            'Macro-F1': metrics.get('f1_macro', np.nan),
            'Loss': metrics.get('loss', np.nan),
            'Precision (Fake)': metrics.get('precision_fake', np.nan),
            'Recall (Fake)': metrics.get('recall_fake', np.nan),
            'F1 (Fake)': metrics.get('f1_fake', np.nan),
            'Precision (True)': metrics.get('precision_true', np.nan),
            'Recall (True)': metrics.get('recall_true', np.nan),
            'F1 (True)': metrics.get('f1_true', np.nan),
        }
        data.append(row)

    return pd.DataFrame(data)


def calculate_contributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个模块对性能的贡献度
    贡献度 = (Baseline - Ablation) / Baseline * 100%
    """
    if df.empty or 'Accuracy' not in df.columns:
        return pd.DataFrame()

    baseline_acc = df.iloc[0]['Accuracy']
    baseline_f1 = df.iloc[0]['Macro-F1']

    contributions = []
    for idx, row in df.iterrows():
        if idx == 0:  # Skip baseline
            continue

        acc_drop = (baseline_acc - row['Accuracy']) / baseline_acc * 100 if baseline_acc > 0 else 0
        f1_drop = (baseline_f1 - row['Macro-F1']) / baseline_f1 * 100 if baseline_f1 > 0 else 0

        contributions.append({
            'Model': row['Model'],
            'Accuracy Drop (%)': acc_drop,
            'F1 Drop (%)': f1_drop,
            'Avg Impact (%)': (acc_drop + f1_drop) / 2,
        })

    return pd.DataFrame(contributions)


def plot_comparison_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    """
    绘制性能对比图表 - 针对5个核心实验优化
    """
    if df.empty:
        print("警告：数据为空，跳过绘图")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置绘图风格
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # ===== 图1: Accuracy 和 Macro-F1 对比 =====
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x_pos = np.arange(len(df))
    width = 0.35

    # Accuracy vs Macro-F1
    ax = axes[0]
    bars1 = ax.bar(x_pos - width/2, df['Accuracy'].fillna(0), width, label='Accuracy', alpha=0.8, color='#2ecc71')
    bars2 = ax.bar(x_pos + width/2, df['Macro-F1'].fillna(0), width, label='Macro-F1', alpha=0.8, color='#3498db')

    # 添加值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['Model'].tolist(), rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.6, 0.85])

    # Loss对比
    ax = axes[1]
    ax.plot(range(len(df)), df['Loss'].fillna(0), marker='o', linewidth=3, markersize=10,
            color='#e74c3c', label='Test Loss')

    # 添加值标签
    for i, (model, loss) in enumerate(zip(df['Model'], df['Loss'])):
        ax.text(i, loss + 0.01, f'{loss:.4f}', ha='center', fontsize=9)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Loss Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['Model'].tolist(), rotation=15, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] 已保存: metrics_comparison.png")


def plot_contribution_analysis(contributions_df: pd.DataFrame, output_dir: Path) -> None:
    """
    绘制各模块贡献度分析图 - 优化为5个实验
    """
    if contributions_df.empty:
        print("警告：贡献度数据为空，跳过绘图")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    x_pos = np.arange(len(contributions_df))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, contributions_df['Accuracy Drop (%)'], width,
                   label='Accuracy Drop', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x_pos + width/2, contributions_df['F1 Drop (%)'], width,
                   label='F1 Drop', alpha=0.8, color='#e74c3c')

    # 添加值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Ablated Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Drop (%)', fontsize=12, fontweight='bold')
    ax.set_title('Component Contribution Analysis\n(性能下降比例 = 模块移除后的性能下降程度)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(contributions_df['Model'].tolist(), rotation=15, ha='right', fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / 'contribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] 已保存: contribution_analysis.png")


def generate_markdown_report(df: pd.DataFrame, contributions_df: pd.DataFrame,
                            output_path: Path) -> None:
    """
    生成Markdown格式的分析报告
    """
    report = []
    report.append("# C3N 消融实验分析报告\n\n")
    report.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report.append(f"实验数量: {len(df)} 个核心实验\n\n")

    # 1. 实验概览
    report.append("## 1. 实验概览\n\n")
    report.append(df[['Model', 'Description']].to_markdown(index=False))
    report.append("\n\n")

    # 2. 主要性能指标
    report.append("## 2. 主要性能指标\n\n")
    report.append(df[['Model', 'Accuracy', 'Macro-F1', 'Loss']].to_markdown(index=False))
    report.append("\n\n")

    # 3. 详细指标（所有字段）
    report.append("## 3. 详细性能指标\n\n")
    report.append(df.to_markdown(index=False))
    report.append("\n\n")

    # 4. 贡献度分析
    if not contributions_df.empty:
        report.append("## 4. 模块贡献度分析\n\n")
        report.append("**贡献度定义**: 模块移除后的性能下降比例 = (Baseline性能 - 移除后性能) / Baseline性能 × 100%\n\n")
        report.append(contributions_df.to_markdown(index=False))
        report.append("\n\n")

        # 重要发现
        sorted_contrib = contributions_df.sort_values('Avg Impact (%)', ascending=False)
        report.append("###  关键发现\n\n")

        for idx, (i, row) in enumerate(sorted_contrib.iterrows(), 1):
            impact = row['Avg Impact (%)']
            if impact > 3:
                level = "HIGH 关键模块"
            elif impact > 1:
                level = "MED 重要模块"
            else:
                level = "LOW 辅助模块"

            report.append(f"{idx}. **{row['Model']}** {level}\n")
            report.append(f"   - 准确度下降: {row['Accuracy Drop (%)']:.3f}%\n")
            report.append(f"   - F1下降: {row['F1 Drop (%)']:.3f}%\n")
            report.append(f"   - 平均影响: {row['Avg Impact (%)']:.3f}%\n")
        report.append("\n")

    # 5. 性能排序
    if not df.empty:
        report.append("## 5. 模型排序\n\n")
        sorted_df = df.sort_values('Accuracy', ascending=False)

        report.append("### 按Accuracy排序\n\n")
        for idx, (i, row) in enumerate(sorted_df.iterrows(), 1):
            report.append(f"{idx}. **{row['Model']}**: Accuracy={row['Accuracy']:.4f}, F1={row['Macro-F1']:.4f}\n")
        report.append("\n")

    # 6. 结论
    if not df.empty:
        best_model = df.loc[df['Accuracy'].idxmax()]
        worst_model = df.loc[df['Accuracy'].idxmin()]

        report.append("## 6. 结论与建议\n\n")
        report.append(f"- **最佳模型**: {best_model['Model']} (Accuracy: {best_model['Accuracy']:.4f})\n")
        report.append(f"- **最简模型**: {worst_model['Model']} (Accuracy: {worst_model['Accuracy']:.4f})\n")
        report.append(f"- **性能差异**: {(best_model['Accuracy'] - worst_model['Accuracy']):.4f} ({(best_model['Accuracy'] - worst_model['Accuracy'])/worst_model['Accuracy']*100:.2f}%)\n\n")

        if not contributions_df.empty:
            most_important = contributions_df.loc[contributions_df['Avg Impact (%)'].idxmax()]
            report.append(f"- **最关键模块**: {most_important['Model']} (平均影响: {most_important['Avg Impact (%)']:.3f}%)\n")

        report.append("\n### 建议\n\n")
        report.append("1. 关键模块应该重点优化和改进\n")
        report.append("2. 辅助模块可考虑简化以减少模型复杂度\n")
        report.append("3. 根据业务需求平衡模型性能和计算成本\n")
        report.append("\n")

    report_text = "".join(report)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"[OK] 已保存: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="C3N 消融实验结果分析")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints",
                       help="checkpoints 目录路径")
    parser.add_argument("--output_dir", type=str, default="results/ablation",
                       help="输出结果目录")
    parser.add_argument("--results_json", type=str, default="results/ablation/results.json",
                       help="结果JSON文件路径")

    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("C3N 消融实验结果分析")
    print("="*70)
    print()

    # 1. 收集结果
    print("[INFO] 收集消融实验结果...")
    print("-" * 70)
    results = collect_ablation_results(checkpoints_dir)
    print()

    # 2. 创建对比表
    print(" 生成对比表...")
    df = create_comparison_table(results)

    if df.empty:
        print("[ERROR] 错误：未找到任何实验结果！")
        print(f"请检查 {checkpoints_dir} 目录中是否存在实验结果")
        return

    # 保存为CSV
    csv_path = output_dir / 'ablation_results.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"[OK] 已保存 CSV: {csv_path}")
    print()

    # 3. 计算贡献度
    print("[INFO] 计算模块贡献度...")
    contributions_df = calculate_contributions(df)

    if not contributions_df.empty:
        contrib_csv = output_dir / 'contribution_analysis.csv'
        contributions_df.to_csv(contrib_csv, index=False, encoding='utf-8')
        print(f"[OK] 已保存 CSV: {contrib_csv}")
    print()

    # 4. 绘制图表
    print("[ART] 生成对比图表...")
    plot_comparison_metrics(df, output_dir)
    if not contributions_df.empty:
        plot_contribution_analysis(contributions_df, output_dir)
    print()

    # 5. 生成报告
    print("[DOC] 生成Markdown报告...")
    report_path = output_dir / 'ablation_report.md'
    generate_markdown_report(df, contributions_df, report_path)
    print()

    # 6. 打印总结
    print("="*70)
    print("[DONE] 分析完成！")
    print("="*70)
    print()
    print(df[['Model', 'Accuracy', 'Macro-F1', 'Loss']].to_string(index=False))
    print()
    print(f" 结果已保存到: {output_dir}")
    print()
    print("生成的文件:")
    print(f"  - {csv_path.name}: 完整性能指标")
    print(f"  - {report_path.name}: Markdown分析报告")
    print(f"  - metrics_comparison.png: 性能对比图")
    if not contributions_df.empty:
        print(f"  - contribution_analysis.png: 贡献度分析图")
        print(f"  - contribution_analysis.csv: 贡献度详细数据")


if __name__ == "__main__":
    main()
