"""Generate five publication-ready charts from the current aggregate results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = [
    'GCN + SAC',
    'GAT + DQN',
    'GAT + PPO',
    'Attack Graph + Q-Learning',
]

MODEL_COLORS = ['#1769AA', '#D97706', '#15803D', '#A21CAF']

# Values are the supplied aggregate experimental results; no rollout data is made.
DATA = {
    'Accuracy': [77.616, 75.288, 75.125, 67.426],
    'Precision': [77.159, 75.288, 75.125, 73.522],
    'Recall': [100.0, 100.0, 100.0, 87.339],
    'F1-score': [87.107, 85.902, 85.796, 79.837],
    'ROC-AUC': [94.909, 67.415, 42.620, 47.065],
    'Defense Success Rate': [24.852, 24.486, 24.533, 42.404],
    'Attack Containment Rate': [91.25, 89.333, 88.083, 99.834],
    'Attack Reduction Rate': [-23.810, -23.810, -23.810, -17.429],
    'Average Reward': [17.538, -31.142, -48.036, -79.482],
    'Unnecessary Isolation Rate': [0.0, 0.0, 0.0, 46.124],
}

VALUE_LABELS = {
    'Accuracy': ['77.616%', '75.288%', '75.125%', '67.426%'],
    'Precision': ['77.159%', '75.288%', '75.125%', '73.522%'],
    'Recall': ['100%', '100%', '100%', '87.339%'],
    'F1-score': ['87.107%', '85.902%', '85.796%', '79.837%'],
    'ROC-AUC': ['94.909%', '67.415%', '42.620%', '47.065%'],
    'Defense Success Rate': ['24.852%', '24.486%', '24.533%', '42.404%'],
    'Attack Containment Rate': ['91.25%', '89.333%', '88.083%', '99.834%'],
    'Attack Reduction Rate': ['-23.810%', '-23.810%', '-23.810%', '-17.429%'],
    'Average Reward': ['17.538', '-31.142', '-48.036', '-79.482'],
    'Unnecessary Isolation Rate': ['0%', '0%', '0%', '46.124%'],
}

OUTPUT_DIR = Path(__file__).resolve().parent / 'comparison_charts'


def style_axis(axis):
    axis.set_facecolor('white')
    axis.grid(axis='y', color='#D8DEE8', linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_color('#AAB3C0')
    axis.spines['bottom'].set_color('#AAB3C0')
    axis.tick_params(axis='both', labelsize=10, colors='#263445')


def label_bars(axis, bars, labels, offset=4):
    for bar, label in zip(bars, labels):
        value = bar.get_height()
        axis.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, offset if value >= 0 else -offset),
            textcoords='offset points',
            ha='center',
            va='bottom' if value >= 0 else 'top',
            fontsize=9,
            color='#202B38',
            clip_on=False,
        )


def grouped_bar_chart(metrics, title, filename, ylabel='Percentage (%)'):
    positions = np.arange(len(MODELS))
    width = 0.19 if len(metrics) == 4 else 0.31
    figure_width = 14.5 if len(metrics) == 4 else 11.5
    figure, axis = plt.subplots(figsize=(figure_width, 6.4), facecolor='white')
    style_axis(axis)

    for metric_index, metric in enumerate(metrics):
        offsets = (metric_index - (len(metrics) - 1) / 2) * width
        bars = axis.bar(
            positions + offsets,
            DATA[metric],
            width=width * 0.92,
            label=metric,
            color=MODEL_COLORS[metric_index],
            edgecolor='white',
            linewidth=0.7,
            zorder=3,
        )
        label_bars(axis, bars, VALUE_LABELS[metric])

    axis.set_title(title, fontsize=16, fontweight='bold', color='#172536', pad=18)
    axis.set_ylabel(ylabel, fontsize=11, color='#263445')
    axis.set_xlabel('Models', fontsize=11, color='#263445', labelpad=12)
    axis.set_xticks(positions)
    axis.set_xticklabels(MODELS, rotation=12, ha='right')
    axis.set_ylim(0, max(max(DATA[metric]) for metric in metrics) * 1.22)
    axis.legend(frameon=False, ncol=len(metrics), loc='upper center', bbox_to_anchor=(0.5, 1.0))
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight', facecolor='white')
    return figure


def make_detection_trend():
    figure, axis = plt.subplots(figsize=(11.5, 6.4), facecolor='white')
    style_axis(axis)
    positions = np.arange(len(MODELS))
    line_styles = [
        ('Accuracy', '#1769AA', 'o'),
        ('F1-score', '#D97706', 's'),
        ('ROC-AUC', '#15803D', '^'),
    ]
    for metric, color, marker in line_styles:
        axis.plot(
            positions,
            DATA[metric],
            label=metric,
            color=color,
            marker=marker,
            markersize=8,
            linewidth=2.4,
            markeredgecolor='white',
            markeredgewidth=1.0,
            zorder=3,
        )
    axis.set_title('Detection Performance Trend Across Models', fontsize=16, fontweight='bold', color='#172536', pad=18)
    axis.set_ylabel('Percentage (%)', fontsize=11, color='#263445')
    axis.set_xlabel('Models', fontsize=11, color='#263445', labelpad=12)
    axis.set_xticks(positions)
    axis.set_xticklabels(MODELS, rotation=12, ha='right')
    axis.set_ylim(0, 112)
    axis.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / '02_model_performance_line.png', dpi=300, bbox_inches='tight', facecolor='white')
    return figure


def make_attack_reduction():
    figure, axis = plt.subplots(figsize=(10.8, 6.2), facecolor='white')
    style_axis(axis)
    positions = np.arange(len(MODELS))
    bars = axis.bar(
        positions,
        DATA['Attack Reduction Rate'],
        width=0.62,
        color=MODEL_COLORS,
        edgecolor='white',
        linewidth=0.8,
        zorder=3,
    )
    axis.axhline(0, color='#263445', linewidth=1.2, zorder=2)
    label_bars(axis, bars, VALUE_LABELS['Attack Reduction Rate'], offset=5)
    axis.set_title('Attack Reduction Rate Comparison', fontsize=16, fontweight='bold', color='#172536', pad=18)
    axis.set_ylabel('Percentage (%)', fontsize=11, color='#263445')
    axis.set_xlabel('Models', fontsize=11, color='#263445', labelpad=12)
    axis.set_xticks(positions)
    axis.set_xticklabels(MODELS, rotation=12, ha='right')
    axis.set_ylim(-32, 5)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / '04_attack_reduction.png', dpi=300, bbox_inches='tight', facecolor='white')
    return figure


def make_reward_and_isolation():
    figure, (reward_axis, isolation_axis) = plt.subplots(
        1, 2, figsize=(13, 6.4), facecolor='white'
    )
    positions = np.arange(len(MODELS))

    for axis in (reward_axis, isolation_axis):
        style_axis(axis)
        axis.set_xticks(positions)
        axis.set_xticklabels(MODELS, rotation=18, ha='right')
        axis.set_xlabel('Models', fontsize=10, color='#263445', labelpad=10)

    reward_bars = reward_axis.bar(
        positions,
        DATA['Average Reward'],
        width=0.64,
        color=MODEL_COLORS,
        edgecolor='white',
        linewidth=0.8,
        zorder=3,
    )
    reward_axis.axhline(0, color='#263445', linewidth=1.1, zorder=2)
    label_bars(reward_axis, reward_bars, VALUE_LABELS['Average Reward'], offset=5)
    reward_axis.set_ylabel('Average Reward', fontsize=11, color='#263445')
    reward_axis.set_title('Average Reward', fontsize=13, fontweight='bold', color='#172536', pad=12)
    reward_axis.set_ylim(-100, 30)

    isolation_bars = isolation_axis.bar(
        positions,
        DATA['Unnecessary Isolation Rate'],
        width=0.64,
        color=MODEL_COLORS,
        edgecolor='white',
        linewidth=0.8,
        zorder=3,
    )
    label_bars(isolation_axis, isolation_bars, VALUE_LABELS['Unnecessary Isolation Rate'], offset=5)
    isolation_axis.set_ylabel('Unnecessary Isolation Rate (%)', fontsize=11, color='#263445')
    isolation_axis.set_title('Unnecessary Isolation Rate', fontsize=13, fontweight='bold', color='#172536', pad=12)
    isolation_axis.set_ylim(0, 58)

    figure.suptitle('Reward and Operational Cost Comparison', fontsize=16, fontweight='bold', color='#172536', y=1.02)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / '05_reward_and_isolation.png', dpi=300, bbox_inches='tight', facecolor='white')
    return figure


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures = []
    figures.append(grouped_bar_chart(
        ['Accuracy', 'Precision', 'Recall', 'F1-score'],
        'Detection Performance Comparison',
        '01_detection_performance.png',
    ))
    figures.append(make_detection_trend())
    figures.append(grouped_bar_chart(
        ['Defense Success Rate', 'Attack Containment Rate'],
        'Defense Performance Comparison',
        '03_defense_performance.png',
    ))
    figures.append(make_attack_reduction())
    figures.append(make_reward_and_isolation())

    paths = [
        OUTPUT_DIR / '01_detection_performance.png',
        OUTPUT_DIR / '02_model_performance_line.png',
        OUTPUT_DIR / '03_defense_performance.png',
        OUTPUT_DIR / '04_attack_reduction.png',
        OUTPUT_DIR / '05_reward_and_isolation.png',
    ]
    for path in paths:
        print(path.resolve())

    # One show call displays all five generated figures for visual inspection.
    plt.show()
    return figures


if __name__ == '__main__':
    main()
