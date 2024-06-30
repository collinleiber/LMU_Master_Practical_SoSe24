import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'footprintmatrix_plots')


def visualize_sorted_dict(sorted_dict, name):
    keys = list(sorted_dict.keys())
    size = len(keys) if len(keys) > 0 else 1

    fig, ax = plt.subplots(figsize=(size * 2, size * 2))

    ax.matshow([[0] * size] * size, cmap='Greys')  # Create an empty heatmap

    # Add gridlines
    ax.set_xticks([x - 0.5 for x in range(1, size)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, size)], minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=5)
    for i, row_key in enumerate(keys):
        for j, col_key in enumerate(keys):
            ax.text(
                j,
                i,
                sorted_dict[row_key][col_key],
                ha='center',
                va='center',
                color='black',
                fontsize=38,
            )

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(keys, fontsize=28)
    ax.set_yticklabels(keys, fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(5)  # Adjust the linewidth for the outer border
    plt.title('Footprint Matrix {}'.format(name), fontsize=22)
    # plt.show()
    plt.savefig(OUTPUT_DIR + '/fpm_{}.png'.format(name))
