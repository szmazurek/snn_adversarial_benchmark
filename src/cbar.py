import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def create_svg_colorbar():
    # Create figure with specific size for just the colorbar
    # Increased height to 8 inches for a longer colorbar
    fig, ax = plt.subplots(figsize=(3, 8))

    # Create the colormap - viridis so lightest is at top (value 1)
    cmap = plt.cm.viridis

    # Create normalization from -1 to 1
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    # Create colorbar
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        orientation="vertical",
        shrink=1.0,
        aspect=40,
    )

    # Set colorbar ticks
    ticks = np.linspace(
        -1, 1, 9
    )  # -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0
    cb.set_ticks(ticks)

    # Custom formatting for tick labels
    formatted_labels = []
    for tick in ticks:
        if tick == 1.0 or tick == -1.0:
            # Format 1.0 and -1.0 as integers
            formatted_labels.append(f"{int(tick)}")
        elif -1 < tick < 1:
            # Format other decimals, removing leading '0' for non-zero values
            # and handling 0.00 specifically
            if tick == 0.0:
                formatted_labels.append("0")
            else:
                # Use string formatting to get '.25' instead of '0.25'
                label = f"{tick:.2f}"
                if label.startswith("0."):
                    label = label[1:]  # Remove leading '0'
                elif label.startswith("-0."):
                    label = "-" + label[2:]  # Remove leading '0' but keep '-'
                formatted_labels.append(label)
        else:
            formatted_labels.append(
                f"{tick:.2f}"
            )  # Fallback for unexpected values

    cb.set_ticklabels(formatted_labels)

    # Rotate tick labels to be horizontal and increase their font size
    cb.ax.tick_params(labelrotation=270, labelsize=18)  # Increased labelsize

    # Add label and increase its font size
    cb.set_label(
        "Correlation coefficient",
        rotation=270,
        labelpad=20,
        fontsize=18,  # Increased fontsize
    )

    # Remove the main axes (we only want the colorbar)
    ax.remove()

    # Adjust layout to minimize whitespace
    plt.tight_layout()

    # Save as PNG with 600 DPI
    plt.savefig(
        "colorbar.png",
        format="png",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=600,
    )
    plt.close()

    print("Colorbar saved as 'colorbar.png'")


if __name__ == "__main__":
    create_svg_colorbar()
