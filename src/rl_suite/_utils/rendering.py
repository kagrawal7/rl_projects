import matplotlib.pyplot as plt
from math import ceil


def render_env_in_notebook(env, figsize=(3, 2)):
    """Render a Gymnasium rgb_array environment inside a notebook."""
    plt.figure(figsize=figsize)
    plt.imshow(env.render())
    plt.margins(x=0, y=0)
    plt.axis("off")
    plt.show()


def display_renders(images, num_cols=4):
    """Display a list of rendered environment frames."""
    n = len(images)
    if not n:
        return

    num_rows = ceil(n / num_cols)
    fig = plt.figure(figsize=(3 * num_cols, 2 * num_rows))

    for i, image in enumerate(images):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.imshow(image)
        ax.margins(x=0, y=0)
        ax.axis("off")

    fig.subplots_adjust(wspace=0.025, hspace=0.2)
    plt.show()


__all__ = ["display_renders", "render_env_in_notebook"]
