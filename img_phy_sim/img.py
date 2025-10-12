import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import block_reduce  # pip install scikit-image

# -------------------------
# >>> Basic Image tools <<<
# -------------------------

def get_bit_depth(img):
    dtype_to_bits = {
        np.uint8: 8,
        np.uint16: 16,
        np.int16: 16,
        np.float32: 32,
        np.float64: 64
    }
    return dtype_to_bits.get(img.dtype.type, "unknown")


def get_width_height(img, channels_before=0):
    height, width = img.shape[0+channels_before:2+channels_before]
    return width, height



def open(src, should_scale=True, should_print=True):
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]

    if should_scale:
        img = img / ((2**get_bit_depth(img)) -1)

    if should_print:
        print(f"Loaded Image:\n    - Image size: {width}x{height}\n    - Bit depth: {get_bit_depth(img)}-bit\n    - Dtype: {img.dtype}")

    return img



def imshow(img, size=8, axis_off=True):
    height, width = img.shape[:2]
    ratio = height / width
    plt.figure(figsize=(size, round(size * ratio)))
    if img.ndim > 2:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap="gray")
    plt.axis('off' if axis_off else 'on')
    plt.show()



# ----------------------------
# >>> Advanced Image Tools <<<
# ----------------------------
#  from https://github.com/xXAI-botXx/AI/blob/main/src/helper/imshow.py

def show_samples(input_samples, pred_samples, real_samples, model_name="Model",
                 n_samples=3, n_cols=4, image_width=4, cmap="gray",
                 normalize=True, invert=False, axis=False,
                 save_to=None, hspace=0.3, wspace=0.2, use_original_style=False):
    """
    General function to plot multiple sets of images: input, prediction, ground truth, difference.

    ---
    Parameters:
    - input_samples, pred_samples, real_samples : list[str] or list[np.ndarray]
        Lists of file paths or arrays for input images, model predictions, and ground truth.
    - model_name : str, optional
        Name of the model to display for prediction column.
    - n_samples : int, optional
        Number of samples to plot.
    - n_cols : int, optional
        Number of columns in the plot (default 4: input, pred, real, difference).
    - image_width : int, optional
        Width of one image in inches.
    - cmap : str, optional
        Colormap to use for displaying images.
    - normalize : bool, optional
        Whether to normalize images to [0,1].
    - invert : bool, optional
        Whether to invert images (255 - image).
    - axis : bool, optional
        Whether to show axes.
    - save_to : str, optional
        Path to save the figure.
    - hspace, wspace : float, optional
        Spacing between images.
    - use_original_style : bool, optional
        Whether to use the current matplotlib style.
    """
    
    def load_image(img):
        if isinstance(img, str):
            arr = cv2.imread(img, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        else:
            arr = img.astype(np.float32)
        if invert:
            arr = 255 - arr
        if normalize:
            arr /= 255.0
        return arr

    # Prepare images and titles
    all_images = []
    titles = []
    sub_images = []

    for idx in range(n_samples):
        all_images.extend([input_samples[idx], pred_samples[idx], real_samples[idx], pred_samples[idx]])
        titles.extend(["Input", model_name, "Ground Truth", "Difference"])
        sub_images.extend([None, None, None, real_samples[idx]])

    # Load all images
    img_arrays = [load_image(im) for im in all_images]
    sub_arrays = [load_image(im) if im is not None else None for im in sub_images]

    # Compute differences where applicable
    for i, sub in enumerate(sub_arrays):
        if sub is not None:
            img_arrays[i] = np.abs(img_arrays[i] - sub)

    n_images = len(img_arrays)
    n_rows = n_images // n_cols + int(n_images % n_cols > 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*image_width, n_rows*image_width))

    if not use_original_style:
        plt_style = 'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'classic'
        plt.style.use(plt_style)

    axes = axes.ravel() if n_images > 1 else [axes]

    for idx, ax in enumerate(axes):
        if idx >= n_images:
            ax.axis('off')
            continue
        ax.imshow(img_arrays[idx], cmap=cmap)
        ax.set_title(titles[idx], fontsize=10)
        if not axis:
            ax.axis('off')

    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, dpi=300)

    plt.show()
    if not use_original_style:
        plt.style.use('default')



def advanced_imshow(img, title=None, image_width=10, axis=False,
           color_space="RGB", cmap=None, cols=1, save_to=None,
           hspace=0.2, wspace=0.2,
           use_original_style=False, invert=False):
    """
    Visualizes one or multiple images.

    Image will be reshaped to: [batch_size/images, width, height, channels]

    ---
    Parameters:
    - img : np.ndarray
        Images with shape [width, height, channels] or [batch_size, width, height, channels].
    - title : str or list, optional
        Title of the whole plot or per-image titles.
    - image_width : int, optional
        Width of one image in the plot.
    - axis : bool, optional
        Whether to show the axis.
    - color_space : str, optional
        The colorspace of the image: RGB, BGR, gray, HSV.
    - cmap : str, optional
        Colormap to use.
    - cols : int, optional
        Number of columns in the plot.
    - save_to : str, optional
        Path to save the figure.
    - hspace, wspace : float, optional
        Spacing between images.
    - use_original_style : bool, optional
        Whether to keep the current matplotlib style.
    - invert : bool, optional
        Whether to invert image colors.
    """
    original_style = plt.rcParams.copy()
    try:
        img_shape = img.shape
    except Exception:
        img = np.array(img)
        img_shape = img.shape
    # Transform to 4D array [N, H, W, C]
    if len(img_shape) == 2:
        img = img.reshape(1, img_shape[0], img_shape[1], 1)
    elif len(img_shape) == 3:
        if img_shape[2] in [1, 3]:  # single image with channels
            img = img.reshape(1, img_shape[0], img_shape[1], img_shape[2])
        else:  # multiple gray images [N,H,W]
            img = img[..., np.newaxis]
    elif len(img_shape) != 4:
        raise ValueError(f"Image(s) have wrong shape: {img_shape}")

    n_images = img.shape[0]
    aspect_ratio = img.shape[2] / img.shape[1]
    rows = n_images // cols + int(n_images % cols > 0)
    width = int(image_width * cols)
    height = int(image_width * rows * aspect_ratio)

    if not use_original_style:
        plt_style = 'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'classic'
        plt.style.use(plt_style)

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(width, height))
    axes = np.array(axes).ravel()
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    if isinstance(title, str):
        fig.suptitle(title, fontsize=20, y=0.95)

    # Invert images if needed
    if invert:
        max_val = 2**(img.dtype.itemsize*8) - 1
        img = max_val - img

    for idx, ax in enumerate(axes):
        if idx >= n_images:
            ax.axis("off")
            continue
        cur_img = img[idx]

        # Handle color spaces
        used_cmap = cmap
        if color_space.lower() == "bgr" and cur_img.shape[2] == 3:
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
            used_cmap = None
        elif color_space.lower() == "hsv" and cur_img.shape[2] == 3:
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_HSV2RGB)
            used_cmap = None
        elif color_space.lower() in ["gray", "grey", "g"]:
            if cur_img.shape[2] == 3:
                cur_img = cv2.cvtColor(cur_img, cv2.COLOR_RGB2GRAY)
            used_cmap = "gray"

        if isinstance(title, (list, tuple)):
            ax.set_title(title[idx], fontsize=12)
        if not axis:
            ax.axis("off")

        ax.imshow(cur_img.squeeze(), cmap=used_cmap)

    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        fig.savefig(save_to, dpi=300)

    plt.show()
    if not use_original_style:
        plt.rcParams.update(original_style)



def show_images(image_paths: list, title=None, image_width=5, axis=False,
                color_space="gray", cmap=None, 
                cols=2, save_to=None, hspace=0.01, wspace=0.01,
                use_original_style=False, invert=False):
    """
    Loads images from paths and visualizes them using `advanced_imshow`.
    """
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if color_space.lower() == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_space.lower() == "hsv":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space.lower() in ["gray", "grey", "g"]:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)

    images = np.array(images)
    advanced_imshow(images, title=title, image_width=image_width, axis=axis,
           color_space=color_space, cmap=cmap, cols=cols, save_to=save_to,
           hspace=hspace, wspace=wspace,
           use_original_style=use_original_style, invert=invert)
    return images



def plot_image_with_values(img, block_size=8, cmap='gray', title=None, 
                           font_size=6, save_to=None):
    """
    Plot an image with mean values computed over non-overlapping blocks, annotated on each block.

    Parameters
    ----------
    img : np.ndarray
        2D grayscale image or 3D single-channel image.
    block_size : int, tuple, optional
        Size of the blocks. If int, square blocks of shape (block_size, block_size) are used.
        Default is 8.
    cmap : str, optional
        Colormap to use for plotting. Default is 'gray'.
    title : str, optional
        Title for the plot.
    font_size : int, optional
        Font size for the annotations. Default is 6.
    save_to : str, optional
        Path to save the figure. If None, figure is not saved.

    Example Usage:
    ```python
    from imshow import plot_image_with_values
    import cv2

    img = cv2.imread('example.png', cv2.IMREAD_GRAYSCALE)
    plot_image_with_values(img, block_size=16, cmap='gray', title='Mean Block Values', font_size=8)
    ```

    Or:
    ```python
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(4*4, 6))
    idx = 0

    img_1 = plot(ax[0][0], path=input_samples[idx], title=f"Input", cmap="gray")
    plot_image_with_values(img_1, block_size=16, ax=ax[1][0])

    img_2 = plot(ax[0][1], path=pred_model[idx], title=f"{model_name}")
    plot_image_with_values(img_2, block_size=16, ax=ax[1][1])

    img_3 = plot(ax[0][2], path=real[idx], title=f"ground truth")
    plot_image_with_values(img_3, block_size=16, ax=ax[1][2])

    img_4 = plot(ax[0][3], path=pred_model[idx], title=f"Difference", sub_image=real[idx])
    plot_image_with_values(img_4, block_size=16, ax=ax[1][3])

    plt.show()
    ```
    """
    # Ensure 2D image
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    elif img.ndim == 3 and img.shape[2] > 1:
        raise ValueError("Only grayscale images are supported for block annotation.")

    # Compute mean over blocks
    mean_img = block_reduce(img, block_size=(block_size, block_size), func=np.mean)
    max_value = mean_img.max()

    # Plot
    plt.figure(figsize=(mean_img.shape[1]/2, mean_img.shape[0]/2))
    plt.imshow(mean_img, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Mean Value')

    # Annotate each block
    for i in range(mean_img.shape[0]):
        for j in range(mean_img.shape[1]):
            val = mean_img[i, j]
            color = 'white' if val < max_value/1.5 else 'black'
            plt.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=font_size)

    plt.title(title or f'Mean Values over {block_size}x{block_size} Blocks')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, dpi=300)
    
    plt.show()

    
def show_image_with_line_and_profile(imgs, axis='row', index=None, titles=None, figsize=(10, 4)):
    """
    Show grayscale images with a selected row or column highlighted,
    and plot its pixel values in a second subplot for each image.

    Parameters
    ----------
    imgs : list of np.ndarray
        List of 2D grayscale images.
    axis : str, optional
        'row' or 'column' to extract the line. Default 'row'.
    index : int, optional
        Index of the row or column to highlight. Defaults to the center.
    titles : list of str, optional
        Titles for each image. Default ["Image 1", "Image 2", ...].
    figsize : tuple, optional
        Figure size per image pair (original + profile).
    Returns
    -------
    line_values_list : list of np.ndarray
        List of pixel values along the selected row/column for each image.

    Example Usage:
    ```python
    line_values = show_image_with_line_and_profile(
        imgs=[img_1, img_2, img_3, img_4], 
        axis='row', 
        index=None, 
        titles=['Input', 'Prediction', 'Ground Truth', 'Difference']
    )
    ```
    """
    assert axis in ['row', 'column'], "axis must be 'row' or 'column'"

    n_images = len(imgs)
    if titles is None:
        titles = [f"Image {i+1}" for i in range(n_images)]
    line_values_list = []

    fig, axes = plt.subplots(n_images, 2, figsize=(figsize[0]*2, figsize[1]*n_images))

    # Ensure axes is 2D
    if n_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (img, ax_pair, title) in enumerate(zip(imgs, axes, titles)):
        h, w = img.shape
        if index is None:
            idx = h // 2 if axis == 'row' else w // 2
        else:
            idx = index

        ax_img, ax_profile = ax_pair

        # Show image with line
        ax_img.imshow(img, cmap='gray', interpolation='nearest')
        if axis == 'row':
            ax_img.axhline(idx, color='red', linewidth=1)
            line_values = img[idx, :]
        else:
            ax_img.axvline(idx, color='red', linewidth=1)
            line_values = img[:, idx]
        line_values_list.append(line_values)

        ax_img.set_title(title)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # Plot pixel values along the line
        x = np.arange(len(line_values))
        ax_profile.plot(x, line_values, color='black', linewidth=1)
        ax_profile.set_title(f'Pixel values along {axis} {idx}')
        ax_profile.set_xlabel('Pixel index')
        ax_profile.set_ylabel('Intensity')
        ax_profile.grid(True)

    plt.tight_layout()
    plt.show()

    return line_values_list

