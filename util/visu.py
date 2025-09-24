from matplotlib.colors import Normalize, ListedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import matplotlib
from typing import List
cmaps = ['winter', 'hsv', 'Wistia', 'BuGn']
import os

def make_episode_visualization(img_s: np.ndarray,
                               img_q: np.ndarray,
                               gt_s: np.ndarray,
                               gt_q: np.ndarray,
                               preds: np.ndarray,
                               save_path: str,
                               mean: List[float] = [0.485, 0.456, 0.406],
                               std: List[float] = [0.229, 0.224, 0.225]):

    # 0) Preliminary checks
    assert len(img_s.shape) == 4, f"Support shape expected : K x 3 x H x W or K x H x W x 3. Currently: {img_s.shape}"
    assert len(img_q.shape) == 3, f"Query shape expected : 3 x H x W or H x W x 3. Currently: {img_q.shape}"
    assert len(preds.shape) == 4, f"Predictions shape expected : T x num_classes x H x W. Currently: {preds.shape}"
    assert len(gt_s.shape) == 3, f"Support GT shape expected : K x H x W. Currently: {gt_s.shape}"
    assert len(gt_q.shape) == 2, f"Query GT shape expected : H x W. Currently: {gt_q.shape}"
    # assert img_s.shape[-1] == img_q.shape[-1] == 3, "Images need to be in the format H x W x 3"
    if img_s.shape[1] == 3:
        img_s = np.transpose(img_s, (0, 2, 3, 1))
    if img_q.shape[0] == 3:
        img_q = np.transpose(img_q, (1, 2, 0))

    assert img_s.shape[-3:-1] == img_q.shape[-3:-1] == gt_s.shape[-2:] == gt_q.shape

    if not os.path.exists("qualitative_results"):
        os.makedirs("qualitative_results")
    """
    if not os.path.exists(os.path.join("qualitative_results", save_path)):
        os.makedirs(os.path.join("qualitative_results", save_path))
    """
    if img_s.min() <= 0:
        img_s *= std
        img_s += mean

    if img_q.min() <= 0:
        img_q *= std
        img_q += mean

    img_s = img_s[0]
    img_q = np.clip(img_q, 0, 1)
    img_s = np.clip(img_s, 0, 1)
    mask_pred = preds.argmax(1)[0]

    gt_s[np.where(gt_s == 255)] = 0
    gt_q[np.where(gt_q == 255)] = 0


    # Create visualizations with different colored transparent overlays
    make_plot(img_q, gt_q, "qualitative_results/" + save_path + "_qry.png", 'blue')    # Blue for query
    make_plot(img_s, gt_s[0], "qualitative_results/" + save_path + "_sup.png", 'red')  # Red for support
    make_plot(img_q, mask_pred, "qualitative_results/" + save_path + "_pred.png", 'hsv') # Green for prediction
    """
    make_plot(img_q, gt_q, "qualitative_results/" + save_path + "/" + save_path + "_qry.png", ['hsv'])
    make_plot(img_s, gt_s[0], "qualitative_results/" + save_path + "/" + save_path + "_sup.png", ['hsv'])
    make_plot(img_q, mask_pred, "qualitative_results/" + save_path + "/" + save_path + "_pred.png", ['hsv'])
    """

def make_plot(img: np.ndarray,
             mask: np.ndarray,
             save_path: str,
             color: str = 'yellow'):  # Now accepts a color name
    sizes = np.shape(img)
    fig = plt.figure()
    fig.set_size_inches(4. * sizes[0] / sizes[1], 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, interpolation='none')

    # Create colormap based on input color
    if color == 'yellow':
        overlay_cmap = ListedColormap([[1.0, 1.0, 0.0, alpha] for alpha in np.linspace(0, 0.5, 256)])
    elif color == 'red':
        overlay_cmap = ListedColormap([[1.0, 0.0, 0.0, alpha] for alpha in np.linspace(0, 0.5, 256)])
    elif color == 'hsv':
        overlay_cmap = ListedColormap([[0.0, 1.0, 0.0, alpha] for alpha in np.linspace(0, 0.5, 256)])
    elif color == 'blue':
        overlay_cmap = ListedColormap([[0.0, 0.0, 1.0, alpha] for alpha in np.linspace(0, 0.5, 256)])
    else:
        overlay_cmap = ListedColormap([[1.0, 1.0, 0.0, alpha] for alpha in np.linspace(0, 0.5, 256)])

    # Apply the same transparent overlay effect as before
    alphas = Normalize(0, .3, clip=True)(mask)
    alphas = np.clip(alphas, 0., 0.5)
    colors = Normalize()(mask)
    colors = overlay_cmap(colors)
    colors[..., -1] = alphas
    ax.imshow(colors)

    plt.savefig(save_path, dpi=300)
    plt.close()

def save_clip_similarity_heatmap(clip_similarity, save_path, que_name, normalize=True):
    """
    Save heatmap images of clip_similarity tensor using que_name only.

    Args:
        clip_similarity (torch.Tensor): shape [B, C, H, W]
        save_path (str): directory path to save
        que_name (list of str): list of query image names
        normalize (bool): normalize to [0, 1] for visualization
    """
    os.makedirs(save_path, exist_ok=True)
    B, C, H, W = clip_similarity.shape

    for b in range(B):
        for c in range(C):
            heatmap = clip_similarity[b, c].detach().cpu()
            if normalize:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            plt.figure(figsize=(4, 4))
            plt.axis('off')
            plt.imshow(heatmap, cmap='jet')

            # Format filename
            name = que_name[b] if isinstance(que_name[b], str) else str(que_name[b])
            name = name.replace('/', '_').replace(' ', '_')
            fname = os.path.join(save_path, f"{name}_sim{c}.png")

            plt.savefig(fname, bbox_inches='tight', pad_inches=0)
            plt.close()
