import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from SubtractDominantMotion import SubtractDominantMotion

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_iters", type=int, default=1e3, help="number of iterations of Lucas-Kanade"
)
parser.add_argument(
    "--threshold",
    type=float,
    default=1e-2,
    help="dp threshold of Lucas-Kanade for terminating optimization",
)
parser.add_argument(
    "--tolerance",
    type=float,
    default=0.02,
    help="binary threshold of intensity difference when computing the mask",
)
parser.add_argument(
    "--seq_file",
    default="../data/antseq.npy",
)

args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance
seq_file = args.seq_file

seq = np.load(seq_file)

# Starting from second frame
for i in range(1, seq.shape[2]):
    It1 = seq[:, :, i]  # Current frame
    It = seq[:, :, i - 1]  # Previous frame

    # Subtract dominant motion from previous frame and current frame and get motion mask for current image
    # motion_mask = SubtractDominantMotion(
    #     image1=It,
    #     image2=It1,
    #     threshold=threshold,
    #     num_iters=num_iters,
    #     tolerance=tolerance,
    #     inv_comp=False,
    # )
    motion_mask_inv = SubtractDominantMotion(
        image1=It,
        image2=It1,
        threshold=threshold,
        num_iters=num_iters,
        tolerance=tolerance,
        inv_comp=True,
    )

    if (i + 1) % 30 == 0:
        # save plot
        fig, ax = plt.subplots(1)
        ax.imshow(It1, cmap="gray")
        # ax.imshow(motion_mask, alpha=0.5)
        ax.imshow(motion_mask_inv, alpha=0.5)
        plt.title("Frame " + str(i + 1))
        plt.axis("off")
        plt.show()
