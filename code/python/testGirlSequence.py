import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from LucasKanade import LucasKanade

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_iters", type=int, default=1e4, help="number of iterations of Lucas-Kanade"
)
parser.add_argument(
    "--threshold",
    type=float,
    default=1e-1,
    help="dp threshold of Lucas-Kanade for terminating optimization",
)
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]  # Template Patch coordinates for first frame

girl_seq_rects = []  # To store the template patch coordinates
girl_seq_rects.append(rect)

fig, ax = plt.subplots(nrows=1, ncols=5)

# Show first frame with patch
# print("Frame 1 Patch Rect:", rect)
# im_1 = seq[:, :, 0]
# ax[0].imshow(im_1)
# ax[0].set_title("Frame 1")
# patch = patches.Rectangle(
#     (rect[0], rect[1]),
#     rect[2] - rect[0],
#     rect[3] - rect[1],
#     linewidth=1,
#     edgecolor="r",
#     facecolor="none",
# )
# ax[0].add_patch(patch)

for i in range(1, seq.shape[2]):
    It1 = seq[:, :, i]  # Current frame
    It = seq[:, :, i - 1]  # Previous frame

    # Get patch shift for current frame
    p = LucasKanade(It=It, It1=It1, rect=rect, num_iters=num_iters, threshold=threshold)

    # Update patch coordinates for current frame
    rect = [
        rect[0] + p[0],  # x1
        rect[1] + p[1],  # y1
        rect[2] + p[0],  # x2
        rect[3] + p[1],  # y2
    ]

    # Store the patch coordinates
    girl_seq_rects.append(rect)

# Save seq rects to NUMPY ARRAY!
girl_seq_rects = np.array(girl_seq_rects)
np.save("girlseqrects.npy", girl_seq_rects)
