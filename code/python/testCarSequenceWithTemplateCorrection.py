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
    default=1e-2,
    help="dp threshold of Lucas-Kanade for terminating optimization",
)
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

# Template drift threshold
epsilon = 0.1

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]  # Template Patch coordinates for first frame
rect_0 = rect.copy()

car_seq_rects_wcrt = []  # To store the template patch coordinates
car_seq_rects_wcrt.append(rect)

fig, ax = plt.subplots(nrows=1, ncols=5)

It0 = seq[:, :, 0]  # First Frame
T = It0  # Initialise template to first frame

for i in range(1, seq.shape[2]):
    It1 = seq[:, :, i]  # Current frame
    It = seq[:, :, i - 1]  # Previous frame

    # Get patch shift for current frame
    p = LucasKanade(
        It=It, It1=It1, rect=rect, num_iters=num_iters, threshold=threshold
    )

    # Get the accumulated patch shift
    p_n = [rect[0] - rect_0[0], rect[1] - rect_0[1]]
    p_n = p_n + p

    # Get patch shift from first frame, beginning at the accumulated patch shift
    p_star = LucasKanade(
        It=It0, It1=It1, rect=rect_0, num_iters=num_iters, threshold=threshold, p0=p_n
    )

    # Perform template update
    if np.linalg.norm(p_star - p_n) < epsilon:
        # No big change in template, use the first frame
        # Update patch coordinates with the patch shift from first frame
        rect = [
            rect_0[0] + p_star[0],  # x1
            rect_0[1] + p_star[1],  # y1
            rect_0[2] + p_star[0],  # x2
            rect_0[3] + p_star[1],  # y2
        ]
    else:
        # There has been a big change in the template and we cannot use the first frame anymore
        # Update patch coordinates with the patch shift from current frame
        rect = [
            rect[0] + p_n[0],  # x1
            rect[1] + p_n[1],  # y1
            rect[2] + p_n[0],  # x2
            rect[3] + p_n[1],  # y2
        ]

    # Store the patch coordinates
    car_seq_rects_wcrt.append(rect)

# Save seq rects to NUMPY ARRAY!
car_seq_rects_wcrt = np.array(car_seq_rects_wcrt)
np.save("carseqrects-wcrt.npy", car_seq_rects_wcrt)
