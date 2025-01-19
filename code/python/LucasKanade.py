import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    """
    # Coordinate:
    # "x" - columns
    # "y" - rows
    # Origin - top left
    # Image = (y, x)

    # Get dimensions
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    img_rows, img_cols = It.shape
    window_rows, window_cols = y2 - y1, x2 - x1

    # Interpolate Source Image
    y = np.arange(0, img_rows, 1)
    x = np.arange(0, img_cols, 1)
    columns_grid, rows_grid = np.meshgrid(
        np.linspace(x1, x2, int(window_cols)), np.linspace(y1, y2, int(window_rows))
    )
    spline_It = RectBivariateSpline(y, x, It)

    # Get Template
    T = spline_It.ev(rows_grid, columns_grid)

    # Interpolate Destination image
    spline_It1 = RectBivariateSpline(y, x, It1)

    # Compute and Interpolate Destination Image Gradients
    It1_gy, It1_gx = np.gradient(It1)
    spline_It1_gx = RectBivariateSpline(y, x, It1_gx)
    spline_It1_gy = RectBivariateSpline(y, x, It1_gy)

    # Translation Warp Jacobian
    d_W_dp = np.array([[1, 0], [0, 1]])

    # Loop variable
    iters = 0

    while iters < num_iters:
        # 0. Update Warp Parameters
        x1_w, y1_w, x2_w, y2_w = x1 + p0[0], y1 + p0[1], x2 + p0[0], y2 + p0[1]

        # Get new window coordinates
        warp_cols_grid, warp_rows_grid = np.meshgrid(
            np.linspace(x1_w, x2_w, int(window_cols)),
            np.linspace(y1_w, y2_w, int(window_rows)),
        )

        # 1. Warp Image
        warped_image = spline_It1.ev(warp_rows_grid, warp_cols_grid)

        # 2. Compute Error Image
        error_image = T - warped_image
        # Flatten
        error_image = error_image.reshape(-1, 1)

        # 3. Compute Gradient Matrix
        # Get gradients at warp position
        It1_gx_w = spline_It1_gx.ev(warp_rows_grid, warp_cols_grid).ravel(order="C")
        It1_gy_w = spline_It1_gy.ev(warp_rows_grid, warp_cols_grid).ravel(order="C")
        # Assemble matrix
        It1_g = np.vstack(
            (It1_gx_w, It1_gy_w)
        ).T  # Left column x derivative, Right Column y derivative
        # print("Warped Gradient Shape:", It1_g.shape)

        # 4. Compute Hessian
        A = It1_g @ d_W_dp
        H = A.T @ A

        # 5. Compute del_p
        del_p = np.linalg.inv(H) @ (A.T) @ error_image

        # 6. Update Parameters
        p0[0] += del_p[0, 0]
        p0[1] += del_p[1, 0]

        iters += 1

        # Threshold
        # if np.square(del_p).sum() < threshold: # Sqaure of norm
        if np.linalg.norm(del_p) < threshold:  # Norm
            # print("Total Iters:", iters)
            break

    return p0
