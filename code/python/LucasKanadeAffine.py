import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform


def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """
    # Coordinate:
    # "x" - columns
    # "y" - rows
    # Origin - top left
    # Image Coordinates = (x, y)

    # Shape check
    assert It1.shape == It.shape

    # Get Dimensions
    img_rows, img_cols = It.shape
    N = img_rows * img_cols

    # Interpolate Source image in this coordinate system
    y = np.arange(0, img_rows)
    x = np.arange(0, img_cols)
    spline_It1 = RectBivariateSpline(x, y, It1.T)

    # get gradients of It1 using its spline
    cols_grid, rows_grid = np.meshgrid(x, y)
    It1_gx = spline_It1.ev(cols_grid, rows_grid, dx=1, dy=0)
    It1_gy = spline_It1.ev(cols_grid, rows_grid, dx=0, dy=1)

    # Affine warp Jacobian for (x,y) img coordinate system
    J = np.array(
        [
            [
                cols_grid.reshape(N),
                rows_grid.reshape(N),
                np.ones(N),
                np.zeros(N),
                np.zeros(N),
                np.zeros(N),
            ],
            [
                np.zeros(N),
                np.zeros(N),
                np.zeros(N),
                cols_grid.reshape(N),
                rows_grid.reshape(N),
                np.ones(N),
            ],
        ]
    )

    J = J.transpose(2, 0, 1)

    # Gradients matrix
    It1_g = np.vstack((It1_gx.ravel(order="C"), It1_gy.ravel(order="C"))).T
    It1_g = It1_g.reshape(N, 1, 2)

    # Loop variable
    iters = 0

    # Initial Guess
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    while iters < num_iters:
        # 1. Warp Image
        It1_warped = affine_transform(It1.T, M).T
        # Get valid points
        mask = np.where(It1_warped > 0, 1, 0)

        # 2. Compute Error Image
        error_image = It - It1_warped
        # Get valid points
        error_image = np.where(mask, error_image, 0)
        # Flatten
        error_image = error_image.reshape(N, 1)

        # 3. Compute Gradient Matrix
        # Get valid gradients
        It1_gx = np.where(mask, It1_gx, 0)
        It1_gy = np.where(mask, It1_gy, 0)
        # Assemble matrix
        It1_g = np.vstack((It1_gx.ravel(order="C"), It1_gy.ravel(order="C"))).T
        It1_g = It1_g.reshape(N, 1, 2)

        # 4. Compute Hessian
        A = (It1_g @ J).squeeze(axis=1)
        H = A.T @ A

        # 5. Compute del_p
        del_p = np.linalg.inv(H) @ (A.T) @ error_image

        # Update Parameters
        p[0] += del_p[0, 0]
        p[1] += del_p[1, 0]
        p[2] += del_p[2, 0]
        p[3] += del_p[3, 0]
        p[4] += del_p[4, 0]
        p[5] += del_p[5, 0]

        # update M with new p
        M = np.array(
            [[1.0 + p[0], p[1], p[2]], [p[3], 1.0 + p[4], p[5]], [0.0, 0.0, 1.0]]
        )
        iters += 1

        # Threshold
        # if np.square(del_p).sum() < threshold: # Sqaure of norm
        if np.linalg.norm(del_p) < threshold:  # Norm
            # print("Total Iters:", iters)
            break
    return M
