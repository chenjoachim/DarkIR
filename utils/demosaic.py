import numpy as np
from scipy import signal
from copy import deepcopy

def demosaic(noisy: np.ndarray) -> np.ndarray:
    R = np.zeros_like(noisy)
    G = deepcopy(noisy)
    B = np.zeros_like(noisy)

    x, y = np.meshgrid(
        np.arange(0, noisy.shape[0]),
        np.arange(0, noisy.shape[1]),
        indexing="ij",
    )

    R_xrange = np.arange(0, noisy.shape[0], 2)
    R_yrange = np.arange(0, noisy.shape[1], 2)
    R_x, R_y = np.meshgrid(R_xrange, R_yrange, indexing="ij")

    B_xrange = np.arange(1, noisy.shape[0], 2)
    B_yrange = np.arange(1, noisy.shape[1], 2)
    B_x, B_y = np.meshgrid(B_xrange, B_yrange, indexing="ij")

    G_kernel = (
        np.array(
            [
                [0, 0, -1, 0, 0],
                [0, 0, 2, 0, 0],
                [-1, 2, 4, 2, -1],
                [0, 0, 2, 0, 0],
                [0, 0, -1, 0, 0],
            ]
        )
        / 8
    )

    RB_kernel_row = (
        np.array(
            [
                [0, 0, 0.5, 0, 0],
                [0, -1, 0, -1, 0],
                [-1, 4, 5, 4, -1],
                [0, -1, 0, -1, 0],
                [0, 0, 0.5, 0, 0],
            ]
        )
        / 8
    )

    RB_kernel_col = (
        np.array(
            [
                [0, 0, -1, 0, 0],
                [0, -1, 4, -1, 0],
                [0.5, 0, 5, 0, 0.5],
                [0, -1, 4, -1, 0],
                [0, 0, -1, 0, 0],
            ]
        )
        / 8
    )

    RB_kernel_other = (
        np.array(
            [
                [0, 0, -1.5, 0, 0],
                [0, 2, 0, 2, 0],
                [-1.5, 0, 6, 0, -1.5],
                [0, 2, 0, 2, 0],
                [0, 0, -1.5, 0, 0],
            ]
        )
        / 8
    )

    G_updated = signal.convolve2d(noisy, G_kernel, boundary="symm", mode="same")
    RB_updated_row = signal.convolve2d(noisy, RB_kernel_row, boundary="symm", mode="same")
    RB_updated_col = signal.convolve2d(noisy, RB_kernel_col, boundary="symm", mode="same")
    RB_updated_other = signal.convolve2d(
        noisy, RB_kernel_other, boundary="symm", mode="same"
    )

    R_demosaiced = deepcopy(noisy)
    B_demosaiced = deepcopy(noisy)
    G_demosaiced = deepcopy(noisy)

    G_demosaiced[R_x, R_y] = G_updated[R_x, R_y]
    G_demosaiced[B_x, B_y] = G_updated[B_x, B_y]

    R_demosaiced[R_x, B_y] = RB_updated_row[R_x, B_y]
    R_demosaiced[B_x, R_y] = RB_updated_col[B_x, R_y]
    R_demosaiced[B_x, B_y] = RB_updated_other[B_x, B_y]

    B_demosaiced[R_x, B_y] = RB_updated_col[R_x, B_y]
    B_demosaiced[B_x, R_y] = RB_updated_row[B_x, R_y]
    B_demosaiced[R_x, R_y] = RB_updated_other[R_x, R_y]

    malvar_demosaiced = np.stack((R_demosaiced, G_demosaiced, B_demosaiced), axis=2)
    
    return malvar_demosaiced