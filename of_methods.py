## Author: Ziga Trojer, zt0006@student.uni-lj.si

import numpy as np
import cv2
from ex1_utils import gaussderiv, gausssmooth

def uniform_kernel(size):
    """
    :param size: size of a kernel
    :return: uniform kernel
    """
    return np.array([[1] * size] * size)

def lucaskanade(im1, im2, N):
    """
    :param im1: first image matrix (greyscale)
    :param im2: second image matrix (greyscale)
    :param N: size of the neighborhood (N x N)
    :return: two matrices of the same size as the input image that contains
            u and v components of the optical flow displacement vectors
    """
    im1 = im1 / 255.
    im2 = im2 / 255.
    img_diff = gausssmooth(im2 - im1, 1)
    I_x1, I_y1 = gaussderiv(im1, 1.5)
    I_x2, I_y2 = gaussderiv(im2, 1.5)
    I_x, I_y, I_t = (I_x1 + I_x2) / 2, (I_y1 + I_y2) / 2, cv2.filter2D(img_diff, -1, uniform_kernel(N))
    I_x2, I_y2 = I_x * I_x, I_y * I_y
    I_xy, I_xt, I_yt = I_x * I_y, I_x * I_t, I_y * I_t

    D = cv2.filter2D(I_x2, -1, uniform_kernel(N)) * \
        cv2.filter2D(I_y2, -1, uniform_kernel(N)) \
        - (cv2.filter2D(I_xy, -1,uniform_kernel(N)) * cv2.filter2D(I_xy, -1,uniform_kernel(N)))
    # replacing zeros with some epsilon, so the division is stable
    D[D == 0] = 1e-10

    u = - ((cv2.filter2D(I_y2, -1, uniform_kernel(N))) * \
        (cv2.filter2D(I_xt, -1, uniform_kernel(N))) - \
        ((cv2.filter2D(I_xy, -1, uniform_kernel(N)))
         * (cv2.filter2D(I_yt, -1, uniform_kernel(N)))))

    v = - ((cv2.filter2D(I_x2, -1, uniform_kernel(N))) * \
        (cv2.filter2D(I_yt, -1, uniform_kernel(N))) - \
        ((cv2.filter2D(I_xy, -1, uniform_kernel(N)))
         * (cv2.filter2D(I_xt, -1, uniform_kernel(N)))))

    u = u / D
    v = v / D
    # replacing values where division is not defined
    #u = np.divide(u, D, out=np.zeros_like(u), where=D != 0)
    #v = np.divide(v, D, out=np.zeros_like(v), where=D != 0)

    return u, v

def lucaskanade_unstable(im1, im2, N):
    """
    unstable algorithm for the purpose of comparing with the stable
    :param im1: first image matrix (greyscale)
    :param im2: second image matrix (greyscale)
    :param N: size of the neighborhood (N x N)
    :return: two matrices of the same size as the input image that contains
            u and v components of the optical flow displacement vectors
    """
    im1 = im1 / 255.
    im2 = im2 / 255.
    img_diff = gausssmooth(im2 - im1, 1)
    I_x1, I_y1 = gaussderiv(im1, 1)
    I_x2, I_y2 = gaussderiv(im2, 1)
    I_x, I_y, I_t = (I_x1 + I_x2) / 2, (I_y1 + I_y2) / 2, cv2.filter2D(img_diff, -1, uniform_kernel(N))
    I_x2, I_y2 = I_x * I_x, I_y * I_y
    I_xy, I_xt, I_yt = I_x * I_y, I_x * I_t, I_y * I_t

    D = cv2.filter2D(I_x2, -1, uniform_kernel(N)) * \
        cv2.filter2D(I_y2, -1, uniform_kernel(N)) \
        - (cv2.filter2D(I_xy, -1,uniform_kernel(N)) * cv2.filter2D(I_xy, -1,uniform_kernel(N)))

    u = - ((cv2.filter2D(I_y2, -1, uniform_kernel(N))) * \
        (cv2.filter2D(I_xt, -1, uniform_kernel(N))) - \
        ((cv2.filter2D(I_xy, -1, uniform_kernel(N)))
         * (cv2.filter2D(I_yt, -1, uniform_kernel(N)))))
    u = u / D

    v = - ((cv2.filter2D(I_x2, -1, uniform_kernel(N))) * \
        (cv2.filter2D(I_yt, -1, uniform_kernel(N))) - \
        ((cv2.filter2D(I_xy, -1, uniform_kernel(N)))
         * (cv2.filter2D(I_xt, -1, uniform_kernel(N)))))
    v = v / D

    return u, v

def horn_schunck(im1, im2, n_iters, lmbd):
    """
    :param im1: first image matrix (greyscale)
    :param im2: second image matrix (greyscale)
    :param n_iters: number of iterations
    :param lmbd: parameter
    :return: two matrices of the same size as the input image that contains
            u and v components of the optical flow displacement vectors
    """
    im1 = im1 / 255.
    im2 = im2 / 255.
    u, v = np.zeros(im1.shape), np.zeros(im1.shape)
    L_d = np.array([[0, 1 / 4, 0], [1 / 4, 0, 1 / 4], [0, 1 / 4, 0]])
    I_x1, I_y1 = gaussderiv(im1, 1)
    I_x2, I_y2 = gaussderiv(im2, 1)
    I_t = im2 - im1
    I_x, I_y = (I_x1 + I_x2) / 2, (I_y1 + I_y2) / 2
    I_x2, I_y2 = I_x * I_x, I_y * I_y
    for _ in range(n_iters):
        u_a = cv2.filter2D(u, -1, L_d)
        v_a = cv2.filter2D(v, -1, L_d)
        D = lmbd + I_x2 + I_y2
        P = I_x * u_a + I_y * v_a + I_t
        u = u_a - I_x * (P / D)
        v = v_a - I_y * (P / D)
    return u, v

def horn_schunck_from_LK(im1, im2, u, v, n_iters, lmbd):
    """
    faster hs algorithm
    :param im1: first image matrix (greyscale)
    :param im2: second image matrix (greyscale)
    :param u: output from LK
    :param v: output from LK
    :param n_iters: number of iterations
    :param lmbd: parameter
    :return: two matrices of the same size as the input image that contains
            u and v components of the optical flow displacement vectors
    """
    im1 = im1 / 255.
    im2 = im2 / 255.
    L_d = np.array([[0, 1 / 4, 0], [1 / 4, 0, 1 / 4], [0, 1 / 4, 0]])
    I_x1, I_y1 = gaussderiv(im1, 1)
    I_x2, I_y2 = gaussderiv(im2, 1)
    I_t = im2 - im1
    I_x, I_y = (I_x1 + I_x2) / 2, (I_y1 + I_y2) / 2
    I_x2, I_y2 = I_x * I_x, I_y * I_y
    for _ in range(n_iters):
        u_a = cv2.filter2D(u, -1, L_d)
        v_a = cv2.filter2D(v, -1, L_d)
        D = lmbd + I_x2 + I_y2
        P = I_x * u_a + I_y * v_a + I_t
        u = u_a - I_x * (P / D)
        v = v_a - I_y * (P / D)
    return u, v