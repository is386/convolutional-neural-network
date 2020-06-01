# Indervir Singh
# is386
# CS615
# HW 3
# Question 3

# from matplotlib import pyplot as plt
import numpy as np

SIZE = (40, 40)
LINE1 = 25
LINE2 = 13
WIDTH = 1
STRIDE = 1
SEED = 1151999

LR = 0.01
TC = 100
L2_RT = 0.1


def training(dataX, dataY, kernel, thetas):
    J = []
    for _ in range(TC):
        avg_errs = []
        kernel_grads = []

        for i, img in enumerate(dataX):
            # Applies convolution
            f_map = convolution(img, kernel)

            # Applies max pooling
            Z, max_pos = max_pooling(f_map)

            # Flattens the pool
            h = Z.flatten()

            # Activation Function
            net = h * thetas[i]
            y_hat = soft_max(net)
            y = dataY[i]

            # Objective Function
            err = cross_entropy(y, y_hat)
            avg_errs.append(err)

            # Gradient of the Objective Function
            grad_theta = cross_entropy_grad(h, y, y_hat, thetas[i])

            # Gradient of the Kernel
            dJ = -1 / np.sum(y_hat)
            kernel_grad = back_convolution(
                img, f_map, dJ, np.sum(thetas[i]), max_pos)
            kernel_grads.append(kernel_grad.T[0].T)
            thetas[i] = thetas[i] - (LR * grad_theta)

        J.append(np.mean(np.asarray(avg_errs)))
        kernel_grad = (kernel_grads[0] + kernel_grads[1]) / 2
        kernel = kernel - (LR * kernel_grad)

    return kernel, thetas, J


def convolution(img, kernel):
    kernel = np.flip(kernel)
    img_h = img.shape[1]
    img_w = img.shape[0]
    k_width = kernel.shape[0]
    f_map = []
    row = 0
    # Counts the row that the top of the window is currently on and adds the width of the kernel to it.
    # If the sum is greater than the height of the image, then the window moves to the next row of the img.
    while row + k_width <= img_h:
        col = 0
        out = []
        # Counts the col that the side of the window is currently on and adds the width of the kernel to it.
        # If the sum is greater than the width of the image, then the window moves to the next col of the img.
        while col + k_width <= img_w:
            # Gets the window from the current row on the image until the width of the kernel. Same for the
            # column.
            window = img[row: row + k_width, col: col + k_width]
            conv = np.sum(np.multiply(kernel, window))
            out.append(conv)
            col += 1
        row += 1
        f_map.append(out)
    return np.asarray(f_map)


def back_convolution(img, f_map, dJ, theta, max_pos):
    img = np.flip(img)
    img_h = img.shape[1]
    img_w = img.shape[0]
    f_width = f_map.shape[0]
    out = []
    row = 0
    # Counts the row that the top of the window is currently on and adds the width of the kernel to it.
    # If the sum is greater than the height of the image, then the window moves to the next row of the img.
    while row + f_width <= img_h:
        col = 0
        out2 = []
        # Counts the col that the side of the window is currently on and adds the width of the kernel to it.
        # If the sum is greater than the width of the image, then the window moves to the next col of the img.
        while col + f_width <= img_w:
            window = img[row: row + f_width, col: col + f_width]
            # Selects values at the positions from where the max values were obtained.
            select_mat = select(window, max_pos).flatten('F')
            grad = dJ * theta.T * select_mat
            out2.append(grad)
            col += 1
        out.append(out2)
        row += 1
    return np.asarray(out)


def select(mat, max_pos):
    select_mat = []
    w = int(len(max_pos) / 2)
    if w == 0:
        w = 1
    # Goes through the max positions from max pooling and grabs the values at those positions
    # in the given window matrix
    for m in max_pos:
        select_mat.append((mat[m[0], m[1]])[0])
    select_mat = np.asarray(select_mat)
    return np.reshape(select_mat, (w, w))


def max_pooling(f_map):
    max_pos = []
    f_map_h = f_map.shape[1]
    f_map_w = f_map.shape[0]
    z = []
    row = 0
    # Counts the row that the top of the window is currently on and adds the width of the pool to it.
    # If the sum is greater than the height of the feature map, then the window moves to the next row of
    # the feature map.
    while row + WIDTH <= f_map_h:
        col = 0
        out = []
        # Counts the col that the side of the window is currently on and adds the width of the pool to it.
        # If the sum is greater than the width of the image, then the window moves to the next col of
        # the feature map.
        while col + WIDTH <= f_map_w:
            # Gets the window from the current row on the feature map until the width of the pool. Same for the
            # column.
            window = f_map[row:row + WIDTH, col:col + WIDTH]
            # Obtains the max value in the window
            max_val = np.max(window)
            out.append(max_val)
            max_pos.append(np.where(f_map == max_val))
            col += STRIDE
        row += STRIDE
        z.append(out)
    return np.asarray(z), max_pos


def soft_max(z):
    return np.exp(z) / np.sum(np.exp(z))


def cross_entropy(y, y_hat):
    return (-1 * np.sum(y * np.log(y_hat)))


def cross_entropy_grad(h, y, y_hat, theta):
    l2 = 2 * (L2_RT * theta)
    return (h.T * (y_hat - y)) - l2


# def plot_j(J, fileName):
#     plt.plot(range(len(J)), J)
#     plt.xlabel("Iterations")
#     plt.ylabel("Average Cross Entropy")
#     plt.savefig(fileName, bbox_inches='tight')
#     plt.close()


# def plot_kernel(kernel, fileName):
#     plt.imshow(kernel, interpolation='nearest')
#     plt.gray()
#     plt.savefig(fileName, bbox_inches='tight')
#     plt.close()


def main():
    np.random.seed(SEED)

    # Initializes the image data and labels.
    img0 = np.zeros(SIZE)
    img1 = np.zeros(SIZE)
    img0[:, LINE1] = 255
    img1[LINE2, :] = 255
    data = np.asarray([img0, img1])
    y = np.asarray([[1, 0], [0, 1]])

    # Initializes the kernels
    kernel = np.random.uniform(-0.001, 0.001, size=SIZE)
    # plot_kernel(kernel, "lce/init_kernel_lce.png")

    # Initializes the weights
    thetas = np.random.uniform(-0.001, 0.001, size=(2, 2))

    kernel, thetas, avg_err = training(data, y, kernel, thetas)
    # plot_j(avg_err, "lce/lce_plot.png")
    # plot_kernel(kernel, "lce/final_kernel_lce.png")

    # print("\nPlots saved in lce/")


if __name__ == "__main__":
    main()
