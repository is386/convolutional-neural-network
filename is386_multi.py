# Indervir Singh
# is386
# CS615
# HW 3
# Question 5

# from matplotlib import pyplot as plt
import numpy as np

SIZE = (40, 40)
LINE1 = 25
LINE2 = 13
WIDTH = 2
STRIDE = 2
SEED = 1151999

LR = 0.1
TC = 50
L2_RT = 0.0001


def training(dataX, dataY, kernels, thetas):
    J = []
    for _ in range(TC):
        avg_errs = []
        all_kernel_grads = []
        for i, img in enumerate(dataX):
            kernel_grads = []
            for kernel in kernels:
                # Applies convolution
                f_map = convolution(img, kernel)

                # Applies max pooling
                Z, max_pos = max_pooling(f_map)

                # Flattens the pool
                h = Z.flatten()

                # Activation Function
                net = np.dot(h, thetas[i])
                y_hat = sigmoid(net)
                y = dataY[i]

                # Objective Function
                err = log_like(y, y_hat)
                avg_err = np.mean(err)
                avg_errs.append(avg_err)

                # Gradient of the Objective Function
                grad_theta = log_like_deriv(h, y, y_hat, thetas[i])

                # Gradient of the Kernel
                dJ = y_hat - y
                kernel_grad = back_convolution(
                    img, f_map, dJ, thetas[i], max_pos)
                kernel_grads.append(kernel_grad)
                thetas[i] = thetas[i] + (LR * grad_theta)

            all_kernel_grads.append(kernel_grads)

        # Update the Kernels
        kernel_grads = (
            (np.asarray(all_kernel_grads[0]) + np.asarray(all_kernel_grads[1])) / 2)
        kernels = kernels + (LR * kernel_grads)

        J.append(np.mean(np.asarray(avg_errs)))

    return kernels, thetas, J


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
            # Gets the window from the current row on the image until the width of the kernel. Same for the
            # column.
            window = img[row: row + f_width, col: col + f_width]
            # Selects values at the positions from where the max values were obtained.
            select_mat = select(window, max_pos).flatten('F')
            grad = dJ * theta.T @ select_mat
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
    return select_mat


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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_like(y, y_hat):
    return (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def log_like_deriv(h, y, y_hat, theta):
    l2 = 2 * (L2_RT * theta)
    return (h.T * (y - y_hat)) + l2


# def plot_j(J, fileName):
#     plt.plot(range(len(J)), J)
#     plt.xlabel("Iterations")
#     plt.ylabel("Average Log Likelihood")
#     plt.savefig(fileName, bbox_inches='tight')
#     plt.close()


# def plot_kernels(kernels, fileName):
#     for i, kernel in enumerate(kernels):
#         plt.imshow(kernel, interpolation='nearest')
#         plt.gray()
#         plt.savefig(fileName + str(i + 1) + ".png", bbox_inches='tight')
#         plt.close()


def main():
    np.random.seed(SEED)

    # Initializes the image data and labels.
    img0 = np.zeros(SIZE)
    img1 = np.zeros(SIZE)
    img0[:, LINE1] = 1
    img1[LINE2, :] = 1
    data = np.asarray([img0, img1])
    y = np.asarray([[0], [1]])

    # Initializes the kernels
    kernels = np.random.uniform(-0.000001, 0.000001, size=(4, 5, 5))

    # Initializes the weights
    thetas = np.random.uniform(-1, 1, size=(2, 324))

    kernels, thetas, avg_err = training(data, y, kernels, thetas)
    # plot_j(avg_err, "multi/mle_plot2.png")
    # plot_kernels(kernels, "multi/final_kernel_mle")

    # print("\nPlots saved in multi/")


if __name__ == "__main__":
    main()
