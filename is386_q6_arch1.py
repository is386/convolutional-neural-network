# Indervir Singh
# is386
# CS615
# HW 3
# Question 6
# Architecture 1

# from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np

YALE_PATH = "./yalefaces/"
SIZE = (40, 40)

WIDTH = 1
STRIDE = 1
SEED = 1151999

LR = 0.01
TC = 25
L2_RT = 0.1
BIAS = 1

classes = {}


def parse_yale_faces():
    data_matrix = []
    yale_faces = [i for i in listdir(
        YALE_PATH) if isfile(join(YALE_PATH, i))]

    for face in yale_faces:
        try:
            face_img = Image.open(join(YALE_PATH, face))
            face_img = face_img.resize(SIZE)
            pixels = np.asarray(face_img).flatten()
            # pixels = np.insert(pixels, 0, BIAS)
            face_img.close()
            sub_n = parse_subj_num(face)
            pixels = np.append(pixels, sub_n)
            data_matrix.append(pixels)

            # saves each class and its total
            if sub_n not in classes:
                classes[sub_n] = 1
            else:
                classes[sub_n] += 1
        except OSError:
            pass

    return np.asarray(data_matrix)


def split_data(data_mat):
    training = np.zeros((1, data_mat.shape[1]))
    testing = np.zeros((1, data_mat.shape[1]))

    for i in classes.keys():
        num_training = 2 * round(classes[i] / 3)
        # gets all of the data for each subject, i
        subj_mat = data_mat[data_mat[:, -1] == i]
        # shuffles that subject's data to reduce bias
        np.random.shuffle(subj_mat)
        # adds 2/3s of the data to training, and 1/3 to testing
        training = np.append(training, subj_mat[0:num_training, :], axis=0)
        testing = np.append(testing, subj_mat[num_training:, :], axis=0)

    # reshuffle to reduce bias
    np.random.shuffle(training)
    np.random.shuffle(testing)
    # labels are in the last column of the data
    return training[1:, :-1], training[1:, -1], testing[1:, :-1], testing[1:, -1]


def parse_subj_num(subject):
    return int("".join(subject.split("subject")[1][0:2]))


def standardize(train_X, test_X):
    for i in range(1, train_X.shape[1]):
        s = np.std(train_X[:, i])
        m = np.mean(train_X[:, i])
        # if the std is 0, then set that feature to 0
        if s == 0:
            train_X[:, i] = 0
        train_X[:, i] = (train_X[:, i] - m) / s
        test_X[:, i] = (test_X[:, i] - m) / s


def create_labels(Y):
    labels = []
    for i in range(len(Y)):
        label = np.zeros((1, 15))
        # gets the subject number
        val = int(Y[i])
        # puts a one at the index representing the subject number
        label[0, val - 1] = 1
        labels.append(label)
    return np.asarray(labels).squeeze()[:, 1:]


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
        kernel_grad = kernel_grads[0]
        for k in kernel_grads[1:]:
            kernel_grad += k
        kernel_grad = kernel_grad / len(kernel_grads)

        kernel = kernel - (LR * kernel_grad)

    return kernel, thetas, J


def convolution(img, kernel):
    # Flips the kernel horizontally and vertically
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
    # Flips the image
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
            # Gets the window from the current row on the image until the width of the kernel. Same for the
            # column.
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


def test_network(x, y, kernel, theta):
    J = []
    correct = 0
    y_hats = []
    for i, img in enumerate(x):
        # Applies convolution
        f_map = convolution(img, kernel)

        # Applies max pooling
        Z, _ = max_pooling(f_map)

        # Flattens the pool
        h = Z.flatten()

        # Activation Function
        net = h * theta[i]
        y_hat = soft_max(net)
        y_hats.append(y_hat)

        # Objective Function
        err = cross_entropy(y[i], y_hat)
        J.append(np.mean(err))

    y_hats = np.asarray(y_hats)

    for i in range(y.shape[1]):
        # gets the index where the label contains a 1. represents the subject
        actual = np.where(y[i] == 1)[0][0]
        # gets the index where the label contains a 1. represents most likely subject
        guess = np.where(y_hats[i] == np.amax(y_hats[i]))[0][0]
        if int(actual) + 2 == guess + 2:
            correct += 1
    return (correct / y.shape[1]) * 100, J


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
    data_mat = parse_yale_faces()
    train_X, train_Y, test_X, test_Y = split_data(data_mat)
    standardize(train_X, test_X)

    temp = []
    for data in train_X:
        temp.append(np.reshape(data, SIZE))
    train_X = np.asarray(temp)

    temp = []
    for data in test_X:
        temp.append(np.reshape(data, SIZE))
    test_X = np.asarray(temp)

    train_labels = create_labels(train_Y)
    test_labels = create_labels(test_Y)

    # Initializes the kernels
    kernel = np.random.uniform(-0.000001, 0.000001, size=SIZE)

    # Initializes the weights
    thetas = np.random.uniform(-0.001, 0.001, size=(1600, len(classes)))

    kernel, thetas, avg_err_train = training(
        train_X, train_labels, kernel, thetas)

    acc0, _ = test_network(
        train_X, train_labels, kernel, thetas)

    acc1, avg_err_test = test_network(
        test_X, test_labels, kernel, thetas)

    print("Train Accuracy:", acc0)
    print("Test Accuracy:", acc1)
    # plot_kernel(kernel, "arch1/final_kernel_arch1.png")
    # plot_j(avg_err_train, "arch1/arch1_plot_train.png")
    # plot_j(avg_err_test, "arch1/arch1_plot_test.png")

    #print("\nPlots saved in arch1/")


if __name__ == "__main__":
    main()
