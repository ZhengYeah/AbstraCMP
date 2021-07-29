import numpy as np
import matplotlib.pyplot as plt


def mnist2img(filename):
    img_array = np.zeros(784)
    with open(filename, "r") as f:
        for i in range(784):
            line = f.readline().strip()
            img_array[i] = int(float(line) * 255)
    img_array = img_array.reshape(28, 28)
    plt.imshow(img_array, cmap="binary")
    plt.show()


if __name__ == "__main__":
    mnist2img("mnist_properties/mnist_property_22.txt")
