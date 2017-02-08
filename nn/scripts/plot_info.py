import argparse
import numpy
import matplotlib.pyplot as plt


def main(exp):
    info = numpy.loadtxt(exp+"/info.csv", delimiter=',')

    n_plots = info.shape[1] - 1
    f, ax = plt.subplots(1, n_plots)
    for i in xrange(n_plots):
        x = info[:, 0]
        y = info[:, i+1]
        ax[i].plot(x, y)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp')
    args = parser.parse_args()
    main(**vars(args))

