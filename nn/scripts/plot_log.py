import argparse
import numpy
import matplotlib.pyplot as plt


def get_logs(exp, y_idx):
    train_log = numpy.loadtxt(exp + "/log.train.csv", delimiter=',')
    assert train_log.ndim == 2, '%d dims for train log, expected 2' % train_log.ndim
    train_x = train_log[:, 0]
    train_y = train_log[:, y_idx]
    val_log = numpy.loadtxt(exp + "/log.valid.csv", delimiter=',')
    assert val_log.ndim == 2, '%d dims for val log, expected 2' % val_log.ndim
    val_x = val_log[:, 0]
    val_y = val_log[:, y_idx]
    return train_x, train_y, val_x, val_y


def get_old_logs(exp):
    log = numpy.loadtxt(exp+"/log.csv", delimiter=',')
    train_x = log[log[:, 0] == 0, 1]
    train_y = log[log[:, 0] == 0, 2]
    val_x = log[log[:, 0] == 1, 1]
    val_y = log[log[:, 0] == 1, 2]
    return train_x, train_y, val_x, val_y


def main(exp, old_logs, y_idx):
    if old_logs:
        assert y_idx == 1
        train_x, train_y, val_x, val_y = get_old_logs(exp)
    else:
        train_x, train_y, val_x, val_y = get_logs(exp, y_idx)

    plt.plot(train_x, train_y, label='train')
    plt.plot(val_x, val_y, label='val')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp')
    parser.add_argument('-y_idx', default=1, type=int)
    parser.add_argument('-old_logs', action='store_true', default=False)
    args = parser.parse_args()
    main(**vars(args))

