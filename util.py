import struct
import torch
import numpy as np
import os.path as osp
# import cv2


class Averager():
    def __init__(self):
        super().__init__()
        self.n = 0
        self.v = 0

    def add(self, correct, total):
        self.v += correct
        self.n += total

    def item(self):
        return self.v/self.n


def save_model(model, name):
    torch.save(model.state_dict(), osp.join('./model', name + '.pth'))


def count_acc(score, label, aver):
    pred = torch.argmax(score, dim=1)
    aver.add((pred == label.long()).type(
        torch.FloatTensor).sum().item(), score.size(0))
    return (pred == label.long()).type(torch.FloatTensor).mean().item()


def process_data(image_file_name, label_file_name):
    images = read_image(image_file_name)
    labels = read_label(label_file_name)
    assert np.size(images, 0) == np.size(labels, 0)
    return images, labels


def read_label(label_file_name):
    label_file = open(label_file_name, 'rb').read()
    offset = 0
    fmt_header = '>2i'
    magic_number, num = struct.unpack_from(fmt_header, label_file, offset)
    print('magic number:%d, number of items:%d' % (magic_number, num))
    offset += struct.calcsize(fmt_header)
    labels = np.zeros(num)
    i = 0
    fmt_header = '>1B'
    for i in range(num):
        labels[i], = struct.unpack_from(fmt_header, label_file, offset)
        offset += struct.calcsize(fmt_header)
    return labels


def read_image(image_file_name):
    image_file = open(image_file_name, 'rb').read()
    offset = 0
    fmt_header = '>4i'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(
        fmt_header, image_file, offset)
    print('magic number:%d, number of images:%d, number of rows:%d, number of cols:%d' % (
        magic_number, num_images, num_rows, num_cols))
    offset += struct.calcsize(fmt_header)
    image_size = num_rows * num_cols
    fmt_header = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    i = 0
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(
            fmt_header, image_file, offset)).reshape(num_rows, num_cols)
        offset += struct.calcsize(fmt_header)
        # cv2.imshow('debug', images[i])
        # cv2.waitKey(0)
    return images


if __name__ == "__main__":
    process_data('./data/train-images-idx3-ubyte',
                 './data/train-labels-idx1-ubyte')
