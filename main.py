import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mnist import Mnist
from model import Convnet
from util import process_data, count_acc, Averager, save_model


def test():
    images, labels = process_data('./data/t10k-images-idx3-ubyte',
                                  './data/t10k-labels-idx1-ubyte')
    test_set = Mnist(images, labels)
    # train_loader = DataLoader(train_set, batch_size=64,
    #                           shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=64,
                             shuffle=True)
    model = Convnet()
    model.load_state_dict(torch.load('./model/model-1.pth'))
    model.eval()
    aver = Averager()
    for i, batch in enumerate(test_loader, 1):
        # image, label = [_.cuda() for _ in batch]
        image, label = batch
        score = model(image)
        count_acc(score, label, aver)
    print('test acc: %f' % aver.item())


def train():
    images, labels = process_data('./data/train-images-idx3-ubyte',
                                  './data/train-labels-idx1-ubyte')
    train_set = Mnist(images, labels)
    # train_loader = DataLoader(train_set, batch_size=64,
    #                           shuffle=True, num_workers=8, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=64,
                              shuffle=True)
    model = Convnet()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)
    aver = Averager()
    for epoch in range(1, 11):
        lr_scheduler.step()
        model.train()
        for i, batch in enumerate(train_loader, 1):
            # image, label = [_.cuda() for _ in batch]
            image, label = batch
            score = model(image)
            loss = F.cross_entropy(score, label.long())
            acc = count_acc(score, label, aver)
            print('epoch %d batch %d acc: %f' % (epoch, i, acc))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch %d acc: %f' % (epoch, aver.item()))
    save_model(model, 'model-1')


if __name__ == "__main__":
    train()
    # test()
