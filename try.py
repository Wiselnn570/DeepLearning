from math import ceil, floor
import numpy as np
from matplotlib import pyplot as plt
import torch


def Print(name: str = "wxl"):
    print("my name is %s" % (name))


class wxl:
    def __init__(self, name="wxl"):
        self.name = name

    def __call__(self):
        print("my name is %s" % (self.name))


def Try(a, b, *, name):
    print("! {}".format(name))


def Test(a, b):
    assert a > 1, a
    assert b < 0, b


def ret():
    return 1, 2


def HelloWord(a: bool = True) -> str:
    print("hello world")
    return "hello world"


if __name__ == "__main__":
    # Print()
    # my_name = wxl("Wiselnn")
    # my_name()
    # your_name = Print
    # your_name("lxw")
    # num = 3.2
    # print(ceil(num))
    # Try(1, 2, name="wxl")
    # x = np.linspace(0, 2*np.pi, 400)
    # y = np.sin(x**2)
    # f, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(3, 3))
    # plt.figure
    # axs[0, 0].plot(x, y)
    # axs[0, 1].set_title('Sharing Y axis')
    # axs[1, 1].scatter(x, y)
    # plt.show()
    # x = [1, 2, 3, 4, 5]
    # y = [2, 4, 6, 8, 10]
    # for i, (a, b) in enumerate(zip(x, y)):
    #     print("position %d, y / x = %d" % (i, b / a))
    # X = torch.zeros(10, requires_grad=True)
    # with torch.no_grad():
    #     y = X * 2
    #     print(y.requires_grad)
    # target = torch.randn(3, 5)
    # target_s = target.clone().softmax(dim=0).sum(dim=0)
    # print(target, '\n', target_s)

    # target = torch.empty(3, dtype=torch.long).random_(5)

    HelloWord(False)
