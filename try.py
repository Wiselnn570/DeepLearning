from math import ceil, floor
import numpy as np
from matplotlib import pyplot as plt

def Print(name = "wxl"):
    print("my name is %s" % (name))


class wxl:
    def __init__(self, name = "wxl"):
        self.name = name
    def __call__(self):
        print("my name is %s" % (self.name))

def Try(a, b, *, name):
    print("! {}".format(name))


if __name__ == "__main__":
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
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    for i, (a, b) in enumerate(zip(x, y)):
        print("position %d, y / x = %d" % (i, b / a))