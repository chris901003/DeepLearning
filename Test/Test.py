from functools import partial


def cal(a, b, c):
    print(a + b + c)


def main():
    cfg = dict(b=10, c=20)
    func = partial(cal, **cfg)
    func(20)


if __name__ == '__main__':
    main()
