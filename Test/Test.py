from functools import partial


class Test:
    def __init__(self):
        self.a = 123


def main():
    t = Test()
    if hasattr(t, 'a'):
        print('a')


if __name__ == '__main__':
    main()
