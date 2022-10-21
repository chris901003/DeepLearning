from importlib import import_module
import sys
import os


def main():
    file_path = '/Users/huanghongyan/Documents/DeepLearning/Test/DynamicImport/second.py'
    base_path = os.path.dirname(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    sys.path.insert(0, base_path)
    mod = import_module(file_name)
    sys.path.pop(0)
    print(mod.datas)


if __name__ == '__main__':
    main()
