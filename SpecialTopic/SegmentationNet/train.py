import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='segformer')
    parser.add_argument('--phi', type=str, default='m')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--pretrained', type=str, default='none')
    parser.add_argument('--load-from', type=str, default='none')
    parser.add_argument('--classes-path', type=str, default='./classes.txt')
    parser.add_argument('--data-prefix', type=str, default='')
    parser.add_argument('--train-annotation-path', type=str, default='./train_annotation.txt')
    parser.add_argument('--eval-annotation-path', type=str, default='./eval_annotation.txt')
    parser.add_argument('--auto-fp16', action='store_false')

    parser.add_argument('--Init-Epoch', type=int, default=0)
    parser.add_argument('')
    args = parser.parse_args()
    return args


def main():
    pass


if __name__ == '__main__':
    main()
