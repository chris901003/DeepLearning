def main():
    mode = __import__('second')
    print(mode.datas['remain'])


if __name__ == '__main__':
    main()
