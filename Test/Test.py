from functools import partial
import logging


def main():
    dev_logger = logging.getLogger(name='dev')
    dev_logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    dev_logger.addHandler(handler)

    file_handler = logging.FileHandler('./log_file.txt')
    file_handler.setFormatter(formatter)
    # dev_logger.addHandler(file_handler)

    dev_logger.debug('debug message')
    dev_logger.info('info message')
    dev_logger.warning('warning message')
    dev_logger.error('error message')
    dev_logger.critical('critical message')


if __name__ == '__main__':
    main()
