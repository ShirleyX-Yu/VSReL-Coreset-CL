# -*-coding:utf8-*-

import argparse
import os
from onedrivedownloader import download


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Download data')
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)

    print('Downloading dataset')
    ln = ("https://unimore365-my.sharepoint.com/:u:/g/personal/263133_unimore_it/"
          "EVKugslStrtNpyLGbgrhjaABqRHcE3PB_r2OEaV7Jy94oQ?e=9K29aD")
    download(ln, filename=os.path.join(args.data_path, 'tiny-imagenet-processed.zip'),
             unzip=True, unzip_path=args.data_path, clean=True)
