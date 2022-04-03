# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import matplotlib.pyplot as plt
from matplotlib import font_manager

import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import try_all_threshold
from skimage.filters import threshold_otsu

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")
fontP = font_manager.FontProperties()
# fontP.set_family('Arial Unicode MS')
# print(set([f.name for f in font_manager.fontManager.ttflist]))
plt.rcParams['font.family'] = 'Arial Unicode MS'


def check_samples(image_dir, table_dir):
    """
    Read one image sample and check the potential enhancement method.
    """
    logging.info("Checking samples in {}".format(image_dir))

    image_dir = Path(image_dir)

    logging.info("Checking samples in {}".format(table_dir))
    df = pd.read_csv(table_dir)

    img = imread(str(image_dir / df.image[0]))
    plt.imshow(img, cmap='gray')
    plt.title(f"{df.name[0]}: {img.shape}")
    plt.colorbar()
    plt.show()

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img[:, :, 0], cmap='gray')
    axs[1].imshow(img[:, :, 1], cmap='gray')
    axs[2].imshow(img[:, :, 2], cmap='gray')
    plt.show()

    # Convert the image to grayscale (3D to 2D)
    img_gray = rgb2gray(img)
    # Compare the grayscale into different binary threshold
    fig, ax = try_all_threshold(img_gray, verbose=False)
    plt.show()

    thresh = threshold_otsu(img_gray)
    img_bin = img_gray > thresh
    # plt.imshow(img_bin, cmap='gray')
    plt.imshow(img_bin, cmap='gray')
    plt.title(f"{df.name[0]}: {img_bin.shape}")
    plt.colorbar()
    plt.show()

    # show random image by id
    plotimage(str(image_dir / df.image[0]), df.name[0])
    plotimage(str(image_dir / df.image[300]), df.name[300])
    logging.info("Done Review Image Sample")


def plotimage(image_filepath, image_title):
    # df.image[id], df.name[id]

    img = imread(image_filepath)
    img_gray = rgb2gray(img)
    # thresh = threshold_otsu(img_gray)
    # img_bin = img_gray > thresh
    plt.imshow(img_gray, cmap='gray')
    plt.colorbar()
    plt.title(f"{image_title}: {img_gray.shape}")
    plt.show()
    return True


def check_image_loader(image_dir, table_dir, dict_dir):
    """
    Read one image sample and check the potential enhancement method.
    """
    logging.info("Checking samples in {}".format(image_dir))

    image_dir = Path(image_dir)

    logging.info("Checking samples in {}".format(table_dir))
    oracle_frame = pd.read_csv(table_dir)

    # logging.info("Checking samples in {}".format(dict_dir))
    # label2name_frame = pd.read_csv(dict_dir)

    # check the image_filename and id and label
    id = np.random.randint(1500)
    img_file = oracle_frame.iloc[id, 0]
    script_name = oracle_frame.iloc[id, 1]
    label = oracle_frame.iloc[id, 2]
    print('Image file name: {}'.format(img_file))
    print('Chinese Oracle Script: {}'.format(script_name))
    print('Script Label / index: {}'.format(label))

    return True


@click.command()
@click.argument('input_image_filepath', type=click.Path(exists=True))
@click.argument('input_label_filepath', type=click.Path())
@click.argument('input_label_name_filepath', type=click.Path())
def main(input_image_filepath, input_label_filepath,
         input_label_name_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('image_dir: {}'.format(input_image_filepath))
    logger.info('image_labe_dir: {}'.format(input_label_filepath))
    logger.info('image_label_name_dir: {}'.format(input_label_name_filepath))

    # check the image
    check_samples(input_image_filepath, input_label_filepath)  # passed

    check_image_loader(input_image_filepath, input_label_filepath,
                       input_label_name_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
