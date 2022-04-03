# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
# here put the import lib
# test this srcipt by python obs/make_dataset.py

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

plt.rcParams['font.family'] = 'Arial Unicode MS'


class OracleDataset(Dataset):
    """Oracle dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.oracle_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.oracle_frame)

    def __getitem__(self, idx):
        img_file = os.path.join(self.root_dir, self.oracle_frame.iloc[idx, 0])
        img = io.imread(img_file)
        # H, W, 1
        img = np.expand_dims(img[:, :, 0], axis=2)

        name = self.oracle_frame.iloc[idx, 1]
        label = self.oracle_frame.iloc[idx, 2]

        sample = {'image': img, 'name': name, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, name, label = sample['image'], sample['name'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return {'image': img, 'name': name, 'label': label}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, name, label = sample['image'], sample['name'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top:top + new_h, left:left + new_w]

        return {'image': image, 'name': name, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, name, label = sample['image'], sample['name'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {
            'image': torch.from_numpy(image),
            'name': name,
            'label': torch.tensor(label)
        }


# Helper function to show a batch
def show_oracle_character(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, name_batch = sample_batched['image'], sample_batched['name']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)  # H, W, C
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    for i in range(batch_size):
        # make_grid default padding is 2 to avoid occlusion of labels.
        plt.text((i % 8) * (im_size + 2) + 5,
                 (i // 8) * (im_size + 2) + 5 + 12,
                 name_batch[i],
                 fontsize=12,
                 color='r')
    plt.title('Batch from dataloader: Oracle Characters')


def create_dataloader(csv_file,
                      root_dir,
                      batch_size=64,
                      rescale_size=50,
                      randomcrop_size=48,
                      datatype='train'):
    """Create a dataloader for the given file and batch size."""
    dataset = OracleDataset(csv_file,
                            root_dir,
                            transform=transforms.Compose([
                                Rescale(rescale_size),
                                RandomCrop(randomcrop_size),
                                ToTensor()
                            ]))

    if datatype == 'train':
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=0)  # change the num_workers = 0 (instead of 4)
    elif datatype == 'validation':
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0)
    else:
        raise ValueError('Invalid datatype: {}'.format(datatype))
    return dataloader, dataset


def run_main(root_dir, csv_file):
    # csv_file = "../data/processed/image_name_label.csv"
    # root_dir = '../data/raw/image/'
    dataloader, dataset = create_dataloader(csv_file,
                                            root_dir,
                                            batch_size=64,
                                            rescale_size=50,
                                            randomcrop_size=48)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['label'].size())

        plt.figure()
        show_oracle_character(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()

        if i_batch == 0:
            break


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
    run_main(input_image_filepath, input_label_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
