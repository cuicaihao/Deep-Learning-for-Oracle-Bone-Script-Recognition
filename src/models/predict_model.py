# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from skimage import transform
from matplotlib.pylab import plt
from pathlib import Path

import os
import sys
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")
# plt.ion()  # interactive mode
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
print(os.path.dirname(os.path.realpath(__file__)))

plt.rcParams['font.family'] = 'Arial Unicode MS'  # enable chinese character
pd.options.display.float_format = '{:,.8f}'.format


class Net(nn.Module):

    def __init__(self, class_number):
        super(Net, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # a. half-of-size

            # Defining another 2D convolution layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # a. half-of-size
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 5 * 5, class_number)  # 793 
            # nn.Linear(800, class_number)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class PredictModel:

    def __init__(self, model_path='model_best.pt', label_path='label.csv'):
        self.model_path = model_path
        # self.model = torch.load(self.model_path)
        self.model = Net(class_number=793)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to('cpu')
        self.label2name_frame = pd.read_csv(label_path)
        self.id2name = dict(
            zip(self.label2name_frame.label, self.label2name_frame.name))
        self.image = None

    def process_image_np(self, image_path):
        if type(image_path) == str:
            if Path(image_path).suffix == '.jpg':
                image_raw = plt.imread(image_path)
                image_raw = image_raw[:, :, 0]
            elif Path(image_path).suffix == '.npy':
                image_raw = np.load(image_path)
                image_raw = image_raw[:, :, 0]
        else:
            image_raw = image_path[:, :, 0]
        image = transform.resize(image_raw, (40, 40))
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        image = torch.from_numpy(image).float()
        self.image = image  # update the image
        return image  # return image # torch.tensor

    def predict(self, image_path):
        image = self.process_image_np(image_path)
        pred = self.model(image)
        pred_label = pred.argmax(dim=1).item()
        pred_character = self.id2name[pred_label]

        return pred_label, pred_character

    def predict_top10(self, image_path):
        image = self.process_image_np(image_path)
        pred = self.model(image)
        prob_dist = torch.sigmoid(pred).detach().numpy()[0]
        prediction_top_10 = self.label2name_frame.copy()
        prediction_top_10['prob'] = prob_dist
        prediction_top_10 = prediction_top_10.sort_values(
            by='prob', ascending=False).head(10)
        return prediction_top_10

    def show_result(self, pred_label, pred_character):
        title = 'Predicted: {}\nTrue: {}'.format(pred_label, pred_character)
        plt.imshow(self.image.squeeze().numpy(), cmap='gray')
        plt.title(title)
        plt.show()


@click.command()
@click.argument('input_model_filepath', type=click.Path())
@click.argument('input_label_name_filepath', type=click.Path())
@click.argument('input_image_filepath', type=click.Path())
def main(input_model_filepath, input_label_name_filepath,
         input_image_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
        # model_path = './models/model_best'
        # label_path = './data/processed/label_name.csv'
        # image_path = './data/raw/image/3653610.jpg'
    """
    logger = logging.getLogger(__name__)
    logger.info('input_model_filepath: {}'.format(input_model_filepath))
    logger.info(
        'input_label_name_filepath: {}'.format(input_label_name_filepath))
    logger.info('input_image_filepath: {}'.format(input_image_filepath))

    # load the model
    model = PredictModel(input_model_filepath, input_label_name_filepath)
    # make prediction
    pred_label, pred_character = model.predict(input_image_filepath)
    print("\nPredicted Label =", pred_label)
    print("\nChinese Character Label =", model.id2name[pred_label])
    # make top 10 prediction
    prediction_top_10 = model.predict_top10(input_image_filepath)
    print(prediction_top_10)
    # show the result
    model.show_result(pred_label, pred_character)
    print("Done Test")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
