# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.data.make_dataloader import create_dataloader, show_oracle_character

import os
import sys
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")
# plt.ion()  # interactive mode
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
print(os.path.dirname(os.path.realpath(__file__)))


def show_loader_batch(dataloader):
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['label'].size())
        plt.figure(figsize=(10, 10))
        show_oracle_character(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        if i_batch == 0:
            break


def test_loader_function(root_dir, csv_file):

    training_loader, dataset = create_dataloader(csv_file,
                                                 root_dir,
                                                 batch_size=16,
                                                 rescale_size=45,
                                                 randomcrop_size=40)

    validation_loader, dataset = create_dataloader(csv_file,
                                                   root_dir,
                                                   batch_size=16,
                                                   rescale_size=45,
                                                   randomcrop_size=40,
                                                   datatype='validation')

    show_loader_batch(training_loader)
    show_loader_batch(validation_loader)


# defining the model architecture
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
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def run_main(root_dir, csv_file, label_name_file, output_dir):

    # root_dir='./data/raw/image/'
    # csv_file="./data/processed/image_name_label.csv"
    # label_name_file="./data/processed/label_name.csv"
    # output_dir="./models"

    training_loader, dataset = create_dataloader(csv_file,
                                                 root_dir,
                                                 batch_size=16,
                                                 rescale_size=45,
                                                 randomcrop_size=40)
    validation_loader, dataset = create_dataloader(csv_file,
                                                   root_dir,
                                                   batch_size=16,
                                                   rescale_size=45,
                                                   randomcrop_size=40,
                                                   datatype='validation')

    # get total class number
    label2name_frame = pd.read_csv(label_name_file)
    class_number = len(label2name_frame)
    # defining the model
    model = Net(class_number)
    # defining the loss function
    loss_fn = nn.CrossEntropyLoss()
    # defining the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(output_dir + '/runs/obs_{}'.format(timestamp))
    epoch_number = 0
    EPOCHS = 200
    best_vloss = 10.
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        running_loss = 0.
        last_loss = 0.
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs = data['image'].float()
            labels = data['label'].long()
            # inputs = Variable(inputs)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            # 1602/16 = 100 reports on the loss for every 25 batches.
            if i % 100 == 99:
                last_loss = running_loss / 100  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch * len(training_loader) + i + 1
                writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
                writer.add_images('mage_batch', inputs, epoch)
        avg_loss = last_loss

        # We don't need gradients on to do reporting
        model.train(False)
        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs = vdata['image'].float()
            vlabels = vdata['label'].long()
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss', {
            'Training': avg_loss,
            'Validation': avg_vloss
        }, epoch_number + 1)
        writer.flush()
        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            model_path = Path(output_dir) / 'model_best.pt'
            torch.save(model.state_dict(), str(model_path))
            writer.add_graph(model, vinputs)
        writer.flush()
        epoch_number += 1

    # Check the Final best model performance
    correct_count, all_count = 0, 0
    for data in validation_loader:
        images = data['image'].float()
        labels = data['label'].long()
        for i in range(len(labels)):
            # img = images[i].view(1, 1, 28, 28)
            img = images[i, :, :, :]
            img = torch.unsqueeze(img, dim=0)
            with torch.no_grad():
                logps = model(img)
            # ps = torch.exp(logps)
            # probab = list(ps.cpu()[0])
            pred_label = logps.argmax(1).item()
            true_label = labels.cpu()[i]
            if (true_label == pred_label):
                correct_count += 1
            all_count += 1

    print("Number Of Images =", all_count)
    print("\nModel Accuracy =", (correct_count / all_count))

    return True


@click.command()
@click.argument('input_image_filepath', type=click.Path(exists=True))
@click.argument('input_label_filepath', type=click.Path())
@click.argument('input_label_name_filepath', type=click.Path())
@click.argument('output_model_filepath', type=click.Path())
def main(input_image_filepath, input_label_filepath, input_label_name_filepath,
         output_model_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('image_dir: {}'.format(input_image_filepath))
    logger.info('image_labe_dir: {}'.format(input_label_filepath))
    logger.info('image_label_name_dir: {}'.format(input_label_name_filepath))
    logger.info('output_model_filepath: {}'.format(output_model_filepath))

    # check the image
    # test_loader_function(input_image_filepath, input_label_filepath) # PASS
    run_main(input_image_filepath, input_label_filepath,
             input_label_name_filepath, output_model_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
