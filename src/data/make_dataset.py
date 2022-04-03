# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd


def create_dataset(input_dir, output_dir):
    """Create a dataset from a image-name database.
    Args:
        input_dir (str): Path to the json file.
        output_dir (str): Path to the output directory.
    """
    logging.info("Creating dataset from {} to {}".format(
        input_dir, output_dir))
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.exists():
        logging.error("Input directory does not exist: {}".format(input_dir))
        return
    if not output_dir.exists():
        logging.info("Output directory does not exist: {}".format(output_dir))
        output_dir.mkdir(parents=True)
    # read the index.json file
    df = pd.read_json(input_dir / 'index.json')
    # create a table of label and name.
    # using integer(label) to represent the chinese character
    #  (string) for data mapping.
    temp = df['name'].value_counts().to_frame()
    temp.reset_index(inplace=True)
    temp.rename(columns={'name': 'count', 'index': 'name'}, inplace=True)
    temp.reset_index(inplace=True)
    temp.rename(columns={'index': 'label'}, inplace=True)

    temp.to_csv(str(output_dir / 'label_name.csv'), index=False)
    logging.info("Done creating label_name_count table.")

    temp.drop(columns=['count'], inplace=True)
    df_merge = pd.merge(df, temp, on='name')
    df_merge.to_csv(str(output_dir / 'image_name_label.csv'), index=False)

    logging.info("Done creating dataset")


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    create_dataset(input_filepath, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()