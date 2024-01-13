

from gpl_improved.utils import BEIR_DATASETS 
from gpl_improved.utils import load_dataset
import click
import os
import logging

# Configure the logging module
logging.basicConfig(level=logging.DEBUG)

@click.command()
@click.option('--data_name', type=click.Choice(os.listdir('./bm25_scores')), help='The name of the data')
@click.option("--output_folder", type=str, help="Output folder to save data")
def main(data_name, output_folder):
    path = load_dataset(dataset_name=BEIR_DATASETS(data_name), output_folder=output_folder)
    print(f"Saved {data_name} to {path}")    
    
if __name__ == "__main__":
    main()