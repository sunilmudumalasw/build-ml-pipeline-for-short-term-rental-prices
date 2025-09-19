#!/usr/bin/env python
"""
Performs basic cleaning on the data and saves the results in W&B
"""
import argparse
import logging
import wandb
import pandas as pd
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info(f'Download input artifact {args.input_artifact}')
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df_sample = pd.read_csv(artifact_local_path)

    # Drop outliers
    logger.info(f'Drop outliers regarding min {args.min_price}, max {args.max_price} price thresholds')
    min_price = args.min_price
    max_price = args.max_price
    idx = df_sample['price'].between(min_price, max_price)
    df_clean = df_sample[idx].copy()

   
    # Normal Distribution of 'minimum_nights'
    logger.info('Transform skewness of feature "minimum_nights" to normal distribution')
    df_clean['minimum_nights'] = np.log(df_clean['minimum_nights'])
    
    # Convert last_review to datetime
    logger.info('Convert feature "last_review" to datetime type')
    df_clean['last_review'] = pd.to_datetime(df_clean['last_review'])
    
    # Drop rows in the dataset that are not in the proper geolocation
    logger.info('Drop rows in the dataset that are not in the proper geolocation')
    idx = df_clean['longitude'].between(-74.25, -73.50) & df_clean['latitude'].between(40.5, 41.2)
    df_clean = df_clean[idx].copy()
    
    # Save cleaned dataframe
    logger.info(f'Save cleaned dataframe to {args.output_artifact_name}')
    df_clean.to_csv(args.output_artifact_name, index=False)

    # Log artifact to Weights & Biases
    logger.info(f'W&B logging artifact {args.output_artifact_name}') 
    artifact = wandb.Artifact(
        name=args.output_artifact_name,
        type=args.output_artifact_type,
        description=args.output_artifact_description,
    )
    artifact.add_file(args.output_artifact_name)
    run.log_artifact(artifact)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step cleans the data")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help='Input artifact as given (sample csv file)',
        required=True
    )

    parser.add_argument(
        '--output_artifact_name', 
        type=str,
        help='Output file name (e.g. cleaned_data.csv)',
        required=True
    )

    parser.add_argument(
        '--output_artifact_type', 
        type=str,
        help='Type of the output file (e.g. cleaned_data)',
        required=True
    )

    parser.add_argument(
        '--output_artifact_description', 
        type=str,
        help='Cleaned data (e.g. outliers, skewness removed; datetime convertion)',
        required=True
    )

    parser.add_argument(
        '--min_price', 
        type=float,
        help='Minimum price to filter the price data for (e.g 10)',
        required=True
    )

    parser.add_argument(
        '--max_price', 
        type=float,
        help='Maximum price to filter the price data for (e.g 350)',
        required=True
    )
    
    args = parser.parse_args()

    go(args)
