import os
import sys

from logger import logging
from exception import CustomException

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Tuple
from data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
  train_data_path = os.path.join('artifacts', 'train.csv')
  test_data_path = os.path.join('artifacts', 'test.csv')
  raw_data_path = os.path.join('artifacts', 'data.csv')

class DataIngestion:
  def __init__(self) -> None:
    self.ingestion_config = DataIngestionConfig()

  def initiate_data_ingestion(self) -> Tuple[str, str]:
    logging.info('Initiating data ingestion')
    try:
      df = pd.read_csv('notebook/data/stud.csv')
      os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)


      df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
      logging.info('Saved raw data into artifacts')

      logging.info('Initiated train test split')
      train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

      train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
      logging.info('Saved train data into artifacts')

      test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
      logging.info('Saved test data into artifacts')

      return (
        self.ingestion_config.train_data_path,
        self.ingestion_config.test_data_path
      )

    except Exception as e:
      raise CustomException(e, sys)
    
if __name__ == '__main__':
  ingestion_obj = DataIngestion()
  train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()
  print(train_data_path, test_data_path)
  data_transformer = DataTransformation()
  train_arr, test_arr, _ = data_transformer.initiate_data_transformation(train_data_path, test_data_path)
  print(train_arr, test_arr)
