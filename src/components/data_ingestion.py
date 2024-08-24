import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
# from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset')

            #Purpose: Ensures that the directory for saving the data files exists.
            # Details:
            # Uses os.path.dirname(self.ingestion_config.train_data_path) to get the directory path from the train_data_path.
            # Creates the directory if it doesn’t already exist (using exist_ok=True to avoid errors if the directory is already present).
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)


            #Saves the DataFrame df to 'artifacts/data.csv', which is specified by self.ingestion_config.raw_data_path. index=False ensures that the row indices are not saved, and header=True includes the column names.
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            #Returns a tuple with paths to the saved training and testing data files.
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            logging.error(f"Error in prediction pipeline: {str(e)}")
            raise CustomException(e,sys)
        

# python -m src.components.data_ingestion (run this command to check this file)
# if __name__ =="__main__":
#     obj=DataIngestion()
#     train_path, test_path = obj.initiate_data_ingestion()

# #checking both data_ingestion and data_transformation to see if preprocessor.pkl fil is being created or not.
#     data_transformation = DataTransformation()
#     data_transformation.initiate_data_transformation(train_path,test_path)