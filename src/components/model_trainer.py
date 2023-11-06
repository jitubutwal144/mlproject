import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from sklearn.ensemble import (
  RandomForestRegressor,
  AdaBoostRegressor,
  GradientBoostingRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.utils import save_object, evaluate_model

from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
  def __init__(self) -> None:
    self.model_trainer_config = ModelTrainerConfig()

  def initiate_model_trainer(self, train_arr, test_arr):
    try:
      X_train, y_train, X_test, y_test = (
        train_arr[:, : -1], # all rows, exclude last column
        train_arr[:, -1], # all rows, last column
        test_arr[:, : -1], # all rows, exclude last column
        test_arr[:, -1] # all rows, last column
      )

      logging.info('Prepared X_train, y_train, X_test, y_test data')

      # Algorithms to tryout
      models = {
        'Random Forest': RandomForestRegressor(),
        'Decision Tree': DecisionTreeRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'Linear Regression': LinearRegression(),
        'XGBRegressor': XGBRegressor(),
        'AdaBoost Regressor': AdaBoostRegressor(),
        'CatBoosting Regressor': CatBoostRegressor(verbose=False)
      }

      # Pramas for hyprparameter tunning
      params = {
          "Decision Tree": {
              'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
              # 'splitter':['best','random'],
              # 'max_features':['sqrt','log2'],
          },
          "Random Forest":{
              # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            
              # 'max_features':['sqrt','log2',None],
              'n_estimators': [8,16,32,64,128,256]
          },
          "Gradient Boosting":{
              # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
              'learning_rate':[.1,.01,.05,.001],
              'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
              # 'criterion':['squared_error', 'friedman_mse'],
              # 'max_features':['auto','sqrt','log2'],
              'n_estimators': [8,16,32,64,128,256]
          },
          "Linear Regression":{},
          "XGBRegressor":{
              'learning_rate':[.1,.01,.05,.001],
              'n_estimators': [8,16,32,64,128,256]
          },
          "CatBoosting Regressor":{
              'depth': [6,8,10],
              'learning_rate': [0.01, 0.05, 0.1],
              'iterations': [30, 50, 100]
          },
          "AdaBoost Regressor":{
              'learning_rate':[.1,.01,0.5,.001],
              # 'loss':['linear','square','exponential'],
              'n_estimators': [8,16,32,64,128,256]
          }
          
      }
      model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

      logging.info(f"Model report: {model_report}")
      model_score_list = list(model_report.values())
      best_model_score = max(sorted(model_score_list))
      
      model_names_list = list(models.keys())
      best_model_name = model_names_list[model_score_list.index(best_model_score)]

      best_model = models[best_model_name]

      if best_model_score < 0.6:
        raise CustomException('Best model not found')
      
      logging.info(f"Found best model: {best_model_name}")

      save_object(
        file_path = self.model_trainer_config.trained_model_file_path,
        obj=best_model
      )
      y_test_predicted = best_model.predict(X_test)
      r2_square = r2_score(y_test, y_test_predicted)

      return r2_square
    
    except Exception as e:
      raise CustomException(e, sys)