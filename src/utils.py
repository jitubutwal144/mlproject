import os
import sys
import pickle
from .exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path, obj):
  try:
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name, exist_ok=True)

    with open(file_path, 'wb') as file_obj:
      pickle.dump(obj, file_obj)

  except Exception as e:
    raise CustomException(e, sys)
  
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
  try:
    report = {}
    for i in range(len(list(models))):
      # Among multiple models, get current model
      model = list(models.values())[i]
      model_name = list(models.keys())[i]
      hyperparameter = params[model_name]

      # For each model train and get best parameters and finally get r2 score
      gs = GridSearchCV(model, hyperparameter, cv=3)
      gs.fit(X_train, y_train)

      # Get best parameters from above and train current model
      model.set_params(**gs.best_params_)
      model.fit(X_train, y_train)

      y_test_predicted = model.predict(X_test)
      test_model_score = r2_score(y_test, y_test_predicted)

      report[model_name] = test_model_score
    
    return report

  except Exception as e:
    raise CustomException(e, sys)
  
def load_object(file_path):
  try:
    with open(file_path, 'rb') as file_obj:
      return pickle.load(file_obj)
    
  except Exception as e:
    raise CustomException(e, sys)