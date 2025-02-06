import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

#loading the dataset
HousingData = pd.read_csv("COMP1816_Housing_Dataset_Regression.csv")

#adding features and target variable, data that is used (x) to make predictions (y)
x = HousingData[["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"]]
y = HousingData["median_house_value"]

#split the dataset into training and test sets
TrainingX = x.iloc[:-190]
TestX = x.iloc[-190:]
TrainingY = y.iloc[:-190]
TestY = y.iloc[-190:]

#preprocessing to handle missing values and prepares for tuning
Format = ColumnTransformer(transformers=[("num", SimpleImputer(strategy="mean"), x.drop(columns=["ocean_proximity"]).columns),("cat", OneHotEncoder(handle_unknown="ignore"), ["ocean_proximity"])])
TrainingX_transformed = Format.fit_transform(TrainingX) 
TestX_transformed = Format.transform(TestX) 

#define models
ModelTypes = {"Multiple Linear Regression": LinearRegression(),"Decision Tree": DecisionTreeRegressor(random_state=42),"Random Forest": RandomForestRegressor(random_state=42),}

#hyperparameter tuning
DecisionTreeParams = {"max_depth": [5, 10, 15, None], "min_samples_split": [2, 5, 10],"min_samples_leaf": [1, 2, 5] }
DTGrid = GridSearchCV(DecisionTreeRegressor(random_state=42), DecisionTreeParams, cv=5, scoring="neg_mean_squared_error")
DTGrid.fit(TrainingX_transformed, TrainingY) 
BestDecisionTree = DTGrid.best_estimator_  

RandomForestParams = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20],"min_samples_split": [2, 5, 10],"min_samples_leaf": [1, 2, 4] }
RFGrid = GridSearchCV(RandomForestRegressor(random_state=42), RandomForestParams, cv=5, scoring="neg_mean_squared_error")
RFGrid.fit(TrainingX_transformed, TrainingY)  
BestRandomForest = RFGrid.best_estimator_

print("Decision Tree Parameters:", DTGrid.best_params_)
print("Random Forest Parameters:", RFGrid.best_params_)

#define new models
TunedModels = {"Multiple Linear Regression": ModelTypes["Multiple Linear Regression"], "Decision Tree": BestDecisionTree,"Random Forest": BestRandomForest }

#train and evaluate models
Data = {}

for ModelName, ModelType in TunedModels.items():
    Formatting = Pipeline(steps=[("preprocessor", Format), ("model", ModelType)])
    Formatting.fit(TrainingX, TrainingY)
    Results = Formatting.predict(TestX)
    MSEResults = mean_squared_error(TestY, Results)
    RMSEResults = np.sqrt(MSEResults)
    R2Results = r2_score(TestY, Results)
    
    Data[ModelName] = {"MSE": MSEResults, "RMSE": RMSEResults, "R2": R2Results}

#convert results to a DataFrame and display
print(pd.DataFrame(Data).T.map(lambda x: "{:,.2f}".format(x)))
