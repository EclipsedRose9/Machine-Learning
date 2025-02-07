import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score

#loading the dataset
TitanicData = pd.read_csv("COMP1816_Titanic_Dataset_Classification.csv")

#removes rows where "Survival" is missing
TitanicData = TitanicData.dropna(subset=["Survival"])

#adding features and target variable, data that is used (x) to make predictions (y)
X = TitanicData[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = TitanicData["Survival"]

#split the dataset into training and test sets
TrainingX = X.iloc[:-140]
TestX = X.iloc[-140:]
TrainingY = y.iloc[:-140]
TestY = y.iloc[-140:]

#reprocessing to handle missing values ,scale numbers and prepares for tuning
Format = ColumnTransformer(transformers=[("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")),("scaler", StandardScaler())]), ["Age", "SibSp", "Parch", "Fare"]),("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("encoder", OneHotEncoder(handle_unknown="ignore"))]), ["Pclass", "Sex", "Embarked"])])
TrainingX_transformed = Format.fit_transform(TrainingX)
TestX_transformed = Format.transform(TestX)

#hyperparameter tuning 
LogRegParams = {"C": [0.01, 0.1, 1, 10, 100]}
LogReg_Grid = GridSearchCV(LogisticRegression(max_iter=200), LogRegParams, cv=5, scoring="accuracy")
LogReg_Grid.fit(TrainingX_transformed, TrainingY)
BestLogReg = LogReg_Grid.best_estimator_

KNNParams = {"n_neighbors": [3, 5, 7, 9, 11]}
KNN_Grid = GridSearchCV(KNeighborsClassifier(), KNNParams, cv=5, scoring="accuracy")
KNN_Grid.fit(TrainingX_transformed, TrainingY)
BestKNN = KNN_Grid.best_estimator_

SVMParams = {"C": [0.1, 1, 10],  "kernel": ["linear", "rbf"]}
SVM_Grid = GridSearchCV(SVC(probability=True), SVMParams, cv=5, scoring="accuracy")
SVM_Grid.fit(TrainingX_transformed, TrainingY)
BestSVM = SVM_Grid.best_estimator_

print("Logistic Regression Parameters:", LogReg_Grid.best_params_)
print("KNN Parameters:", KNN_Grid.best_params_)
print("SVM Parameters:", SVM_Grid.best_params_)

#define models
TunedModels = {"Logistic Regression": BestLogReg,"K-Nearest Neighbors (KNN)": BestKNN,"Support Vector Machine (SVM)": BestSVM}

#train and evaluate models
Data = {}

for ModelName, ModelType in TunedModels.items():
    Formatting = Pipeline(steps=[("preprocessor", Format), ("model", ModelType)])
    Formatting.fit(TrainingX, TrainingY)
    Results = Formatting.predict(TestX)
    Accuracy = accuracy_score(TestY, Results)
    Precision = precision_score(TestY, Results, zero_division=1)
    F1 = f1_score(TestY, Results, zero_division=1)
    Data[ModelName] = {"Accuracy": Accuracy, "Precision": Precision, "F1-score": F1}

#convert results to a DataFrame and display
ResultsDF = pd.DataFrame(Data).T.map(lambda x: "{:.2f}".format(x))
print(ResultsDF)
