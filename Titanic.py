import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, f1_score

#loading the dataset
TitanicData = pd.read_csv("COMP1816_Titanic_Dataset_Classification.csv")

#removes rows where "Survival" is missing
TitanicData = TitanicData.dropna(subset=["Survival"])

#adding features and target variable, data that is used (x) to make predictions (y)
X = TitanicData[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = TitanicData["Survival"]  # Target variable (survival status)

#split the dataset into training and test sets
TrainingX = X.iloc[:-140]
TestX = X.iloc[-140:]
TrainingY = y.iloc[:-140]
TestY = y.iloc[-140:]

Format = ColumnTransformer(transformers=[("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]), ["Age", "SibSp", "Parch", "Fare"]),("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), ["Pclass", "Sex", "Embarked"])])

#define models
ModelTypes = {"Logistic Regression": LogisticRegression(max_iter=200),"K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),"Support Vector Machine (SVM)": SVC(kernel="rbf", probability=True)}

#train and evaluate models
Data = {}

for ModelName, ModelType in ModelTypes.items():
    Formatting = Pipeline(steps=[("preprocessor", Format), ("model", ModelType)])
    Formatting.fit(TrainingX, TrainingY)
    Results = Formatting.predict(TestX)
    Accuracy = accuracy_score(TestY, Results)
    Precision = precision_score(TestY, Results, zero_division=1)
    F1 = f1_score(TestY, Results, zero_division=1)
    Data[ModelName] = {"Accuracy": Accuracy,"Precision": Precision,"F1-score": F1}

#convert results to a DataFrame and display
ResultsDF = pd.DataFrame(Data).T.map(lambda x: "{:.2f}".format(x))
print(ResultsDF)
