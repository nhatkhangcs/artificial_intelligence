import pandas as pd
import statistics as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

columnNames = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin'
               , 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

dataset = pd.read_excel("E:\\SOFT\\General_Subjects\\artificial_intelligence\\AI_projects\\pima_diabetes\\diabetes.xlsx", names = columnNames)
#skinCol = dataset["SkinThickness"]

#print(skinCol)
print(dataset.describe())

dataset["SkinThickness"] = dataset["SkinThickness"].replace(0, st.mean(dataset["SkinThickness"]))
dataset["BMI"] = dataset["BMI"].replace(0, st.mean(dataset["BMI"]))

dataset.loc[dataset["DiabetesPedigreeFunction"] > 1.0, "DiabetesPedigreeFunction"] = 1.0

labels = dataset.pop('Outcome')

X_train, X_test, y_train, y_test = train_test_split(dataset, labels, train_size = 0.8)

model = LogisticRegression(max_iter = 800)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

sample = model.predict(X_test)

for i in sample:
    print(i)

print("\n")

for i in y_test:
    print(i)
        