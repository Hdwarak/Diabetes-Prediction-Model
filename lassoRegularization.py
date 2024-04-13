import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV

filepath = '/Users/harikadwarakacharla/Downloads/diabetes.csv'
file=pd.read_csv(filepath)
file.size
file.head(20)

#feature
x=file[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
#target
y=file["Outcome"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#Random Forest model before doing feature selection
randForrest=RandomForestClassifier(random_state=56)
randForrest.fit(x_train,y_train)
y_predRandForrest=randForrest.predict(x_test)
accuracy_randfor=accuracy_score(y_test,y_predRandForrest)
print("Accuracy for Random Forest Model before feature selection :",accuracy_randfor*100)

lasso = LassoCV(cv=5, random_state=0).fit(x_train, y_train)

# Identifying non-zero coefficients (selected features)
selected_features = x_train.columns[lasso.coef_ != 0]
print(selected_features)

x_train_lasso = x_train[selected_features]
x_test_lasso = x_test[selected_features]

# Random Forest Model after doing feature selection
randomFor = RandomForestClassifier(random_state=58)
randomFor.fit(x_train_lasso, y_train)
y_predrandomFor = randomFor.predict(x_test_lasso)
accuracy_randfor = accuracy_score(y_test, y_predrandomFor)
print("Random Forest Model Accuracy after feature selection :", accuracy_randfor*100)


