import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('framingham.csv')
df
df.shape
df.info()
df.drop(['education'],axis=1,inplace=True) #unrelated attribute
df.isnull().sum()
df.dropna(axis=0, inplace=True)
df.shape
df.info()
df
#Exploratory Data Analysis(EDA)
#distribution of the data
fig = plt.figure(figsize = (15,20))
ax = fig.gca()
df.hist(ax = ax)
plt.show()
df.corr()
plt.figure(figsize = (10,10))
sns.heatmap(df.corr()>0.5,annot=True)
sns.countplot(x=df['TenYearCHD'])
#Number of people who have disease vs age
plt.figure(figsize=(15,6))
sns.countplot(x='age',data = df, hue = 'TenYearCHD',palette='husl')
plt.show()
#Number of people who have disease vs age
ax = sns.countplot(x='male',data = df, hue = 'TenYearCHD',palette='husl')
for p in ax.patches:
      ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() ))
plt.show()
sns.countplot(data=df, x='currentSmoker', hue='TenYearCHD',palette='husl')
plt.show()
#Without Feature Selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = df.drop(['TenYearCHD'],axis=1)
scaler=StandardScaler() #Use Standard Scaler to normalize data on one scale.
X=scaler.fit_transform(X)
Y = df['TenYearCHD']
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.20,random_state=11, stratify= Y)
sns.countplot(x=Y_train)
plt.title("Before SMOTE")
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state = 11,sampling_strategy='minority')
X_train, Y_train = smote.fit_resample(X_train, Y_train)
sns.countplot(x=Y_train)
plt.title("After SMOTE")
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score
def report(model):
    model.fit(X_train,Y_train)
    model_predict = model.predict(X_test)
    accuracy = accuracy_score(Y_test,model_predict)
    print("Accuracy: ",accuracy)
    print("Confusion Matrix:\n",confusion_matrix(Y_test,model_predict))
    print("\nClassification Report:\n",classification_report(Y_test,model_predict))
    precision = precision_score(Y_test, model_predict)
    recall = recall_score(Y_test, model_predict)
    f1 = f1_score(Y_test, model_predict)
    return accuracy,precision,recall,f1
def roc(model):
    # ROC curve and AUC 
    probs = model.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    probs = probs[:, 1]
    
    # calculate AUC
    auc = roc_auc_score(Y_test, probs)

    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(Y_test, probs)
    
    # plot curve
    sns.set_style('whitegrid')
    plt.figure(figsize=(10,6))
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.title(f"AUC = {round(auc,3)}")
    plt.show()
    return auc
#Decision Tree
dtree= DecisionTreeClassifier(random_state=7)
dtacc,dtpre,dtrec,dtf1 = report(dtree)
dtauc = roc(dtree)
