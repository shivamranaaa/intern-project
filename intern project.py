import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
plt.rcParams['figure.figsize']=7,7

import warnings
warnings.filterwarnings("ignore")

df=pd.read_excel("Data_Cortex_Nuclear.xls")

df.head()

df.shape

df.info()

df.describe()


null_percenatge=((df.isnull().sum())/df.shape[0])*100
null_percenatge

null_percenatge[null_percenatge>1]


df.drop(["H3MeK4_N","BCL2_N"],axis=1,inplace=True)


null_percenatge=((df.isnull().sum())/df.shape[0])*100
null_percenatge



null_val_col=[i for i,j in null_percenatge.items() if j>0]

null_val_col

for i in null_val_col:
    df[f"{i}"]=df[f"{i}"].fillna(df[f"{i}"].median())

df.isnull().sum().sum()


num_col=[]
for i in df.columns:
    if df[f"{i}"].dtype=="float64" or df[f"{i}"].dtype=="int64":
        num_col.append(i) 


len(num_col)

#plt.rcParams["figure.figsize"]=(15,200)
#x=1
#for i in num_col:
#    plt.subplot(38,2,x)
#    sns.boxplot(df[f"{i}"])
#    x=x+1


# # Encoding

df.drop("MouseID",axis=1,inplace=True)


categorical_col=[i for i in df.columns if df[f"{i}"].dtype=="object"]
categorical_col=categorical_col[:-1]
categorical_col


for i in categorical_col:
    print(df[f"{i}"].value_counts())
    print("----------------------------------")


# Labe_Encoding

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in categorical_col:
    df[f"{i}"]=le.fit_transform(df[f"{i}"])
df["class"]=df["class"].map({"c-CS-m":0,"c-CS-s":1,"c-SC-m":2,"c-SC-s":3,"t-CS-m":4,"t-CS-s":5,"t-SC-m":6,"t-SC-s":7})


#  corr


plt.rcParams["figure.figsize"]=(30,30)
#sns.heatmap(df.corr(),annot=True,cmap="coolwarm")


# # Splting the Data

X=df.iloc[:,:-1]
y=df["class"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

y.value_counts()


# # Feature Selection

plt.figure(figsize=(12,10))
cor = X_train.corr()

# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.42)  #perfect  model at 0.525 
len(set(corr_features))


corr_features

X_train=X_train.drop(corr_features,axis=1)
X_test=X_test.drop(corr_features,axis=1)

X_train.shape


# # Model


from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

def model(yo):
    model_=yo()
    model_.fit(X_train,y_train)
    
    y_pred=model_.predict(X_test)
    
    cv_val=cross_val_score(model_,X,y,scoring="accuracy",cv=5)
    cv_score=np.mean(cv_val)
    
    print("Model Name is :",yo)
    print("Cross val score : ",cv_score)
    print("classification report")
    print(classification_report(y_pred,y_test))
    print("---------------------------------------------------------------")


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB


diff_models=[LogisticRegression,RandomForestClassifier,XGBClassifier]


for i in diff_models:
    model(i)


xg=XGBClassifier().fit(X_train,y_train)
    
y_pred=xg.predict(X_test)
    
cv_val=cross_val_score(xg,X,y,scoring="accuracy",cv=5)
cv_score=np.mean(cv_val)
    
print("Model Name is :",XGBClassifier)
print("Cross val score : ",cv_score)
print("classification report")
print(classification_report(y_pred,y_test))
print("---------------------------------------------------------------")

X_test


#  Pickle

import pickle

pickle.dump(xg, open("inter_model.pkl", 'wb'))


# testing the pickle file

import pickle
import numpy as np

my_model = pickle.load(open("inter_model.pkl", 'rb'))

#[0.182518,0.174822,1.104068,0.181133,0.252521,1]

my_model.predict(np.array([[0.182518,0.174822,1.104068,0.181133,0.252521,1]]))

#int_features = [0.182518,0.174822,1.104068,0.181133,0.252521,1]
int_features = [0.300101,0.215078,0.408875,0.171069,0.223381,1.000000]
final_features = np.array([int_features])
final = my_model.predict(final_features)
final

#{"c-CS-m":0,"c-CS-s":1,"c-SC-m":2,"c-SC-s":3,"t-CS-m":4,"t-CS-s":5,"t-SC-m":6,"t-SC-s":7}

if final==[0]:
    final="c-CS-m"
elif final==[1]:
    final="c-CS-s"
elif final==[2]:
    final="c-SC-m"
elif final==[3]:
    final="c-SC-s"
elif final==[4]:
    final="t-CS-m"
elif final==[5]:
    final="t-CS-s"
elif final==[6]:
    final="t-SC-m"
elif final==[7]:
    final="t-SC-s"
print(final)