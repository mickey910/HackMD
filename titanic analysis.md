# titanic analysis
###### tags: `data analysis`
{%hackmd BJrTq20hE %}

## 環境

請填寫當前執行使用的環境

|環境|名稱|版本|
|-|-|-|
|作業系統|windows|10 22H2|
|程式執行環境|jupyter notebook|6.4.12|
|python 版本|python3|3.6.9|
|安裝環境的方式|pip|22.2.2|  

## 計算資源

請填寫執行程式所需要的計算資源，請確保所有資源限制在單一個人桌上型電腦且能夠以總市價低於 5 萬的手段取得。

|計算資源|答案|
|-|-|
|使用 CPU 核心數|4|
|記憶體需求|$\leq 16$GB|
|有無使用 GPU|無|
|GPU 版本|無|
|GPU 記憶體需求|$\leq 2$GB|  

## 參考連結

- 有無參考他人之實驗結果：有
- 參考連結:  
Titanic Survival Predictions (Beginner): 
https://www.kaggle.com/code/nadintamer/titanic-survival-predictions-beginner  
Titanic [EDA] + Model Pipeline + Keras NN: 
https://www.kaggle.com/code/kabure/titanic-eda-model-pipeline-keras-nn  

步驟:
1. Import Necessary Libraries
2. Read In and Explore the Data
3. Data Analysis and Data Visualization
4. Data fixing
5. Cleaning Data
6. Create Model and optimize Model  

(參考:Titanic Survival Predictions (Beginner))  

### 1. Import Necessary Libraries  
# 請勿更動此區塊程式碼
```python=
import time
import numpy as np
import pandas as pd

EXECUTION_START_TIME = time.time() # 計算執行時間

df = pd.read_csv('train.csv')      # 讀取資料，請勿更改路徑
```  
## 資料分析與前處理


### 初步分析
從train.info得知資料數、類別和資料型態  
從describe()的conunt得知'Age','Cabin','Embarked'有資料缺缺失。  
- 其中, 'Cabin'缺失資料過多,難以修復,打算將其放棄。  
- 而'Age'我認為年紀與生存機率有很重要關聯，因此想辦法將資料補完整,    
- 'Embarked'只缺失3個,可將C,Q,S三值隨意取兩值補入,不會對數據造成太大影響。
  
### 2. Read In and Explore the Data  
```python=
# 讀取資料
train = pd.read_csv('train.csv')
train.info()
train.describe(include="all")
```  
![](https://i.imgur.com/TurERNI.png)

### 3. Data Analysis and Data Visualization  
```python=
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px
sns.barplot(x= "Age",y= "Survived",data= train)
plt.show()
sns.barplot(x= "Fare",y= "Survived",data= train)
plt.show()
sns.barplot(x= "Sex",y= "Survived",data= train)
plt.show()
sns.barplot(x= "Pclass",y= "Survived",data= train)  
plt.show()
sns.barplot(x= "SibSp",y= "Survived",data= train)  
plt.show()
sns.barplot(x= "Parch",y= "Survived",data= train)  
plt.show()
sns.barplot(x= "Fare",y= "Survived",data= train)  
plt.show()
sns.barplot(x= "Embarked",y= "Survived",data= train)
plt.show()
```
![](https://i.imgur.com/WuHuQXR.png)
![](https://i.imgur.com/7Hevw9T.png)
![](https://i.imgur.com/XzQ5TpP.png)
![](https://i.imgur.com/rHX7bKv.png)
![](https://i.imgur.com/QJr3rMJ.png)
![](https://i.imgur.com/Xau0FGk.png)
![](https://i.imgur.com/XdKyZBr.png)
![](https://i.imgur.com/Yqyfpe4.png) 

視覺化後，可更直觀的感受資料，但其中"Age"、"Fare"的資料因x軸間隔過細，無法清楚觀看，因此將其x軸作區間分類，重新處裡  
先看看資料的分佈再決定要如何分隔  
### 乘客年齡與登船費的分布
```python=
px.histogram(train, x="Age", nbins=60)  
```
![](https://i.imgur.com/lsH0OIM.png)
```python=
px.histogram(train, x="Fare", nbins=60)  
```
![](https://i.imgur.com/pJzMbvx.png)
```python=
abins = [0,12,20,45,60,100]
alabels = ["baby/kid" , "teenager" , "adult" ,"middle" , "senior"]
train['AgeGroup'] = pd.cut(train["Age"], bins = abins, labels = alabels)
sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()
fbins = [-2,50,100,200,600]
flabels = ["$0-50" , "$51-100" , "$101-200" ,"$200-600" ]
train['FareGroup'] = pd.cut(train["Fare"], bins = fbins, labels = flabels)
sns.barplot(x="FareGroup", y="Survived", data=train)
plt.show()
train.describe(include="all")
```
![](https://i.imgur.com/BJpf95W.png)
![](https://i.imgur.com/hnFnBS1.png)
![](https://i.imgur.com/TVlIYkd.png)
### 4. Data fixing
須修復的資料類別有"Embarked","Age"  
#### "Embarked"，"S"佔絕大多數，將NULL以"S"補上，不會造成過大的更動影響  
```python=
px.histogram(train, x="Embarked") 
```
![](https://i.imgur.com/GOeq5BL.png)

```python=
train.fillna({"Embarked": "S"}, inplace = True)
train.info()
```
![](https://i.imgur.com/UwbdQw3.png)
#### "Age"缺失的資料為177個，數量有一定的占比，不能直接全以相同的平均值或眾數值補齊，須想辦法保持資料的平衡性  
```python=
px.histogram(train, x="Age", nbins=60)  
```
![](https://i.imgur.com/cO9h5JD.png)
最初想法:  
將各年齡層計算相對比例，再依比例補入各年齡層的中位數。缺點:此法可能忽略掉與其他資料間的相依性質，雖保持資料的平衡但可能打亂其中的關聯性。  
較佳處理方式:  
參考kaggle這兩篇notebooke: Titanic Survival Predictions (Beginner)、Titanic [EDA] + Model Pipeline + Keras NN  
以乘客的稱謂"title"為依具，並搭配"Pclass"及"Sex"來，推論乘客的年齡。在英文中，稱謂能在年紀上做出模糊的分隔，因此以此方法來修復資料  
title提取方式:https://blog.csdn.net/m0_46352099/article/details/107787199  
https://blog.csdn.net/Koala_Tree/article/details/78725881  

先從"Name"中提取出，乘客的稱謂"Title"  
```python!
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])
```
![](https://i.imgur.com/8d8gSiY.png)

將船員和貴族的細項歸成一類  
```python=
train['Title'] = train['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'officer')    
train['Title'] = train['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
```
![](https://i.imgur.com/PG8npDz.png)

將處理好得"Title"資料視覺化，經統計後，貴族及婦女生還機率上相對較高。  
```python=
a = sns.barplot(x= 'Title',y= "Survived",data= train)
plt.xticks(rotation = 45)
plt.show()

sns.countplot(x='Title', data=train,hue="Survived")
plt.show()
```
![](https://i.imgur.com/x5DuUQr.png)
![](https://i.imgur.com/gER7b3k.png)

 接著藉由上面分類好的"Title"、"Pclass"和"Sex"來綜合推測每個稱謂其年齡區間  
```python=
age_SPT = train.groupby(["Sex","Pclass","Title"])["Age"]

print(age_SPT.median())
```
![](https://i.imgur.com/av4olEZ.png)
```python=
train.loc[train.Age.isnull(), 'Age'] = train.groupby(['Sex','Pclass','Title']).Age.transform('median')
px.histogram(train, x="Age", nbins=60)
abins = [0,12,20,45,60,100]
alabels = ["baby/kid" , "teenager" , "adult" ,"middle" , "senior"]
train['AgeGroup'] = pd.cut(train["Age"], bins = abins, labels = alabels)
sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()
```
![](https://i.imgur.com/sQcciaq.png)

填充預測資料後，在區間18-20和26-28的人數增多許多，而其餘年齡層變化不大。由此可以小小反推，可能(male,Pclass 3,Mr)和(female,Pclass 3,Miss)這兩類人之中有很多人未記錄年紀  
### 5. Cleaning Data
須補齊的資料現在已經補完了，現在區清除之後不參與分析的資料的  
```python=
train = train.drop(['Cabin'], axis = 1)
train = train.drop(['Ticket'], axis = 1)
train = train.drop(['Name'], axis = 1)
train = train.drop(['Age'], axis = 1)
train = train.drop(['Fare'], axis = 1)
train.describe(include="all")

```  
![](https://i.imgur.com/mmUml0m.png)
```python=
train.head()
```
![](https://i.imgur.com/W6MPi3G.png)

**將分類改以數值方式表示**  
```python=
embarked_map = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_map)
sex_map = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_map)
faregroup_map = {"$0-50" : 1 , "$51-100" : 2 , "$101-200" : 3 ,"$200-600" : 4}
train['FareGroup'] = train['FareGroup'].map(faregroup_map)
agegroup_map = {"baby/kid" : 1, "teenager" :2 , "adult" : 3 ,"middle" : 4 , "senior" : 5}
train['AgeGroup'] = train['AgeGroup'].map(agegroup_map)
title_map = {'Mr' : 1,'Mrs' : 2,'Miss' : 3,'Master' : 4,'officer' :5,'Royal' :6}
train['Title'] = train['Title'].map(title_map)
train.head()
```
![](https://i.imgur.com/DgBl3Ir.png)

## 模型訓練  
random seed參考資料:  
随机种子Seed的讲人话解释:https://zhuanlan.zhihu.com/p/545344518  
sklearn函数：train_test_split:https://zhuanlan.zhihu.com/p/248634166  
### 6. Create Model and optimize Model  
**切分trainind data和valid data**  
```python=
from sklearn.model_selection import train_test_split

train_x = train.drop(['Survived', 'PassengerId'], axis=1)
train_y = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(train_x, train_y, test_size = 0.22, random_state = 0)
```
**數據套入模型運算**  
## DecisionTree
DecisionTree決策數，是一種分類演算法，其分類方式可以用倒著的樹形結構來表達。
在樹中的每個節點代表著一個feature特徵，而根據特徵的差異，再向下分類，最終會到達leaf葉子處，為最終的分類結果。  
參考資料:  
https://www.gushiciku.cn/pl/gcfJ/zh-tw  
https://www.kaggle.com/code/nadintamer/titanic-survival-predictions-beginner  
https://blog.csdn.net/qq236237606/article/details/105106247  
```python=
# 模型訓練-DecisionTree

from sklearn.model_selection import KFold             # 匯入 K 次交叉驗證工具
from sklearn.tree import DecisionTreeClassifier       # 匯入決策樹模型
from sklearn.metrics import accuracy_score            # 匯入準確度計算工具

kf = KFold(n_splits=5,                                # 設定 K=5 值
           random_state=1012,
           shuffle=True)
kf.get_n_splits(train_x)                              # 給予資料範圍

train_acc_list = []                                   # 儲存每次訓練模型的準確度
valid_acc_list = []                                   # 儲存每次驗證模型的準確度

for train_index, valid_index in kf.split(train_x):    # 每個迴圈都會產生不同部份的資料
    train_x_split = train_x.iloc[train_index]         # 產生訓練資料
    train_y_split = train_y.iloc[train_index]         # 產生訓練資料標籤
    valid_x_split = train_x.iloc[valid_index]         # 產生驗證資料
    valid_y_split = train_y.iloc[valid_index]         # 產生驗證資料標籤
    
    DC1 = DecisionTreeClassifier(random_state=1012) # 創造決策樹模型
    DC1.fit(train_x_split, train_y_split)           # 訓練決策樹模型
    
    train_pred_y = DC1.predict(train_x_split)       # 確認模型是否訓練成功
    train_acc = accuracy_score(train_y_split,         # 計算訓練資料準確度
                               train_pred_y)
    valid_pred_y = DC1.predict(valid_x_split)       # 驗證模型是否訓練成功
    valid_acc = accuracy_score(valid_y_split,         # 計算驗證資料準確度
                               valid_pred_y)
    
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

print((
    'average train accuracy: {}\n' +
    '    min train accuracy: {}\n' +
    '    max train accuracy: {}\n' +
    'average valid accuracy: {}\n' +
    '    min valid accuracy: {}\n' +
    '    max valid accuracy: {}').format(
    np.mean(train_acc_list),                          # 輸出平均訓練準確度
    np.min(train_acc_list),                           # 輸出最低訓練準確度
    np.max(train_acc_list),                           # 輸出最高訓練準確度
    np.mean(valid_acc_list),                          # 輸出平均驗證準確度
    np.min(valid_acc_list),                           # 輸出最低驗證準確度
    np.max(valid_acc_list)                            # 輸出最高驗證準確度
))

DC1.fit(x_train, y_train)
y_pred = DC1.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print("without KFlod valid accuracy:",acc_decisiontree)
```
![](https://i.imgur.com/1CEGImC.png)

```python=
# 模型訓練-DecisionTree調參後

from sklearn.model_selection import KFold             # 匯入 K 次交叉驗證工具
from sklearn.tree import DecisionTreeClassifier       # 匯入決策樹模型
from sklearn.metrics import accuracy_score            # 匯入準確度計算工具

kf = KFold(n_splits=5,                                # 設定 K=5 值
           random_state=1012,
           shuffle=True)
kf.get_n_splits(train_x)                              # 給予資料範圍

train_acc_list = []                                   # 儲存每次訓練模型的準確度
valid_acc_list = []                                   # 儲存每次驗證模型的準確度

for train_index, valid_index in kf.split(train_x):    # 每個迴圈都會產生不同部份的資料
    train_x_split = train_x.iloc[train_index]         # 產生訓練資料
    train_y_split = train_y.iloc[train_index]         # 產生訓練資料標籤
    valid_x_split = train_x.iloc[valid_index]         # 產生驗證資料
    valid_y_split = train_y.iloc[valid_index]         # 產生驗證資料標籤
    
    DC2 = DecisionTreeClassifier(splitter= 'best',
                                   min_weight_fraction_leaf= 0,
                                   min_samples_split= 5,
                                   min_samples_leaf= 1,
                                   max_depth= 109,
                                   criterion='gini',
                                   random_state=1012) # 創造決策樹模型
    DC2.fit(train_x_split, train_y_split)           # 訓練決策樹模型
    
    train_pred_y = DC2.predict(train_x_split)       # 確認模型是否訓練成功
    train_acc = accuracy_score(train_y_split,         # 計算訓練資料準確度
                               train_pred_y)
    valid_pred_y = DC2.predict(valid_x_split)       # 驗證模型是否訓練成功
    valid_acc = accuracy_score(valid_y_split,         # 計算驗證資料準確度
                               valid_pred_y)
    
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

print((
    'average train accuracy: {}\n' +
    '    min train accuracy: {}\n' +
    '    max train accuracy: {}\n' +
    'average valid accuracy: {}\n' +
    '    min valid accuracy: {}\n' +
    '    max valid accuracy: {}').format(
    np.mean(train_acc_list),                          # 輸出平均訓練準確度
    np.min(train_acc_list),                           # 輸出最低訓練準確度
    np.max(train_acc_list),                           # 輸出最高訓練準確度
    np.mean(valid_acc_list),                          # 輸出平均驗證準確度
    np.min(valid_acc_list),                           # 輸出最低驗證準確度
    np.max(valid_acc_list)                            # 輸出最高驗證準確度
))

DC2.fit(x_train, y_train)
y_pred = DC2.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print("without KFlod valid accuracy:",acc_decisiontree)
```
![](https://i.imgur.com/tPZvwFm.png)
調整參數前，訓練與驗證準確度差距10%(0.88-0.78)，已經有出現overfitting的現象。  
經調參後，降為8%(0.87-0.79)，驗證準確度提升1%  
```python=
#利用RandomizedSearchCV找尋最佳超參數-DecisionTree
from sklearn.model_selection import RandomizedSearchCV

#建立參數的各自區間
criterion=['gini']
splitter= ['best']
max_depth= [int(x) for x in np.linspace(10, 1000, num=11)]
min_samples_split= [2, 5, 10]
min_samples_leaf= [1, 2, 4]
min_weight_fraction_leaf= [0 ,0.2,0.5]

random_grid = {'criterion': criterion, 'splitter': splitter, 'max_depth': max_depth,
              'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
              'min_weight_fraction_leaf':min_weight_fraction_leaf}

model =  DecisionTreeClassifier(random_state=1012)
dt_random = RandomizedSearchCV(estimator = model, param_distributions=random_grid,
                              n_iter=10, cv=5,  random_state=1012, error_score='raise')

dt_random.fit(train_x, train_y)
dt_random.best_params_
```
![](https://i.imgur.com/d4pCfyf.png)

## AdaBoost
Adaboost是一種ensemble learing，ensemble learing是將好幾個監督是學習模型組合在一起，產生更強大的模型。
ensemble learing有三類，分別為Bagging, Boosting, Stacking，而Adaboost為Boosting的一種。  
AdaBoost由多個分類器串起來，前一個分類器分錯的樣本會加重其權重後，用來訓練下一個分類器。這樣能加強模型在較難分類的樣本上的分類能力。  
參考資料:  
https://ithelp.ithome.com.tw/articles/10247936  
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html  
https://zhuanlan.zhihu.com/p/41536315  
https://www.gushiciku.cn/pl/gcfJ/zh-tw  
https://blog.csdn.net/JohnsonSmile/article/details/88759761  
```python=
# 模型訓練-AdaBoost

from sklearn.model_selection import KFold             # 匯入 K 次交叉驗證工具
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier   # 匯入決策樹模型
from sklearn.metrics import accuracy_score            # 匯入準確度計算工具

kf = KFold(n_splits=5,                                # 設定 K=5 值
           random_state=1012,
           shuffle=True)
kf.get_n_splits(train_x)                              # 給予資料範圍

train_acc_list = []                                   # 儲存每次訓練模型的準確度
valid_acc_list = []                                   # 儲存每次驗證模型的準確度

for train_index, valid_index in kf.split(train_x):    # 每個迴圈都會產生不同部份的資料
    train_x_split = train_x.iloc[train_index]         # 產生訓練資料
    train_y_split = train_y.iloc[train_index]         # 產生訓練資料標籤
    valid_x_split = train_x.iloc[valid_index]         # 產生驗證資料
    valid_y_split = train_y.iloc[valid_index]         # 產生驗證資料標籤
    
    AB1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=1012)
    AB1.fit(train_x_split, train_y_split)
    
    train_pred_y = AB1.predict(train_x_split)       # 確認模型是否訓練成功
    train_acc = accuracy_score(train_y_split,         # 計算訓練資料準確度
                               train_pred_y)
    valid_pred_y = AB1.predict(valid_x_split)       # 驗證模型是否訓練成功
    valid_acc = accuracy_score(valid_y_split,         # 計算驗證資料準確度
                               valid_pred_y)
    
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

print((
    'average train accuracy: {}\n' +
    '    min train accuracy: {}\n' +
    '    max train accuracy: {}\n' +
    'average valid accuracy: {}\n' +
    '    min valid accuracy: {}\n' +
    '    max valid accuracy: {}').format(
    np.mean(train_acc_list),                          # 輸出平均訓練準確度
    np.min(train_acc_list),                           # 輸出最低訓練準確度
    np.max(train_acc_list),                           # 輸出最高訓練準確度
    np.mean(valid_acc_list),                          # 輸出平均驗證準確度
    np.min(valid_acc_list),                           # 輸出最低驗證準確度
    np.max(valid_acc_list)                            # 輸出最高驗證準確度
))
AB1.fit(x_train, y_train)
y_pred = AB1.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print("without KFlod valid accuracy:",acc_decisiontree)
```
![](https://i.imgur.com/d8n3UCQ.png)
```python=
# 模型訓練-AdaBoost調參後

from sklearn.model_selection import KFold             # 匯入 K 次交叉驗證工具
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier   # 匯入決策樹模型
from sklearn.metrics import accuracy_score            # 匯入準確度計算工具

kf = KFold(n_splits=5,                                # 設定 K=5 值
           random_state=1012,
           shuffle=True)
kf.get_n_splits(train_x)                              # 給予資料範圍

train_acc_list = []                                   # 儲存每次訓練模型的準確度
valid_acc_list = []                                   # 儲存每次驗證模型的準確度

for train_index, valid_index in kf.split(train_x):    # 每個迴圈都會產生不同部份的資料
    train_x_split = train_x.iloc[train_index]         # 產生訓練資料
    train_y_split = train_y.iloc[train_index]         # 產生訓練資料標籤
    valid_x_split = train_x.iloc[valid_index]         # 產生驗證資料
    valid_y_split = train_y.iloc[valid_index]         # 產生驗證資料標籤
    
    AB2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                               n_estimators= 307, learning_rate= 0.1, algorithm= 'SAMME.R', random_state=1012)
    AB2.fit(train_x_split, train_y_split)
    
    train_pred_y = AB2.predict(train_x_split)       # 確認模型是否訓練成功
    train_acc = accuracy_score(train_y_split,         # 計算訓練資料準確度
                               train_pred_y)
    valid_pred_y = AB2.predict(valid_x_split)       # 驗證模型是否訓練成功
    valid_acc = accuracy_score(valid_y_split,         # 計算驗證資料準確度
                               valid_pred_y)
    
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

print((
    'average train accuracy: {}\n' +
    '    min train accuracy: {}\n' +
    '    max train accuracy: {}\n' +
    'average valid accuracy: {}\n' +
    '    min valid accuracy: {}\n' +
    '    max valid accuracy: {}').format(
    np.mean(train_acc_list),                          # 輸出平均訓練準確度
    np.min(train_acc_list),                           # 輸出最低訓練準確度
    np.max(train_acc_list),                           # 輸出最高訓練準確度
    np.mean(valid_acc_list),                          # 輸出平均驗證準確度
    np.min(valid_acc_list),                           # 輸出最低驗證準確度
    np.max(valid_acc_list)                            # 輸出最高驗證準確度
))
AB2.fit(x_train, y_train)
y_pred = AB2.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print("without KFlod valid accuracy:",acc_decisiontree)
```
![](https://i.imgur.com/jDovBQM.png)
調整參數前，訓練與驗證準確度差距2%(0.82-0.80)。  
經調參後，降為1%(0.82-0.81)，驗證準確度提升1%  
```python=
#利用RandomizedSearchCV找尋最佳超參數-AdaBoost

#建立參數的各自區間
n_estimators= [int(x) for x in np.linspace(10, 1000, num=11)]
learning_rate= [0.1,0.5,1]
algorithm=['SAMME.R']


random_grid = {'n_estimators': n_estimators, 'learning_rate':learning_rate, 'algorithm':algorithm}

model = AdaBoostClassifier(random_state=1012)
ad_random = RandomizedSearchCV(estimator = model, param_distributions=random_grid,
                              n_iter=10, cv=5, random_state=1012)

ad_random.fit(train_x,train_y)
ad_random.best_params_
```
![](https://i.imgur.com/9XV0GaB.png)

## RandomForest
RandomForest屬於ensemble learing中的Bagging。  
RandomForest的運作方法是建立多棵decision tree，每棵decision tree都對訓練資料重新採樣，最後把每棵樹的結果蒐集起來，進行投票來得出預測最終的結果。  
參考資料:  
https://ithelp.ithome.com.tw/articles/10247936  
https://www.gushiciku.cn/pl/gcfJ/zh-tw  
https://ithelp.ithome.com.tw/m/articles/10267379?sc=iThomeR  
```python=
# 模型訓練-RandomForest

from sklearn.ensemble import RandomForestClassifier


kf = KFold(n_splits=5,                                # 設定 K=5 值
           random_state=1012,
           shuffle=True)
kf.get_n_splits(train_x)                              # 給予資料範圍

train_acc_list = []                                   # 儲存每次訓練模型的準確度
valid_acc_list = []                                   # 儲存每次驗證模型的準確度

for train_index, valid_index in kf.split(train_x):    # 每個迴圈都會產生不同部份的資料
    train_x_split = train_x.iloc[train_index]         # 產生訓練資料
    train_y_split = train_y.iloc[train_index]         # 產生訓練資料標籤
    valid_x_split = train_x.iloc[valid_index]         # 產生驗證資料
    valid_y_split = train_y.iloc[valid_index]         # 產生驗證資料標籤
    

    RF1 = RandomForestClassifier(n_estimators= 200,
                                   min_samples_split= 5,
                                   min_samples_leaf= 4,
                                   max_features= 'auto',
                                   max_depth= 10,
                                   bootstrap= True,
                                   random_state=1012) # 創造決策樹模型
    RF1.fit(train_x_split, train_y_split)           # 訓練決策樹模型
    
    train_pred_y = RF1.predict(train_x_split)       # 確認模型是否訓練成功
    train_acc = accuracy_score(train_y_split,         # 計算訓練資料準確度
                               train_pred_y)
    valid_pred_y = RF1.predict(valid_x_split)       # 驗證模型是否訓練成功
    valid_acc = accuracy_score(valid_y_split,         # 計算驗證資料準確度
                               valid_pred_y)
    
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

print((
    'average train accuracy: {}\n' +
    '    min train accuracy: {}\n' +
    '    max train accuracy: {}\n' +
    'average valid accuracy: {}\n' +
    '    min valid accuracy: {}\n' +
    '    max valid accuracy: {}').format(
    np.mean(train_acc_list),                          # 輸出平均訓練準確度
    np.min(train_acc_list),                           # 輸出最低訓練準確度
    np.max(train_acc_list),                           # 輸出最高訓練準確度
    np.mean(valid_acc_list),                          # 輸出平均驗證準確度
    np.min(valid_acc_list),                           # 輸出最低驗證準確度
    np.max(valid_acc_list)                            # 輸出最高驗證準確度
))
RF1.fit(x_train, y_train)
y_pred = RF1.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print("without KFlod valid accuracy:",acc_decisiontree)
```
![](https://i.imgur.com/QqWRr8o.png)
```python=
# 模型訓練-RandomForest調參後

from sklearn.ensemble import RandomForestClassifier


kf = KFold(n_splits=5,                                # 設定 K=5 值
           random_state=1012,
           shuffle=True)
kf.get_n_splits(train_x)                              # 給予資料範圍

train_acc_list = []                                   # 儲存每次訓練模型的準確度
valid_acc_list = []                                   # 儲存每次驗證模型的準確度

for train_index, valid_index in kf.split(train_x):    # 每個迴圈都會產生不同部份的資料
    train_x_split = train_x.iloc[train_index]         # 產生訓練資料
    train_y_split = train_y.iloc[train_index]         # 產生訓練資料標籤
    valid_x_split = train_x.iloc[valid_index]         # 產生驗證資料
    valid_y_split = train_y.iloc[valid_index]         # 產生驗證資料標籤
    

    RF2 = RandomForestClassifier(n_estimators= 452,
                                   min_samples_split= 10,
                                   min_samples_leaf= 4,
                                   max_features= 'auto',
                                   max_depth= 406,
                                   bootstrap= True,
                                   random_state=1012) # 創造決策樹模型
    RF2.fit(train_x_split, train_y_split)           # 訓練決策樹模型
    
    train_pred_y = RF2.predict(train_x_split)       # 確認模型是否訓練成功
    train_acc = accuracy_score(train_y_split,         # 計算訓練資料準確度
                               train_pred_y)
    valid_pred_y = RF2.predict(valid_x_split)       # 驗證模型是否訓練成功
    valid_acc = accuracy_score(valid_y_split,         # 計算驗證資料準確度
                               valid_pred_y)
    
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

print((
    'average train accuracy: {}\n' +
    '    min train accuracy: {}\n' +
    '    max train accuracy: {}\n' +
    'average valid accuracy: {}\n' +
    '    min valid accuracy: {}\n' +
    '    max valid accuracy: {}').format(
    np.mean(train_acc_list),                          # 輸出平均訓練準確度
    np.min(train_acc_list),                           # 輸出最低訓練準確度
    np.max(train_acc_list),                           # 輸出最高訓練準確度
    np.mean(valid_acc_list),                          # 輸出平均驗證準確度
    np.min(valid_acc_list),                           # 輸出最低驗證準確度
    np.max(valid_acc_list)                            # 輸出最高驗證準確度
))
RF2.fit(x_train, y_train)
y_pred = RF2.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print("without KFlod valid accuracy:",acc_decisiontree)
```
![](https://i.imgur.com/WQS1oU5.png)
調整參數前，訓練與驗證準確度差距2%(0.84-0.82)。  
經調參後，保持差距2%(0.84-0.82)，驗證準確度無提升  
```python=
#利用RandomizedSearchCV找尋最佳超參數-RandomForest
from sklearn.model_selection import RandomizedSearchCV

#建立參數的各自區間
n_estimators = [int(x) for x in np.linspace(start=10, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 1000, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

model = RandomForestClassifier(random_state=1012)
rf_random = RandomizedSearchCV(estimator = model, param_distributions=random_grid,
                              n_iter=10, cv=5, random_state=1012)

rf_random.fit(train_x,train_y)
rf_random.best_params_
```
![](https://i.imgur.com/ssFZtpO.png)

## GradientBoosting
GradientBoosting與AdaBoost一樣屬於ensemble learing中的Boosting類。  
同樣結構也是由多個分類器串起來，但核心概念不同，其最終結果為所有分類器的推測結論累加而成，且每一個分類器學習的是前一個分類器的預測誤差。過程中，期望能漸漸縮小預測誤差，當最後一個分類器的預測誤差為零時，最終結果便會與真實資料相符合。  
參考資料:  
https://zhuanlan.zhihu.com/p/108641227  
https://blog.csdn.net/VariableX/article/details/107200334  
```python=
# 模型訓練-GradientBoosting

from sklearn.ensemble import GradientBoostingClassifier       # 匯入決策樹模型


kf = KFold(n_splits=5,                                # 設定 K=5 值
           random_state=1012,
           shuffle=True)
kf.get_n_splits(train_x)                              # 給予資料範圍

train_acc_list = []                                   # 儲存每次訓練模型的準確度
valid_acc_list = []                                   # 儲存每次驗證模型的準確度

for train_index, valid_index in kf.split(train_x):    # 每個迴圈都會產生不同部份的資料
    train_x_split = train_x.iloc[train_index]         # 產生訓練資料
    train_y_split = train_y.iloc[train_index]         # 產生訓練資料標籤
    valid_x_split = train_x.iloc[valid_index]         # 產生驗證資料
    valid_y_split = train_y.iloc[valid_index]         # 產生驗證資料標籤
    
    GB1 = GradientBoostingClassifier(random_state=1012) # 創造決策樹模型
    GB1.fit(train_x_split, train_y_split)           # 訓練決策樹模型
    
    train_pred_y = GB1.predict(train_x_split)       # 確認模型是否訓練成功
    train_acc = accuracy_score(train_y_split,         # 計算訓練資料準確度
                               train_pred_y)
    valid_pred_y = GB1.predict(valid_x_split)       # 驗證模型是否訓練成功
    valid_acc = accuracy_score(valid_y_split,         # 計算驗證資料準確度
                               valid_pred_y)
    
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

print((
    'average train accuracy: {}\n' +
    '    min train accuracy: {}\n' +
    '    max train accuracy: {}\n' +
    'average valid accuracy: {}\n' +
    '    min valid accuracy: {}\n' +
    '    max valid accuracy: {}').format(
    np.mean(train_acc_list),                          # 輸出平均訓練準確度
    np.min(train_acc_list),                           # 輸出最低訓練準確度
    np.max(train_acc_list),                           # 輸出最高訓練準確度
    np.mean(valid_acc_list),                          # 輸出平均驗證準確度
    np.min(valid_acc_list),                           # 輸出最低驗證準確度
    np.max(valid_acc_list)                            # 輸出最高驗證準確度
))
GB1.fit(x_train, y_train)
y_pred = GB1.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print("without KFlod valid accuracy:",acc_decisiontree)
```
![](https://i.imgur.com/JCI3dtZ.png)
```python=
# 模型訓練-GradientBoosting調參後

from sklearn.ensemble import GradientBoostingClassifier       # 匯入決策樹模型


kf = KFold(n_splits=5,                                # 設定 K=5 值
           random_state=1012,
           shuffle=True)
kf.get_n_splits(train_x)                              # 給予資料範圍

train_acc_list = []                                   # 儲存每次訓練模型的準確度
valid_acc_list = []                                   # 儲存每次驗證模型的準確度

for train_index, valid_index in kf.split(train_x):    # 每個迴圈都會產生不同部份的資料
    train_x_split = train_x.iloc[train_index]         # 產生訓練資料
    train_y_split = train_y.iloc[train_index]         # 產生訓練資料標籤
    valid_x_split = train_x.iloc[valid_index]         # 產生驗證資料
    valid_y_split = train_y.iloc[valid_index]         # 產生驗證資料標籤
    
    GB2 = GradientBoostingClassifier(subsample= 1,
                                       n_estimators= 1336,
                                       min_samples_split= 5,
                                       min_samples_leaf= 4,
                                       max_features= 'sqrt',
                                       max_depth= 901,
                                       learning_rate= 0.001,
                                       random_state=1012) # 創造決策樹模型
    GB2.fit(train_x_split, train_y_split)           # 訓練決策樹模型
    
    train_pred_y = GB2.predict(train_x_split)       # 確認模型是否訓練成功
    train_acc = accuracy_score(train_y_split,         # 計算訓練資料準確度
                               train_pred_y)
    valid_pred_y = GB2.predict(valid_x_split)       # 驗證模型是否訓練成功
    valid_acc = accuracy_score(valid_y_split,         # 計算驗證資料準確度
                               valid_pred_y)
    
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

print((
    'average train accuracy: {}\n' +
    '    min train accuracy: {}\n' +
    '    max train accuracy: {}\n' +
    'average valid accuracy: {}\n' +
    '    min valid accuracy: {}\n' +
    '    max valid accuracy: {}').format(
    np.mean(train_acc_list),                          # 輸出平均訓練準確度
    np.min(train_acc_list),                           # 輸出最低訓練準確度
    np.max(train_acc_list),                           # 輸出最高訓練準確度
    np.mean(valid_acc_list),                          # 輸出平均驗證準確度
    np.min(valid_acc_list),                           # 輸出最低驗證準確度
    np.max(valid_acc_list)                            # 輸出最高驗證準確度
))
GB2.fit(x_train, y_train)
y_pred = GB2.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print("without KFlod valid accuracy:",acc_decisiontree)
```
![](https://i.imgur.com/qfHfauq.png)
調整參數前，訓練與驗證準確度差距5%(0.86-0.81)。  
經調參後，訓練與驗證準確度差距4%(0.85-0.81)，驗證準確度無提升  
```python=
#利用RandomizedSearchCV找尋最佳超參數-GradientBoosting
from sklearn.model_selection import RandomizedSearchCV

#建立參數的各自區間
learning_rate=[0.001, 0.01, 0.1, 1]
n_estimators=[int(x) for x in np.linspace(start=10, stop=2000, num=10)]
subsample=[0.5, 0.6, 0.7, 0.8, 0.9, 1]
min_samples_split=[2,5,10]
min_samples_leaf=[1,2,4]
max_depth=[int(x) for x in np.linspace(10, 1000, num=11)]
max_features=['log2', 'sqrt', None]

random_grid = {'learning_rate':learning_rate, 'n_estimators':n_estimators,
               'subsample':subsample,'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf,
               'max_depth':max_depth, 'max_features':max_features}

model = GradientBoostingClassifier(random_state=1012)
gb_random = RandomizedSearchCV(estimator = model, param_distributions=random_grid,
                              n_iter=10, cv=5, random_state=1012)

gb_random.fit(train_x,train_y)
gb_random.best_params_
```
![](https://i.imgur.com/v8IYKJd.png)
## SVM
SVM支援向量機，一種二分類模型，其理念是找出一個函數(超平面)能與類別之間距離最大化。  
可以想成小朋友睡午覺時，中間不能跨越的那條楚河漢界，將你我之間地盤最大化的概念。  
參考資料:    
https://zhuanlan.zhihu.com/p/77750026  
https://www.kaggle.com/code/nadintamer/titanic-survival-predictions-beginner  
https://www.zhihu.com/tardis/zm/art/31886934?source_id=1005  
```python=
# 模型訓練-svm

from sklearn.svm import SVC      # 匯入決策樹模型


kf = KFold(n_splits=5,                                # 設定 K=5 值
           random_state=1012,
           shuffle=True)
kf.get_n_splits(train_x)                              # 給予資料範圍

train_acc_list = []                                   # 儲存每次訓練模型的準確度
valid_acc_list = []                                   # 儲存每次驗證模型的準確度

for train_index, valid_index in kf.split(train_x):    # 每個迴圈都會產生不同部份的資料
    train_x_split = train_x.iloc[train_index]         # 產生訓練資料
    train_y_split = train_y.iloc[train_index]         # 產生訓練資料標籤
    valid_x_split = train_x.iloc[valid_index]         # 產生驗證資料
    valid_y_split = train_y.iloc[valid_index]         # 產生驗證資料標籤
    
    SVM1 = SVC(random_state=1012) # 創造決策樹模型
    SVM1.fit(train_x_split, train_y_split)           # 訓練決策樹模型
    
    train_pred_y = SVM1.predict(train_x_split)       # 確認模型是否訓練成功
    train_acc = accuracy_score(train_y_split,         # 計算訓練資料準確度
                               train_pred_y)
    valid_pred_y = SVM1.predict(valid_x_split)       # 驗證模型是否訓練成功
    valid_acc = accuracy_score(valid_y_split,         # 計算驗證資料準確度
                               valid_pred_y)
    
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

print((
    'average train accuracy: {}\n' +
    '    min train accuracy: {}\n' +
    '    max train accuracy: {}\n' +
    'average valid accuracy: {}\n' +
    '    min valid accuracy: {}\n' +
    '    max valid accuracy: {}').format(
    np.mean(train_acc_list),                          # 輸出平均訓練準確度
    np.min(train_acc_list),                           # 輸出最低訓練準確度
    np.max(train_acc_list),                           # 輸出最高訓練準確度
    np.mean(valid_acc_list),                          # 輸出平均驗證準確度
    np.min(valid_acc_list),                           # 輸出最低驗證準確度
    np.max(valid_acc_list)                            # 輸出最高驗證準確度
))
SVM1.fit(x_train, y_train)
y_pred = SVM1.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print("without KFlod valid accuracy:",acc_decisiontree)
```
![](https://i.imgur.com/iI6HUDz.png)
```python=
# 模型訓練-svm調參後

from sklearn.svm import SVC      # 匯入決策樹模型


kf = KFold(n_splits=5,                                # 設定 K=5 值
           random_state=1012,
           shuffle=True)
kf.get_n_splits(train_x)                              # 給予資料範圍

train_acc_list = []                                   # 儲存每次訓練模型的準確度
valid_acc_list = []                                   # 儲存每次驗證模型的準確度

for train_index, valid_index in kf.split(train_x):    # 每個迴圈都會產生不同部份的資料
    train_x_split = train_x.iloc[train_index]         # 產生訓練資料
    train_y_split = train_y.iloc[train_index]         # 產生訓練資料標籤
    valid_x_split = train_x.iloc[valid_index]         # 產生驗證資料
    valid_y_split = train_y.iloc[valid_index]         # 產生驗證資料標籤
    
    SVM2 = SVC(kernel= 'rbf',
                gamma= 'scale',
                degree= 3,
                C= 1.0,
                random_state=1012) # 創造決策樹模型
    SVM2.fit(train_x_split, train_y_split)           # 訓練決策樹模型
    
    train_pred_y = SVM2.predict(train_x_split)       # 確認模型是否訓練成功
    train_acc = accuracy_score(train_y_split,         # 計算訓練資料準確度
                               train_pred_y)
    valid_pred_y = SVM2.predict(valid_x_split)       # 驗證模型是否訓練成功
    valid_acc = accuracy_score(valid_y_split,         # 計算驗證資料準確度
                               valid_pred_y)
    
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

print((
    'average train accuracy: {}\n' +
    '    min train accuracy: {}\n' +
    '    max train accuracy: {}\n' +
    'average valid accuracy: {}\n' +
    '    min valid accuracy: {}\n' +
    '    max valid accuracy: {}').format(
    np.mean(train_acc_list),                          # 輸出平均訓練準確度
    np.min(train_acc_list),                           # 輸出最低訓練準確度
    np.max(train_acc_list),                           # 輸出最高訓練準確度
    np.mean(valid_acc_list),                          # 輸出平均驗證準確度
    np.min(valid_acc_list),                           # 輸出最低驗證準確度
    np.max(valid_acc_list)                            # 輸出最高驗證準確度
))
SVM2.fit(x_train, y_train)
y_pred = SVM2.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print("without KFlod valid accuracy:",acc_decisiontree)
```
![](https://i.imgur.com/BxOoDyE.png)
調整參數前，訓練與驗證準確度差距1%(0.83-0.82)。  
經調參後，訓練與驗證準確度差距1%(0.83-0.82)，驗證準確度無提升  
```python=
#利用RandomizedSearchCV找尋最佳超參數-svm
from sklearn.model_selection import RandomizedSearchCV

#建立參數的各自區間
C=[0.01,0.5,1.0,1.5,2]
kernel=['rbf','linear','poly']
degree=[3,5,10]
gamma=['scale','auto']

random_grid = {'C':C,'kernel':kernel, 'degree': degree, 'gamma':gamma }

model = SVC(random_state=1012)
svm_random = RandomizedSearchCV(estimator = model, param_distributions=random_grid,
                              n_iter=10, cv=5, random_state=1012)

svm_random.fit(train_x,train_y)
svm_random.best_params_
```
![](https://i.imgur.com/ZKfThEO.png)
### Create Model and optimize Model總結:
上述模型中RandomForest, GradientBoosting, SVM的驗證準確度能有82%的準卻度，而DecisionTree和AdaBoost的驗證準確度較差(分別為0.79和0.81)  
再調整過模型的超參數之後，訓練與驗證準確度差距會有些許的下降，而驗證準確度無明顯。因此如想再將準確度提升，就需要再嘗試其他模型或者從訓練數據去進行調整。  

# test資料預處理  
```python=
test = pd.read_csv("test.csv")

#take a look at the testing data
test.info()
test.describe(include="all")
```

![](https://i.imgur.com/eI6Ztqd.png)
![](https://i.imgur.com/vlF4aNy.png)
```python=
abins = [0,12,20,45,60,100]
alabels = ["baby/kid" , "teenager" , "adult" ,"middle" , "senior"]
test['AgeGroup'] = pd.cut(test["Age"], bins = abins, labels = alabels)

test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(test['Title'], test['Sex'])

test['Title'] = test['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'officer')    
test['Title'] = test['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace('Ms', 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')


age_SPT = test.groupby(["Sex","Pclass","Title"])["Age"]
print(age_SPT.median())

test.loc[test.Age.isnull(), 'Age'] = test.groupby(['Sex','Pclass','Title']).Age.transform('median')
px.histogram(test, x="Age", nbins=60)
abins = [0,12,20,45,60,100]
alabels = ["baby/kid" , "teenager" , "adult" ,"middle" , "senior"]
test['AgeGroup'] = pd.cut(test["Age"], bins = abins, labels = alabels)

test.loc[test.Fare.isnull(), 'Fare'] = test.groupby(['Sex','Pclass','Title']).Fare.transform('median')
fbins = [-2,50,100,200,600]
flabels = ["$0-50" , "$51-100" , "$101-200" ,"$200-600" ]
test['FareGroup'] = pd.cut(test["Fare"], bins = fbins, labels = flabels)

test = test.drop(['Cabin'], axis = 1)
test = test.drop(['Ticket'], axis = 1)
test = test.drop(['Name'], axis = 1)
test = test.drop(['Age'], axis = 1)
test = test.drop(['Fare'], axis = 1)

order = ['PassengerId', 'Pclass','Sex', 'SibSp', 'Parch', 'Embarked', 'AgeGroup', 'FareGroup', 'Title']
test = test[order]

embarked_map = {"S": 1, "C": 2, "Q": 3}
test['Embarked'] = test['Embarked'].map(embarked_map)
sex_map = {"male": 0, "female": 1}
test['Sex'] = test['Sex'].map(sex_map)
faregroup_map = {"$0-50" : 1 , "$51-100" : 2 , "$101-200" : 3 ,"$200-600" : 4}
test['FareGroup'] = test['FareGroup'].map(faregroup_map)
agegroup_map = {"baby/kid" : 1, "teenager" :2 , "adult" : 3 ,"middle" : 4 , "senior" : 5}
test['AgeGroup'] = test['AgeGroup'].map(agegroup_map)
title_map = {'Mr' : 1,'Mrs' : 2,'Miss' : 3,'Master' : 4,'officer' :5,'Royal' :6}
test['Title'] = test['Title'].map(title_map)
test.info()
test.head()
```
![](https://i.imgur.com/tcZrCXI.png)
![](https://i.imgur.com/rrTLZoo.png)
**Creating Submission File**
參考資料:  
https://www.kaggle.com/code/nadintamer/titanic-survival-predictions-beginner  
```python=
#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = SVM2.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission_SVM2.csv', index=False)
```

![](https://i.imgur.com/wJ9n4Bi.png)
