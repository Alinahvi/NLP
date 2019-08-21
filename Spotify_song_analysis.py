# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import misc
#from sklearn import tree
#from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
%matplotlib inline
from io import StringIO
from io import StringIO
import io
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus



# Importing and vieing the dataset
dataset = pd.read_csv('data.csv')
#dataset.head()
#dataset.info()



#Designing test set and training set
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size = 0.15)

features = ['danceability', 'loudness', 'valence', 'energy', 'instrumentalness','acousticness','key','speechiness', 'duration_ms']
X_train = train[features]
y_train = train['target']

X_test = test[features]
y_test = test['target']



#Desion Tree

# classifier for minimum n sample in each leaf
from sklearn.tree import DecisionTreeClassifier, export_graphviz
classifier = DecisionTreeClassifier(min_samples_split = 50)
dt = classifier.fit(X_train, y_train)

# classifier for minimum entropy
from sklearn.tree import DecisionTreeClassifier, export_graphviz
classifier2= DecisionTreeClassifier(criterion = 'entropy')
dt = classifier2.fit(X_train, y_train)

#uding random forest
from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier3.fit(X_train, y_train)


#tree visualiztion
def show_tree(tree, features, path):
    f = io.StringIO()
    export_graphviz(tree, out_file=f, feature_names = features)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img = misc.imread(path)
    plt.rcParams["figure.figsize"] = [20, 20]
    plt.imshow(img)
#saving and showing the tree
show_tree(dt, features, 'dec_tree_o1.png')




#prediction fror n sample minimum
y_pred1 = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred1)*100
print('Accuracy using limit n sample in each leaf: ', round(score, 1))

##Predition for entropy
y_pred2 = classifier2.predict(X_test)
from sklearn.metrics import accuracy_score
score2 = accuracy_score(y_test, y_pred2)*100
print('Accuracy using entropy: ', round(score2, 1))

#prediction using random forest
y_pred3 = classifier3.predict(X_test)
from sklearn.metrics import accuracy_score
score3 = accuracy_score(y_test, y_pred3)*100
print('Accuracy using random forest: ', round(score3, 1))