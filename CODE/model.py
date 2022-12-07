# %%
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import re
import warnings
from nltk.corpus import stopwords
import nltk
import pandas as pd
import numpy as np

# %%
data = pd.read_csv('../DATA/data.csv')

# %%
data.head()

# %%
data['Category'].unique()

# %%
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
data['Category'].unique()

# %%
warnings.filterwarnings("ignore")

# Loading stopwords from nltk library
stop_words = set(stopwords.words('english'))
# Function for text preprocessing


def txt_preprocessing(total_text, index, column, df):
    if type(total_text) is not int:
        string = ""
        # Replace every special character with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)
        # Remove multiple spaces
        total_text = re.sub('\s+', ' ', total_text)
        # Converting to lowercase
        total_text = total_text.lower()

        for word in total_text.split():
            # If word is not a stopword then retain that word from the data
            if not word in stop_words:
                string += word + " "
        df[column][index] = string


# %%
for index, row in data.iterrows():
    if type(row['Text']) is str:
        txt_preprocessing(row['Text'], index, 'Text', data)

data.head()

# %%
ind = list(data.index)
np.random.shuffle(ind)

train_len = int(data.shape[0]*0.75)
train_ind = ind[:train_len]
training_data = data.iloc[train_ind, :]
# training_data.head()

test_ind = ind[train_len:]
testing_data = data.iloc[test_ind, :]
# testing_data.head()

print('Training_data size -> {}'.format(training_data.shape))
print('Testing_data size -> {}'.format(testing_data.shape))

assert data.shape[0] == len(train_ind) + \
    len(test_ind), 'Not equal distribution'

# %%


class NB:
    def __init__(self, target, dataframe):
        self.df = dataframe
        # Target/Category Column
        self.c_n = target
        # Column Names
        self.cols = list(self.df.columns)
        self.cols.remove(self.c_n)

        self.store = {}
        self.likelihood_for_all_()

    def likelihood_cal(self, x, y, z):
        """ 
        x -> Column Name (String)
        y -> Column Value (String)
        z -> Class value (String)
        c_n -> Class Name (Target)

        Returns -> P(x = y | c_n = z)
        """
        df = self.df

        if x not in self.cols:
            raise KeyError(
                "Feature(column) not present in the Training Dataset")

        res = len(df[(df[x] == y) & (df[self.c_n] == z)]) / \
            len(df[df[self.c_n] == z])

        if res == 0.0:
            return 1/(len(df[df[self.c_n] == z]) + len(df[x].unique()))

        return res

    def likelihood_for_all_(self):
        df = self.df

        dict1 = {}
        for x in self.cols:
            dict2 = {}
            for y in df[x].unique():
                dict3 = {}
                for z in df[self.c_n].unique():
                    #print('P({}="{}"|{}="{}") = {}'.format(x,y,self.c_n,z,self.likelihood_cal(x, y, z)))
                    dict3[z] = self.likelihood_cal(x, y, z)
                dict2[y] = dict3
            dict1[x] = dict2

        self.store = dict1

    def likelihood_expr(self, class_val, expr):
        val = 1

        for k, v in expr:
            try:
                store_val = self.store[k][v][class_val]
            except:
                store_val = self.likelihood_cal(k, v, class_val)

            val *= store_val

        return val

    def prior(self, class_val):
        df = self.df
        return len(df[df[self.c_n] == class_val])/df.shape[0]

    def predict(self, X):
        df = self.df

        if type(X) == pd.core.series.Series:
            values_list = [list(X.items())]

        elif type(X) == pd.core.frame.DataFrame:
            values_list = [list(y.items()) for x, y in X.iterrows()]

        else:
            raise TypeError('{} is not supported type'.format(type(X)))

        predictions_list = []
        for values in values_list:
            likelihood_priors = {}
            for class_val in df[self.c_n].unique():
                likelihood_priors[class_val] = self.prior(
                    class_val)*self.likelihood_expr(class_val, values)
            # print(likelihood_priors)

            normalizing_prob = np.sum([x for x in likelihood_priors.values()])
            probabilities = [(y/normalizing_prob, x)
                             for x, y in likelihood_priors.items()]

            if len(probabilities) == 2:
                # For 2 Class Predictions
                max_prob = max(probabilities)[1]
                predictions_list.append(max_prob)

            else:
                # For Mulit Class Predictions
                exp_1 = [np.exp(x) for x, y in probabilities]
                exp_2 = np.sum(exp_1)
                softmax = exp_1/exp_2
                # print(softmax)
                class_names = [y for x, y in probabilities]
                softmax_values = [(x, y) for x, y in zip(softmax, class_names)]
                # print(softmax_values)
                max_prob = max(softmax_values)[1]
                predictions_list.append(max_prob)

        print(probabilities)
        return predictions_list

    def accuracy_score(self, X, Y):
        assert len(X) == len(Y), 'Given values are not equal in size'

        total_matching_values = [x == y for x, y in zip(X, Y)]
        return (np.sum(total_matching_values)/len(total_matching_values))*100

    def calculate_confusion_matrix(self, X, Y):
        df = self.df

        unique_class_values = df[self.c_n].unique()
        decimal_class_values = list(range(len(unique_class_values)))
        numerical = {x: y for x, y in zip(
            unique_class_values, decimal_class_values)}

        x = [numerical[x] for x in X]
        y = [numerical[y] for y in Y]

        n = len(decimal_class_values)
        confusion_matrix = np.zeros((n, n))

        for i, j in zip(x, y):
            if i == j:
                confusion_matrix[i][i] += 1
            elif i != j:
                confusion_matrix[i][j] += 1

        return confusion_matrix

    def precision_score(self, X, Y):
        assert len(X) == len(Y), 'Given values are not equal in size'

        confusion_matrix = self.calculate_confusion_matrix(X, Y)
        tp = confusion_matrix[0][0]
        fp = confusion_matrix[1][0]

        return tp / (tp+fp)

    def recall_score(self, X, Y):
        assert len(X) == len(Y), 'Given values are not equal in size'

        confusion_matrix = self.calculate_confusion_matrix(X, Y)
        tp = confusion_matrix[0][0]
        fn = confusion_matrix[0][1]

        return tp / (tp+fn)


# %%
genx = NB(target='Category', dataframe=training_data)

# %%
y_test = list(testing_data.iloc[0:20, 2])

y_pred = genx.predict(testing_data.iloc[0:20, 0:1])
print(y_test)
print(y_pred)
print('Accuracy Score -> {} %'.format(round(genx.accuracy_score(y_test, y_pred), 3)))
# print('Precison Score -> {}'.format(round(genx.precision_score(y_test,y_pred),3)))
# print('Recall Score -> {}'.format(round(genx.recall_score(y_test,y_pred),3)))

# %%

train_data = pd.read_csv('../DATA/data.csv')

# %%
# Dividing the data into train and test set
X_train = train_data
y_train = train_data['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.20)

print("NUMBER OF DATA POINTS IN TRAIN DATA :", X_train.shape[0])
print("NUMBER OF DATA POINTS IN TEST DATA  :", X_test.shape[0])

# %%

text_vectorizer = CountVectorizer()
train_text_encoded = text_vectorizer.fit_transform(X_train['Text'])

train_text_features = text_vectorizer.get_feature_names_out()
train_text_feature_counts = train_text_encoded.sum(axis=0).A1
text_feature_dict = dict(
    zip(list(train_text_features), train_text_feature_counts))

print("Total Number of Unique Words in Train Data :", len(train_text_features))

# %%
print(len(text_feature_dict))
for word, frequency in list(text_feature_dict.items()):
    if word[0] >= '0' and word[0] <= '9':
        del text_feature_dict[word]

print(len(text_feature_dict))

# %%

train_text_encoded = normalize(train_text_encoded, axis=0)
test_text_encoded = text_vectorizer.transform(X_test['Text'])
test_text_encoded = normalize(test_text_encoded, axis=0)

# %%

alpha = [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]

cv_log_error_array = []

for i in alpha:
    clf = MultinomialNB(alpha=i)
    clf.fit(train_text_encoded, y_train)

    nb_sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    nb_sig_clf.fit(train_text_encoded, y_train)

    sig_clf_probs = nb_sig_clf.predict_proba(test_text_encoded)

    cv_log_error_array.append(
        log_loss(y_test, sig_clf_probs, labels=clf.classes_, eps=1e-15))

best_alpha = np.argmin(cv_log_error_array)

clf = MultinomialNB(alpha=alpha[best_alpha])
clf.fit(train_text_encoded, y_train)

nb_sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
nb_sig_clf.fit(train_text_encoded, y_train)

predict_y = nb_sig_clf.predict_proba(train_text_encoded)
print('For values of best alpha =', alpha[best_alpha], "The train log loss is:", log_loss(
    y_train, predict_y, labels=clf.classes_, eps=1e-6))

predict_y = nb_sig_clf.predict_proba(test_text_encoded)
print('For values of best alpha =', alpha[best_alpha], "The cross validation log loss is:", log_loss(
    y_test, predict_y, labels=clf.classes_, eps=1e-6))

# %%

clf = MultinomialNB(alpha=0.1)
clf.fit(train_text_encoded, y_train)

nb_sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
nb_sig_clf.fit(train_text_encoded, y_train)

predict_y = nb_sig_clf.predict_proba(train_text_encoded)
print("The train log loss is:", log_loss(
    y_train, predict_y, labels=clf.classes_, eps=1e-6))

predict_y = nb_sig_clf.predict_proba(test_text_encoded)
print("The cross validation log loss is:", log_loss(
    y_test, predict_y, labels=clf.classes_, eps=1e-6))

# %%
predicted_y = nb_sig_clf.predict(test_text_encoded)
train_accuracy = (nb_sig_clf.score(train_text_encoded, y_train)*100)
cv_accuracy = (accuracy_score(predicted_y, y_test)*100)

print("Naive Bayes Train Accuracy -", train_accuracy)
print("Naive Bayes CV Accuracy -", cv_accuracy)

# %%


# %%
