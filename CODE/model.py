# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load_data
train_data = pd.read_csv('../DATA/data.csv')


# %%
train_data.head()

# %%
print("NUMBER OF DATA POINTS -", train_data.shape[0])
print("NUMBER OF FEATURES -", train_data.shape[1])
print("FEATURES -", train_data.columns.values)

# %%
train_data['Category'].value_counts()

# %%
train_data.dropna(inplace=True)
train_data.isna().sum()

# %%
target_category = train_data['Category'].unique()
print(target_category)

# %%
news_cat = train_data['Category'].value_counts()

plt.figure(figsize=(10, 5))
my_colors = ['yellow', 'violet', 'crimson', 'm', 'b']
news_cat.plot(kind='bar', color=my_colors)
plt.grid()
plt.xlabel("News Categories")
plt.ylabel("Datapoints Per Category")
plt.title("Distribution of Datapoints Per Category")
plt.show()

# %%
warnings.filterwarnings("ignore")

# loading_the_stop_words_from_nltk_library_
stop_words = set(stopwords.words('english'))


def txt_preprocessing(total_text, index, column, df):
    if type(total_text) is not int:
        string = ""

        # replace_every_special_char_with_space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)

        # replace_multiple_spaces_with_single_space
        total_text = re.sub('\s+', ' ', total_text)

        # converting_all_the_chars_into_lower_case
        total_text = total_text.lower()

        for word in total_text.split():
            # if_the_word_is_a_not_a_stop_word_then_retain_that_word_from_the_data
            if not word in stop_words:
                string += word + " "

        df[column][index] = string

# %%
# train_data_text_processing_stage_


for index, row in train_data.iterrows():
    if type(row['Text']) is str:
        txt_preprocessing(row['Text'], index, 'Text', train_data)
    else:
        print("THIS INDEX SHOULD NOT OCCUR :", index)

train_data.head()

# %%
X_train = train_data
y_train = train_data['Category']

X_train, X_cv, y_train, y_cv = train_test_split(
    X_train, y_train, test_size=0.20, stratify=y_train, random_state=0)

print("NUMBER OF DATA POINTS IN TRAIN DATA :", X_train.shape[0])
print("NUMBER OF DATA POINTS IN CROSS VALIDATION DATA :", X_cv.shape[0])

# %%

text_vectorizer = CountVectorizer(min_df=3)
train_text_ohe = text_vectorizer.fit_transform(X_train['Text'])

# getting all the feature names (words)
train_text_features = text_vectorizer.get_feature_names()

# train_text_ohe.sum(axis=0).A1 will sum every row and returns (1*number of features) vector
train_text_fea_counts = train_text_ohe.sum(axis=0).A1

# zip(list(text_features),text_fea_counts) will zip a word with its number of times it occured
text_fea_dict = dict(zip(list(train_text_features), train_text_fea_counts))

print("Total Number of Unique Words in Train Data :", len(train_text_features))

# %%
