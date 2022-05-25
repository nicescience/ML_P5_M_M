import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from langdetect import detect
import re
import nltk; nltk.download('popular')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
import spacy
from spacy import displacy
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from IPython.core.display import display, HTML
from sklearn.decomposition import NMF
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_score
from sklearn.ensemble import RandomForestClassifier
import joblib
#from wordcloud import WordCloud


#read data 
df_init=pd.read_csv('data/QueryResults.csv',encoding='latin-1')
#df explore
print('df.shape : ' ,df_init.shape)
print('df.head() : ', df_init.head())
print('\n')
print(df_init.info(),'\n')
print('null values ? ', df_init.isnull().values.any(),'\n')

#1.1 Analyse de la longueur des questions et filtre sur les questions ayant des tags dans le top 30 des tags les plus utilisés sur Stackoverflow et dont la langue est anglais
#Body length
body_length=df_init['Body'].str.len()
print('longueur de la question :' ,body_length )

fig = plt.figure(figsize=(20, 12))
ax = sns.displot(x=body_length,binwidth=3500)

plt.axvline(df_init.Body.str.len().median() - df_init.Body.str.len().min(),
            color="r", linestyle='--',
            label="médiane : "+str(df_init.Body.str.len().median()))

plt.title("longueur du corps des questions Stackoverflow "
          )
plt.legend()
plt.show()
#==> la majeure partie des questions ont une longueur <5000

#je garde  les questions ayant un body <5000 caractères
df_init = df_init[df_init.Body.str.len() < 5000]
print('shape df ', df_init.shape)

#analyse tags
print('df[tags]',df_init['Tags'].head())
# Replacer < > avec ,
df_init['Tags'] = df_init['Tags'].str.translate(str.maketrans({'<': '', '>': ','}))
df_init['Tags'] =df_init['Tags'].str[:-1]
print('remplacement de < > par , dans tags ',df_init['Tags'].head())

def count_split_tags(df, column, separator):

    list_words = []
    for word in df[column].str.split(separator):
        list_words.extend(word)
    df_list_words = pd.DataFrame(list_words, columns=["Tag"])
    df_list_words = df_list_words.groupby("Tag")\
        .agg(tag_count=pd.NamedAgg(column="Tag", aggfunc="count"))
    df_list_words.sort_values("tag_count", ascending=False, inplace=True)
    return df_list_words

tags_list = count_split_tags(df=df_init, column='Tags', separator=',')
print("Le jeu de données compte {} tags.".format(tags_list.shape[0]))

# Plot the results of splits
fig = plt.figure(figsize=(15, 8))
sns.barplot(data=tags_list.iloc[0:30, :],
            x=tags_list.iloc[0:30, :].index,
            y="tag_count")
plt.xticks(rotation=90)
plt.title("Le top 30 des tags les plus utilisés")
plt.show()

# Create a list of Tags and count the number
df_init['Tags_list'] = df_init['Tags'].str.split(',')
df_init['Tags_count'] = df_init['Tags_list'].apply(lambda x: len(x))

# Plot the result
fig = plt.figure(figsize=(10, 6))
ax = sns.countplot(x=df_init.Tags_count, color="#f48020")
ax.set_xlabel("Tags")
plt.title("Nombre de tags utilisé par question",
          fontsize=18, color="#641E15")
plt.show()

def filter_tag(x, top_list):

    temp_list = []
    for item in x:
        if (item in top_list):
            #x.remove(item)
            temp_list.append(item)
    return temp_list

top_tags = list(tags_list.iloc[0:30].index)
df_init['Tags_list'] = df_init['Tags_list']\
                    .apply(lambda x: filter_tag(x, top_tags))
df_init['number_of_tags'] = df_init['Tags_list'].apply(lambda x : len(x))
df_init = df_init[df_init.number_of_tags > 0]
print("shape du jeu de données : {} questions.".format(df_init.shape[0]))

#analyse de langue
def detect_lang(x):
    try:
        return detect(x)
    except:
        pass

df_init['short_body'] = df_init['Body'].apply(lambda x: x[0:100])
df_init['lang'] =df_init.short_body.apply(detect_lang)

print('langue détectée dans les questions ',pd.DataFrame(df_init.lang.value_counts()))
#l'anglais est majoritaire dans le jeu de données donc on supprime les autres langues pour simplifier l'analyse
df_init = df_init[df_init['lang']=='en']
print("shape du jeu de données après filtre sur la langue==en ", df_init.shape)

#1.2 text preprocessing
#suppression des balises HTML qui sont dans le Body des questions
def remove_html(x):
    beautifulsoup = BeautifulSoup(x,"lxml")
    to_remove = beautifulsoup.findAll("code")
    for code in to_remove:
        code.replace_with(" ")
    return str(beautifulsoup)

df_init['Body'] = df_init['Body'].apply(remove_html)
# Delete all html tags
df_init['Body'] = [BeautifulSoup(text,"lxml").get_text() for text in df_init['Body']]
print("suppression balise html du body ", df_init['Body'].head(),'\n')




"""doctest=nlp("marwa teste ceci depuis quatre jours ")
for token in doctest:
    print("printing ",token.text,token.pos_)"""
    

def text_cleaner(x, nlp):
    # POS not in "NOUN", "PROPN"
    #x = remove_pos(nlp, x, pos_list)
    # Case lower
    x = x.lower()
    # Remove unicode characters
    x = x.encode("ascii", "ignore").decode()
    # Remove English contractions
    x = re.sub("\'\w+", '', x)
    # Remove ponctuation but not # (for C# for example)
    x = re.sub('[^\\w\\s#]', '', x)
    # Remove links
    x = re.sub(r'http*\S+', '', x)
    # Remove numbers
    x = re.sub(r'\w*\d+\w*', '', x)
    # Remove extra spaces
    x = re.sub('\s+', ' ', x)
    
    # Return cleaned text
    return x

    #clean body
nlp = spacy.load("en_core_web_sm")
pos_list=["NOUN","PROPN"]
def remove_pos(d):
    doc1= nlp(d)
    list_text_row = []
    for token in doc1:
        if(token.pos_ in pos_list) :
            list_text_row.append(token.text)
    join_text_row = " ".join(list_text_row)
    join_text_row = join_text_row.lower().replace("c #", "c#")
    return join_text_row

tqdm.pandas()
df_init['Body_cleaned'] = df_init["Body"].progress_apply(lambda x : text_cleaner(x, nlp))

print("spacy clean ", df_init['Body_cleaned'].head(20),'\n')
df_init['Body_cleaned'] = df_init["Body"].progress_apply(lambda x : remove_pos(x))
print("spacy keep noun ", df_init['Body_cleaned'].head(20),'\n')

lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

df_init['Body_cleaned'] = df_init.Body_cleaned.apply(lemmatize_text)
print('\n')
print("lemma text ", df_init['Body_cleaned'])

# Create a list of all tokens for Body
full_corpus = []
for i in df_init['Body_cleaned']:
    full_corpus.extend(i)


# Calculate distribition of words in Body token list
body_dist = nltk.FreqDist(full_corpus)
body_dist = pd.DataFrame(body_dist.most_common(2000),
                         columns=['Word', 'Frequency'])


# Plot word cloud with tags_list (frequencies)
"""fig = plt.figure(1, figsize=(17, 12))
ax = fig.add_subplot(1, 1, 1)
wordcloud = WordCloud(width=900, height=500,
                      background_color="black",
                      max_words=500, relative_scaling=1,
                      normalize_plurals=False)\
    .generate_from_frequencies(body_dist.set_index('Word').to_dict()['Frequency'])

ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
plt.title("Word Cloud of 500 most popular words on Body feature\n",
          fontsize=18, color="#641E16")
plt.show()"""

def text_cleaner(x, nlp, pos_list, lang="english"):
    x = remove_pos(x)
    # Case normalization
    x = x.lower()
    # Remove unicode characters
    x = x.encode("ascii", "ignore").decode()
    # Remove English contractions
    x = re.sub("\'\w+", '', x)
    # Remove ponctuation but not # (for C# for example)
    x = re.sub('[^\\w\\s#]', '', x)
    # Remove links
    x = re.sub(r'http*\S+', '', x)
    # Remove numbers
    x = re.sub(r'\w*\d+\w*', '', x)
    # Remove extra spaces
    x = re.sub('\s+', ' ', x)
        
    # Tokenization
    x = nltk.tokenize.word_tokenize(x)
    # List of stop words in select language from NLTK
    stop_words = stopwords.words(lang)
    # Remove stop words
    x = [word for word in x if word not in stop_words 
         and len(word)>2]
    # Lemmatizer
    wn = nltk.WordNetLemmatizer()
    x = [wn.lemmatize(word) for word in x]
    
    # Return cleaned text
    return x


pos_list = ["NOUN","PROPN"]
tqdm.pandas()
df_init['Title_cleaned'] = df_init.Title.progress_apply(lambda x: text_cleaner(x,nlp,pos_list,"english"))

print("nettoyage titre ",df_init['Title_cleaned'].head(20))

df_init = df_init[['Title_cleaned',
             'Body_cleaned',
             'Score',
             'Tags_list']]

df_init = df_init.rename(columns={'Title_cleaned': 'Title',
                            'Body_cleaned': 'Body',
                            'Tags_list': 'Tags'})

print ("dataset nettoye ", df_init.head(),"\n")

#Transformation en matrix de TF-IDF features
df_init["Text"] = df_init["Title"] + df_init["Body"]
print("df_init[Text].head(3)", df_init["Text"].head(3))
X = df_init["Text"]
y = df_init["Tags"]

print( "X shape ", X.shape)
print("y shape ",y.shape)
vectorizer = TfidfVectorizer(analyzer="word",
                             max_df=0.5,
                             min_df=0.001,
                             ngram_range=[1,3],
                             tokenizer=None,
                             preprocessor=' '.join,
                             stop_words='english',
                             lowercase=False)

vectorizer.fit(X)
X_tf_idf = vectorizer.transform(X)

print('X_tf_idf.shape',X_tf_idf.shape,'\n')
print('y.shape',y.shape)


multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(y)
y_binarized = multilabel_binarizer.transform(y)

# Create train and test split (30%)
X_train, X_test, y_train, y_test = train_test_split(X_tf_idf, y_binarized,
                                                    test_size=0.3, random_state=10)
print("X_train shape : {}".format(X_train.shape))
print("X_test shape : {}".format(X_test.shape))
print("y_train shape : {}".format(y_train.shape))
print("y_test shape : {}".format(y_test.shape))

#Modèle Logistic regression ( OvR)
# Initialize Logistic Regression with OneVsRest
param_logit = {"estimator__C": [2,3,4,5,8,10, 0.1],
               "estimator__penalty": ["l1", "l2"],
               "estimator__solver": ["liblinear"]}

logit_cv = GridSearchCV(OneVsRestClassifier(LogisticRegression()),
                              param_grid=param_logit,
                              n_jobs=-1,
                              cv=5,
                              return_train_score = True
                              )
best=logit_cv.fit(X_train, y_train)

best.best_params_

best.best_score_

y_pred_logit=logit_cv.predict(X_test)

#to do add label 
print(classification_report(y_test,y_pred_logit))

jaccard_score(y_test, y_pred_logit, average='weighted')

# Inverse transform
y_test_pred_logit_inversed = multilabel_binarizer.inverse_transform(y_pred_logit)
y_test_inversed = multilabel_binarizer.inverse_transform(y_test)

print(" Logistic reg model , 2 first tags : actual vs predicted")

print("Actual:", y_test_inversed[0:3])
print("Predicted:", y_test_pred_logit_inversed[0:3])

#Random Forest classifier
param_rfc = {"estimator__max_depth": [10, 15,20],
             "estimator__max_features" : ['auto', 'sqrt'],
             "estimator__class_weight": ["balanced"]}

rfc_cv = GridSearchCV(OneVsRestClassifier(RandomForestClassifier()),
                            param_grid=param_rfc,
                            cv=2,
                            scoring="f1_weighted",
                            return_train_score = True,
                            verbose=3)

rfc_cv.fit(X_train, y_train)

print("Best params for RandomForestClassifier")

rfc_best_params = rfc_cv.best_params_
print(rfc_best_params)

rfc_best_params

rfc_best = {}
for i, j in rfc_best_params.items():
    rfc_best[i.replace("estimator__","")] =j

# Predict
y_test_pred_labels_tfidf_rfc = rfc_cv.predict(X_test)

# Inverse transform
y_test_pred_inversed_rfc = multilabel_binarizer.inverse_transform(y_test_pred_labels_tfidf_rfc)


print("Random Forest : Print  predicted Tags vs actual Tags")
print("True:", y_test_inversed[0:4])
print("Predicted:", y_test_pred_inversed_rfc[0:4])


jaccard_score(y_test, y_test_pred_labels_tfidf_rfc , average='weighted')

print(classification_report(y_test,y_test_pred_labels_tfidf_rfc))

# Export fitted model and Preprocessor
joblib.dump(logit_cv,'logisticreg_nlp_model.pkl')
joblib.dump(vectorizer,'tfidf_vectorizer.pkl')
joblib.dump(multilabel_binarizer,'multilabel_binarizer.pkl')
