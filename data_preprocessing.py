
import re
import numpy as np
import pandas as  pd

import gensim
import gensim.corpora as corpora
import seaborn as sns
from gensim.utils import simple_preprocess
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score,plot_confusion_matrix
from gensim.models import CoherenceModel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords

import spacy
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import data_preprocessing
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler
import seaborn as sb
from sklearn.manifold import TSNE
from mlxtend.evaluate import bias_variance_decomp
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
nltk.download('averaged_perceptron_tagger')
lemmatizer = WordNetLemmatizer()

def get_data_from_tsv(trainPath,testPath,validationPath):
    train_df = pd.read_csv(trainPath, sep='\t', names=["ID", "Label", "Content", "Subjects", "Speaker", "Job", "State", "Party", "Barely True", "False", "Half True", "Mostly True", "Pants on Fire", "Venue"])
    test_df = pd.read_csv(testPath, sep='\t', names=["ID", "Label", "Content", "Subjects", "Speaker", "Job", "State", "Party", "Barely True", "False", "Half True", "Mostly True", "Pants on Fire", "Venue"])
    val_df = pd.read_csv(validationPath, sep='\t', names=["ID", "Label", "Content", "Subjects", "Speaker", "Job", "State", "Party", "Barely True", "False", "Half True", "Mostly True", "Pants on Fire", "Venue"])
    
    return train_df,test_df,val_df


def get_world_cloud(data_textOnly):
    wordcloud = WordCloud().generate(data_textOnly)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40).generate(data_textOnly)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    
# Define function for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts,stop_words):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts,bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts,bigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts,nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def preprocess_text(sen):
    snow=nltk.stem.SnowballStemmer('english')
    sen=sen.lower()
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    words=[snow.stem(word) for word in sentence.split()]
    sentence=" ".join(words)
    return sentence.strip()

def filter_data(df):
    filter = df["Content"] != ""
    df = df[filter]
    df = df.dropna()
    return df

def format_data(df,pred="single"):
    X = []
    sentences = list(df["Content"])
    for sen in sentences:
        X.append(preprocess_text(sen))
    y = df["Label"].values
    if pred=="single":
        T=[]
        for val in y:
            if val=="TRUE" or val=="mostly-true":
                T.append(1)
            else:
                T.append(0)
        y=np.array(T)
    elif pred=="multi":
         Y=[]
         for val in y:
            if val=="TRUE" or val=="true":
                Y.append(1)  
            elif val=="mostly-true":
                Y.append(0.75)
            elif val=="half-true":
                Y.append(0.50)
            elif val=="barely-true":
                Y.append(0.25)
            elif val=="FALSE" or val=="false":
                Y.append(0)
            elif val=="pants-fire":
                Y.append(-0.25)
            else:
                print(val)
         y=np.array(Y)
    return X,y

def data_preprocess(trainPath,testPath,validationPath,pred="single"):
    
    hate_comment,test_df,val_df = get_data_from_tsv(trainPath,testPath,validationPath)
    hate_comment=filter_data(hate_comment)
    test_df=filter_data(test_df)
    val_df=filter_data(val_df)
    
    train,y=format_data(hate_comment,pred=pred)
    test,testY=format_data(test_df,pred=pred)
    val,valY=format_data(val_df,pred=pred)
    
    return train,y,test,testY,val,valY
    
        


def get_accuracy(train,y,test,testY,model,lr,show_coefficients=False):
    review_model=model.fit_transform(train)
    final_model=lr.fit(review_model,y)
    accuracy=final_model.score(model.fit_transform(test),testY)
    print("Accuracy is "+str(accuracy))
    if show_coefficients:
        df=pd.DataFrame({'Word':model.get_feature_names(),'Coefficient':final_model.coef_.tolist()[0]}).sort_values(['Coefficient','Word'],ascending=[0,1])
        print("-----------------Top 15 positive words------------")
        print(df.head(15).to_string(index=False))
        print("-----------------Top 15 negative words------------")
        print(df.tail(15).to_string(index=False))
    return final_model

def get_accuracy_with_confusion_matrix(train,y,test,testY,model,classicmodel,nump=False,senti="false"):
    if senti=="add":
        train_senti,test_senti=append_senti_to_vect(train,test)
    train_model=model.fit_transform(train)
    test_model=model.fit_transform(test)
    if nump:
        train_model=train_model.toarray()
        test_model=test_model.toarray()
    if senti=="add":
        train_model=np.c_[train_model,train_senti]
        test_model=np.c_[test_model,test_senti]
    final_model=classicmodel.fit(train_model,y)
    yhat=final_model.predict(test_model)
    print("Accuracy :", np.mean(yhat == testY))
    print(classification_report(testY, yhat))
    mse, bias, var = bias_variance_decomp(final_model,train_model,y,test_model,testY,loss='mse',num_rounds=200, random_seed=1)
    print("MSE : "+str(mse))
    print("Bias : "+str(bias))
    print("Variance : "+str(var))
    
    confusionMatrix(testY,yhat)
    return final_model

def confusionMatrix(testset,predicted):
    #Confusion matrix
    print("======Confusion Matrix======")
    matrix = confusion_matrix(testset,predicted)
    print('\n',matrix)

    pd.crosstab(np.array(predicted), np.array(testset), rownames=['Actual'], colnames=['Predicted'], margins=True)

    p = sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
        
def plot_data_using_TSNE(X,y,vec="count"):
    # Initializing vectorizer for bigram
    if vec=="count":
        count_vect = CountVectorizer(ngram_range=(1,2),max_features=300)
    else:
        count_vect = TfidfVectorizer(ngram_range=(1,2),max_features=300)

    # Initializing standard scaler
    std_scaler = StandardScaler(with_mean=False)

    # Creating count vectors and converting into dense representation
    sample_points = X
    sample_points = count_vect.fit_transform(sample_points)
    sample_points = std_scaler.fit_transform(sample_points)
    sample_points = sample_points.todense()

    # Storing class label in variable
    labels = y

    # Getting shape
    print(sample_points.shape, labels.shape)
    
    tsne_data = sample_points
    tsne_labels = labels

    # Initializing with most explained variance
    model = TSNE(n_components=2, random_state=15, perplexity=50, n_iter=2000)

    # Fitting model
    tsne_data = model.fit_transform(tsne_data)

    # Adding labels to the data point
    tsne_data = np.vstack((tsne_data.T, tsne_labels)).T

    # Creating data frame
    tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1', 'Dim_2', 'label'))

    # Plotting graph for class labels
    sb.FacetGrid(tsne_df, hue='label', size=5).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
    plt.title("TSNE with default parameters")
    plt.xlabel("Dim_1")
    plt.ylabel("Dim_2")
    plt.show()
    
def get_accuracies(train,y,test,testY,model,lr,show_coefficients=False):
    review_model=model.fit_transform(train)
    final_model=lr.fit(review_model,y)
    accuracy=final_model.score(model.fit_transform(test),testY)
#     print("Accuracy is "+str(accuracy))
    if show_coefficients:
        df=pd.DataFrame({'Word':model.get_feature_names(),'Coefficient':final_model.coef_.tolist()[0]}).sort_values(['Coefficient','Word'],ascending=[0,1])
        print("-----------------Top 15 positive words------------")
        print(df.head(15).to_string(index=False))
        print("-----------------Top 15 negative words------------")
        print(df.tail(15).to_string(index=False))
    return final_model, accuracy
 
 
def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None
 

 
 
def swn_polarity(text):
    """
    Return a sentiment polarity: 0 = negative, 1 = positive
    """
 
    sentiment = 0.0
    tokens_count = 0
 
 
    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
 
        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
 
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
 
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
 
            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
 
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1
 
    # judgment call ? Default to positive or negative
    if not tokens_count:
        return 0
 
    # sum greater than 0 => positive sentiment
    if sentiment >= 0:
        return 1
 
    # negative sentiment
    return 0



def append_senti_to_sent(train,test,val):
    
    def senti_append(sentences): 
        X = []  
        for i in range(len(sentences)):
            sen=sentences[i]
            if swn_polarity(sen)==1:
                X.append(sen+" positive")
            else:
                X.append(sen+" negative")
        return X
    train=senti_append(train)
    test=senti_append(test)
    val=senti_append(val)
    return train,test,val


def append_senti_to_vect(train,test):
    
    def senti_append(sentences): 
        senti = []  
        for i in range(len(sentences)):
            sen=sentences[i]
            if swn_polarity(sen)==1:
                senti.append(1)
            else:
                senti.append(0) 
        return senti
    train_senti=senti_append(train)
    test_sent=senti_append(test)
    return train_senti,test_sent
     
    