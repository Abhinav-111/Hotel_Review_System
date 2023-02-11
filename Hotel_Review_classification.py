#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_excel("hotel_reviews.xlsx")


# In[3]:


df.head()


# In[4]:


df["Rating"].value_counts()


# In[34]:


#Rating Count
sns.set_style("darkgrid")
sns.countplot(x="Rating",hue="Rating",data=df)


# In[35]:


#Percentage of Rating distribution
print(round(df.Rating.value_counts(normalize=True)*100,2))
round(df.Rating.value_counts(normalize=True)*100,2)
round(df.Rating.value_counts(normalize=True)*100,2).plot(kind="bar",figsize=(50,30),color="green",edgecolor="orange")
plt.xlabel("Ratings",fontsize=45)
plt.ylabel("Percentage",fontsize=45)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title("Rating Percentage",fontsize=60)
plt.show()


# In[38]:



#Pie plot of percentage of ratings
plt.figure(figsize=(10,5))
plt.title('Percentage of Ratings', fontsize=10)
df.Rating.value_counts().plot(kind='pie', labels=['Rating5', 'Rating4', 'Rating3', 'Rating2', 'Rating1'],
                              wedgeprops=dict(width=.7), autopct='%1.0f%%', startangle= -20, 
                              textprops={'fontsize': 10})


# In[ ]:





# In[ ]:





# In[5]:


df_neg= df.loc[df["Rating"]<=3]
df_neg = df_neg.reset_index(drop=True)


# In[6]:


df_pos=df.loc[df["Rating"] > 3]
df_pos= df_pos.reset_index(drop=True)


# In[7]:


df_all =pd.concat([df_neg,df_pos],axis=0)
df_all= df_all.reset_index(drop=True)


# In[8]:


len(df_neg)


# In[9]:


len(df_pos)


# In[10]:


len(df_all)


# In[11]:


df_all["Sentiment"]=np.where(df_all["Rating"]>3,"Positive","Negative")


# In[12]:


df_all


# In[13]:


#Perform Eda


# In[14]:


df_all.shape


# In[15]:


df_all.drop(["Rating"],axis=1)


# In[16]:


df_all.isnull().sum()


# In[17]:


df.info()


# In[18]:


df_all["Sentiment"].unique()


# In[19]:


df_all["Sentiment"].value_counts()


# In[20]:


sns.countplot(df_all["Sentiment"])


# In[21]:


#Apply labelencoding to make target feature into numerical(positive:1,negative:0)


# In[22]:


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
df_all["Sentiment"]=label.fit_transform(df_all["Sentiment"])


# In[23]:


df_all.tail()


# In[24]:


# Divide data into independent and dependent data


# In[25]:


X=df_all["Review"]
y=df_all["Sentiment"]


# In[44]:


df_all = df_all.sample(frac=1).reset_index(drop=True)


# In[45]:


# Remove special characters from the sentence
def clean_text(sentence):
     # Convert to lower case
    sentence = sentence.lower()
    # split the sentence
    sentence = sentence.split()
    # Join the sentence
    sentence = " ".join(sentence)
    # Remove special characters from the sentence
    sentence = re.sub(f'[{re.escape(string.punctuation)}]', "", sentence)
    
    return sentence


# In[68]:


def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("[0-9" "]+"," ",text)
    text = re.sub('[‘’“”…]', '', text)
    return text

clean = lambda x: clean_text(x)
df_all["Cleaned_Reviews"]=pd.DataFrame(df_all.Review.apply(clean))


# In[51]:


#Word frequency
freq = pd.Series(' '.join(df_all['Review']).split()).value_counts()[:20] # for top 20
freq


# In[52]:


#removing stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
df_all['Review'] = df_all['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[53]:


#word frequency after removal of stopwords
freq_Sw = pd.Series(' '.join(df_all['Review']).split()).value_counts()[:20] # for top 20
freq_Sw


# In[54]:


# count vectoriser tells the frequency of a word.
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
vectorizer = CountVectorizer(min_df = 1, max_df = 0.9)
X = vectorizer.fit_transform(df_all["Review"])
word_freq_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'occurrences':np.asarray(X.sum(axis=0)).ravel().tolist()})
word_freq_df['frequency'] = word_freq_df['occurrences']/np.sum(word_freq_df['occurrences'])
#print(word_freq_df.sort('occurrences',ascending = False).head())


# In[55]:


word_freq_df.head(30)


# In[26]:


#Remove all Special and numeric character from the data and also remove stopwards and apply stemming 


# In[27]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer 


# In[28]:


ps=PorterStemmer()
corpus= []

for i in range (len(X)):
    print(i)
    review = re.sub("[^a-zA-Z]"," ",X[i])
    review = review.lower()
    review = review.split()
    review= [ps.stem(word) for word in review if word not in set(stopwords.words("english"))]
    review= " ".join (review)
    corpus.append(review)


# In[29]:


corpus


# In[69]:


#Bi-gram
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  #for tri-gram, put ngram_range=(3,3)
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]


# In[70]:


from sklearn.feature_extraction.text import CountVectorizer
top2_words = get_top_n2_words(df_all["Review"], n=200) #top 200
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
top2_df.head()


# In[71]:


#Bi-gram plot
import matplotlib.pyplot as plt
import seaborn as sns
top20_bigram = top2_df.iloc[0:20,:]
fig = plt.figure(figsize = (10, 5))
plot=sns.barplot(x=top20_bigram["Bi-gram"],y=top20_bigram["Freq"])
plot.set_xticklabels(rotation=45,labels = top20_bigram["Bi-gram"])


# In[72]:


#Tri-gram
def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]


# In[73]:


top3_words = get_top_n3_words(df_all["Review"], n=200)
top3_df = pd.DataFrame(top3_words)
top3_df.columns=["Tri-gram", "Freq"]


# In[74]:


top3_df


# In[75]:


#Tri-gram plot
import seaborn as sns
top20_trigram = top3_df.iloc[0:20,:]
fig = plt.figure(figsize = (10, 5))
plot=sns.barplot(x=top20_trigram["Tri-gram"],y=top20_trigram["Freq"])
plot.set_xticklabels(rotation=45,labels = top20_trigram["Tri-gram"])


# ## WordCloud

# In[76]:


string_Total = " ".join(df_all["Review"])


# In[77]:


#wordcloud for entire corpus
from wordcloud import WordCloud
wordcloud_stw = WordCloud(
                background_color= 'black',
                width = 3000,
                height = 2500
                ).generate(string_Total)
plt.imshow(wordcloud_stw)


# In[78]:


df_all["Cleaned_Review_Lemmatized"]=corpus


# In[79]:


#Polarity and subjectivity#
import textblob
from textblob import TextBlob


# In[80]:


df_all["Polarity"]=df_all["Cleaned_Reviews"].apply(lambda x:TextBlob(x).sentiment.polarity)
df_all["Subjectivity"]=df_all["Cleaned_Reviews"].apply(lambda x:TextBlob(x).sentiment.subjectivity)


# In[82]:


#Printing 5 reviews with highest polarity
print("5 Random Reviews with Highest Polarity:")
for index,review in enumerate(df_all.iloc[df_all['Polarity'].sort_values(ascending=False)[:5].index]['Cleaned_Reviews']):
    print('Review {}:\n'.format(index+1),review)


# In[83]:


#Printing 5 reviews with negative polarity  
print("5 Random Reviews with Lowest Polarity:")
for index,review in enumerate(df_all.iloc[df_all['Polarity'].sort_values(ascending=True)[:5].index]['Cleaned_Reviews']):
  print('Review {}:\n'.format(index+1),review)   


# In[86]:


#Frequency Distribution based on Polarity
plt.figure(figsize=(20,10),facecolor="green",edgecolor="orange")
plt.margins(0.02)
plt.xlabel("Polarity",fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel("Frequency",fontsize=20)
plt.yticks(fontsize=20)
plt.hist(df_all["Polarity"],bins=20)
plt.title("Frequency Distribution based on Polarity",fontsize=25)
plt.show()


# In[87]:


#Sentiment distribution based on ratings
polarity_avg = df_all.groupby('Rating')['Polarity'].mean().plot(kind='bar', figsize=(25,10),color="green",edgecolor="orange")
plt.xlabel('Rating', fontsize=25)
plt.ylabel('Average Sentiment', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Average Sentiment per Rating Distribution', fontsize=25)
plt.show()


# In[88]:


#Counting no of words in each review
df_all["Word Count"]=df_all["Cleaned_Review_Lemmatized"].apply(lambda x:len(str(x).split()))


# In[89]:


#Average no of words wrt ratings
word_avg=df_all.groupby("Rating")["Word Count"].mean().plot(kind="bar",figsize=(25,15),color="green",edgecolor="orange")
plt.xlabel('Rating',fontsize=15)
plt.ylabel("Average Count of Words",fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title("Average Count of Words wrt Ratings",fontsize=25)
plt.show()


# In[90]:


#Counting no of letters in each review
df_all['review_len'] = df_all['Cleaned_Review_Lemmatized'].astype(str).apply(len)


# In[91]:


#Average no of letters wrt ratings
letter_avg = df_all.groupby('Rating')['review_len'].mean().plot(kind='bar', figsize=(25,15),color="green",edgecolor="orange")
plt.xlabel('Rating', fontsize=15)
plt.ylabel('Count of Letters in Rating', fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Average Number of Letters wrt Rating ', fontsize=20)
plt.show()


# In[92]:


#Corelation of features
corelation=df_all[["Rating","Polarity","Word Count","review_len","Subjectivity"]].corr()


# In[93]:


sns.heatmap(corelation,xticklabels=corelation.columns,yticklabels=corelation.columns,annot=True)


# In[94]:


#Counting 100 most common words in our data
from nltk.probability import FreqDist
mostcommon = FreqDist(df_all["Cleaned_Review_Lemmatized"]).most_common(100)
wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(str(corpus))
fig = plt.figure(figsize=(30,10), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Top 100 Most Common Words', fontsize=100)
plt.tight_layout(pad=0)
plt.show()


# In[95]:


polarity_positive_data=pd.DataFrame(df_all.groupby("Cleaned_Reviews")["Polarity"].mean().sort_values(ascending=True))


# In[97]:


plt.figure(figsize=(25,15))
plt.xlabel('Polarity',fontsize=25)
plt.ylabel('Reviews',fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Polarity of Reviews',fontsize=30)
polarity_graph=plt.barh(np.arange(len(polarity_positive_data.index)),polarity_positive_data['Polarity'],color='purple',)


# In[105]:


#Apply TfidVectorizer to make text data into vectors


# In[106]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# In[107]:


cv= TfidfVectorizer(max_features=4000)
X=cv.fit_transform(corpus).toarray()


# In[108]:


X.shape


# In[109]:


#Split data into test and train


# In[110]:


X_train,X_test,Y_Train,Y_test= train_test_split(X,y,test_size=0.2,random_state=101)


# In[111]:


X_train.shape,X_test.shape,Y_Train.shape,Y_test.shape


# In[112]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[113]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier


# In[114]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[115]:


def evaluation_metric(y_test,y_hat,model_name):
    
    accuracy=accuracy_score(y_hat,y_test)
    print("Model: ", model_name)
    print("\nAccuracy: ", accuracy)
    print(classification_report(y_hat, y_test))
    
    plt.figure(figsize=(10,6))
    sns.heatmap(confusion_matrix(y_hat, y_test), annot=True, fmt=".2f")
    plt.show()
    return accuracy


# In[116]:


#Defining navie-bayes model


# In[117]:


classifier_NB = GaussianNB()
classifier_NB.fit(X_train, Y_Train)
pred_NB_train=classifier_NB.predict(X_train)
classifier_NB_train=np.mean(pred_NB_train==Y_Train)
classifier_NB_train
pred_NB_test=classifier_NB.predict(X_test)
classifier_NB_test=np.mean(pred_NB_test==Y_test)
classifier_NB_test


# In[118]:


classifier_NB_accuracy=evaluation_metric(pred_NB_test,Y_test,"GaussianNB")


# In[119]:


classifier_MNB = MultinomialNB()
classifier_MNB.fit(X_train, Y_Train)
pred_MNB_train=classifier_MNB.predict(X_train)
classifier_MNB_train=np.mean(pred_MNB_train==Y_Train)
classifier_MNB_train
pred_MNB_test=classifier_MNB.predict(X_test)
classifier_MNB_test=np.mean(pred_MNB_test==Y_test)
classifier_MNB_test


# In[120]:


classifier_MNB_accuracy=evaluation_metric(pred_MNB_test,Y_test,"MultinomialNB")


# In[121]:


#DecissionTree
classifier_DT = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
classifier_DT.fit(X_train,Y_Train)
pred_DT_train=classifier_DT.predict(X_train)
classifier_DT_train=np.mean(pred_DT_train==Y_Train)
classifier_DT_train
pred_DT_test=classifier_DT.predict(X_test)
classifier_DT_test=np.mean(pred_DT_test==Y_test)
classifier_DT_test


# In[122]:


classifier_DT_accuracy=evaluation_metric(pred_DT_test,Y_test,"DecisionTreeClassifier")


# In[123]:


#Logistic Regression
classifier_LR=LogisticRegression()
classifier_LR.fit(X_train,Y_Train)
pred_LR_train=classifier_LR.predict(X_train)
classifier_LR_train=np.mean(pred_LR_train==Y_Train)
classifier_LR_train
pred_LR_test=classifier_LR.predict(X_test)
classifier_LR_test=np.mean(pred_LR_test==Y_test)
classifier_LR_test


# In[124]:


classifier_LR_accuracy=evaluation_metric(pred_LR_test,Y_test,"LogisticRegression")


# In[125]:


#Adaboost
classifier_ADA=AdaBoostClassifier()
classifier_ADA.fit(X_train,Y_Train)
pred_ADA_train=classifier_ADA.predict(X_train)
classifier_ADA_train=np.mean(pred_ADA_train==Y_Train)
classifier_ADA_train
pred_ADA_test=classifier_ADA.predict(X_test)
classifier_ADA_test=np.mean(pred_ADA_test==Y_test)
classifier_ADA_test


# In[126]:


classifier_ADA_accuracy=evaluation_metric(pred_ADA_test,Y_test,"AdaBoostClassifier")


# In[127]:


#Random Forest
classifier_RF=RandomForestClassifier()
classifier_RF.fit(X_train,Y_Train)
pred_RF_train=classifier_RF.predict(X_train)
classifier_RF_train=np.mean(pred_RF_train==Y_Train)
classifier_RF_train
pred_RF_test=classifier_RF.predict(X_test)
classifier_RF_test=np.mean(pred_RF_test==Y_test)
classifier_RF_test


# In[128]:


classifier_RF_accuracy=evaluation_metric(pred_RF_test,Y_test,"RandomForestClassifier")


# In[129]:


df_models=dict()
df_models["GaussianNB"]=classifier_NB_test
df_models["MultimonialNB"]=classifier_MNB_test
df_models["DecisionTree"]=classifier_DT_test
df_models["LogisticRegression"]=classifier_LR_test
df_models["ADABoost"]=classifier_ADA_test
df_models["RandomForest"]=classifier_RF_test


# In[130]:


df_models=pd.DataFrame(list(df_models.items()),columns=["Model","Accuracy"])
df_models


# In[131]:


df_models=df_models.sort_values("Accuracy",ascending=False)
df_models


# In[132]:


sns.barplot(x="Accuracy",y="Model",data=df_models)


# In[133]:


#Difference between actual data and predicted data


# In[134]:


pd.DataFrame(np.c_[Y_test,pred_LR_test], columns=["Actual","Predicted"])


# In[135]:


#Saving logistic Regression model and TfidfVectorizer


# In[136]:


import pickle
pickle.dump(cv,open("count_Vectorizer1.pkl","wb"))
pickle.dump(classifier_LR,open("hotel_reviews_classification1.pkl","wb"))


# In[137]:


# loading ogistic Regression model and TfidfVectorizer


# In[138]:


save_cv= pickle.load(open("count_Vectorizer1.pkl","rb"))
model= pickle.load(open("hotel_reviews_classification1.pkl","rb"))


# In[139]:


# defining my function to test model


# In[140]:


def test_model(sentence):
    sen=save_cv.transform([sentence]).toarray()
    res=model.predict(sen)[0]
    if res==1:
        return "Positive review"
    else:
        return "Negative review"


# In[141]:


df_all.head()


# In[142]:


df_all.to_csv('data6.csv')


# In[143]:


#Test 1st positive review and check what does the model predict and it predicted correct


# In[147]:


sen= "excellent stayed hotel monaco past w/e delight, reception staff friendly professional room smart comfortable bed, particularly liked reception small dog received staff guests "
res= test_model(sen)
print(res)


# In[152]:


#Test 2nd negative review and check what does the model predict and it predicted correct


# In[153]:


sen="horrible customer service hotel stay february 3rd 4th 2007my friend picked hotel monaco appealing website online package included champagne late checkout 3 free valet gift spa weekend, friend checked room hours earlier came later, pulled valet young man just stood, asked valet open said, pull bags didn__Ç_é_ offer help, got garment bag suitcase came car key room number says not valet, car park car street pull, left key working asked valet park car gets, went room fine bottle champagne oil lotion gift spa, dressed went came got bed noticed blood drops pillows sheets pillows, disgusted just unbelievable, called desk sent somebody 20 minutes later, swapped sheets left apologizing, sunday morning called desk speak management sheets aggravated rude, apparently no manager kind supervisor weekend wait monday morning, young man spoke said cover food adding person changed sheets said fresh blood rude tone, checkout 3pm package booked, 12 1:30 staff maids tried walk room opening door apologizing closing, people called saying check 12 remind package, finally packed things went downstairs check, quickly signed paper took, way took closer look room, unfortunately covered food offered charged valet, called desk ask charges lady answered snapped saying aware problem experienced monday like told earlier, life treated like hotel, not sure hotel constantly problems lucky ones stay recommend anybody know,   "
res= test_model(sen)
print(res)


# In[ ]:




