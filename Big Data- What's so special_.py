
# coding: utf-8

# In[1]:


# converting business.json to spark dataframe
business_json=spark.read.json('gs://bucketonee/yelp_academic_dataset_business.json')


# In[2]:


# Converting review.json to spark dataframe
review_json=spark.read.json('gs://bucketonee/yelp_academic_dataset_review.json')


# In[3]:


# Converting spark dataframe to pandas dataframe
yelp_business = business_json.select("*").toPandas()


# In[20]:


# installing and importing modules
get_ipython().system(u' pip install wordcloud')
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer 
ps = PorterStemmer() 
stop_words = set(stopwords.words('english')) 
import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
from nltk.util import ngrams
import collections
WNL = nltk.WordNetLemmatizer()


# In[32]:


# preparing stopwords list
def prepareStopWords():
 
    stopwordsList = []
 
    # Load default stop words and add a few more specific to my text.
    stopwordsList = stopwords.words('english')
    stopwordsList.append('dont')
    stopwordsList.append('didnt')
    stopwordsList.append('doesnt')
    stopwordsList.append('cant')
    stopwordsList.append('couldnt')
    stopwordsList.append('couldve')
    stopwordsList.append('im')
    stopwordsList.append('ive')
    stopwordsList.append('isnt')
    stopwordsList.append('theres')
    stopwordsList.append('wasnt')
    stopwordsList.append('wouldnt')
    stopwordsList.append('a')
    stopwordsList.append('also')
    stopwordsList.append('japanese')
    stopwordsList.append('chinese')
    stopwordsList.append('food')
    stopwordsList.append('restaurant')
    stopwordsList.append('place')
    stopwordsList.append('great')
    stopwordsList.append('go')
    stopwordsList.append('get')
    stopwordsList.append('Services')
 
    return stopwordsList


# In[64]:


def special(name_of_business):
    
    # Extracting business id for the name of business and corresponding reviews
    s=''
    id1=str(list(yelp_business[yelp_business['name']==name_of_business]['business_id'])[0])
    review1=review_json.filter(review_json.business_id.isin([id1])).toPandas()
    review=review1[(review1['stars']==4.0)|(review1['stars']==5.0) ]['text']
    
    # creating a single string of reviews per business
    for x in review:
        s=s+x
    rawText=s
    
    #pre-processing the text
    rawText = rawText.lower() #converting to lower case

    # Removing single quote early since it causes problems with the tokenizer.
    rawText = rawText.replace("'", "")

    tokens = nltk.word_tokenize(rawText) # Tokenizing string to words
    text = nltk.Text(tokens)

    # Loading stop words list
    stopWords = prepareStopWords()

    # Removing extra chars and remove stop words.
    text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text]
    text_content = [word for word in text_content if word not in stopWords]

    # After the punctuation above is removed there may still be empty entries in the list.
    # Remove any entries where the len is zero.
    text_content = [s for s in text_content if len(s) != 0]

    d=dict()
    text_content = [WNL.lemmatize(t) for t in text_content] # lemmatization to reduce number of similar words
    unigram=text_content # one word list
    bigrams = ngrams(text_content,2) # 2 continuous word list
    trigrams=ngrams(text_content,3) # 3 continuous words list
    
    # using Collections.Counter to get the occurance frequency of each word and n grams
    scoredList = {k: v for k, v in (collections.Counter(bigrams).items())}
    d={k: v for k, v in (collections.Counter(unigram).items())}
    for i in scoredList.items():
        d[i[0][0]+'_'+i[0][1]]=i[1]
    scoredList1 = {k: v for k, v in (collections.Counter(trigrams).items())}
    for i in scoredList1.items():
        d[i[0][0]+'_'+i[0][1]+'_'+i[0][2]]=i[1]
    # Sorting to get most frequent words
    s=sorted(d.items(), key=lambda x: x[1], reverse=True)
    # Generating and returning a list of 5 specialities 
    l=list()
    for x in s[0:5]:
        l.append(str(x[0]))
    return l


# In[70]:


# extracting list of specialities for first random 50 businesses
l1=list()
for x in yelp_business['name'][:50]:
    l1.append(str(special(x)))   


# In[72]:


# creating a dataframe to store in big query
b=yelp_business[['business_id','name']][:50]
b['special']=l1
b


# In[76]:


# Storing the dataframe to big query
import google.datalab.bigquery as bq
bigquery_dataset_name = ('bighw3', 'Speciality')
dataset = bq.Dataset(name = bigquery_dataset_name)

bigquery_table_name = ('bighw3', 'Speciality', 'special')
table = bq.Table(bigquery_table_name)

if not dataset.exists():
    dataset.create()

# Creating or overwriting the existing table if it exists
table_schema = bq.Schema.from_data(b)
table.create(schema = table_schema, overwrite = True)

# Writing the DataFrame to a BigQuery table
table.insert(b)


# #  What's so special at a particular location?

# In[70]:


def special2(name_of_location):
    
    # Extracting categories for each city
    st=''
    categories=yelp_business[yelp_business['city']==name_of_location]['categories']
        
    # creating a single string of reviews per business
    for x in categories:
        if x!= None:
            st=st+x.replace(',','')
    #pre-processing the text
  # Removing single quote early since it causes problems with the tokenizer.
    rawText = rawText.replace("'", "")

    tokens = nltk.word_tokenize(rawText) # Tokenizing string to words
    text = nltk.Text(tokens)

    # Loading stop words list
    stopWords = prepareStopWords()

    # Removing extra chars and remove stop words.
    text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text]
    text_content = [word for word in text_content if word not in stopWords]

    # After the punctuation above is removed there may still be empty entries in the list.
    # Remove any entries where the len is zero.
    text_content = [s for s in text_content if len(s) != 0]

    d=dict()
    text_content = [WNL.lemmatize(t) for t in text_content] # lemmatization to reduce number of similar words
    unigram=text_content # one word list
    bigrams = ngrams(text_content,2) # 2 continuous word list
    trigrams=ngrams(text_content,3) # 3 continuous words list
    
    # using Collections.Counter to get the occurance frequency of each word and n grams
    scoredList = {k: v for k, v in (collections.Counter(bigrams).items())}
    d={k: v for k, v in (collections.Counter(unigram).items())}
    for i in scoredList.items():
        d[i[0][0]+'_'+i[0][1]]=i[1]
    scoredList1 = {k: v for k, v in (collections.Counter(trigrams).items())}
    for i in scoredList1.items():
        d[i[0][0]+'_'+i[0][1]+'_'+i[0][2]]=i[1]
    # Sorting to get most frequent words
    s=sorted(d.items(), key=lambda x: x[1], reverse=True)
    # Generating and returning a list of 5 specialities 
    l=list()
    for x in s[0:5]:
        l.append(str(x[0]))
    return l


# In[71]:


special2('Mooresville')


# In[72]:


# extracting list of locations and their specialities for some random locations
l1=list()
l2=list()
for x in list(set(yelp_business[yelp_business['city']!='']['city']))[2:23]:
    l1.append(str(special2(x)))
    l2.append(x)


# In[76]:


# creating a dataframe to store in big query
import pandas as pd
df = pd.DataFrame(l2, columns =['Location'])
df['Specialities']=l1
df


# In[79]:


bigquery_dataset_name = ('bighw3', 'Speciality')
dataset = bq.Dataset(name = bigquery_dataset_name)

bigquery_table_name = ('bighw3', 'Speciality', 'special2')
table = bq.Table(bigquery_table_name)

if not dataset.exists():
    dataset.create()

# Create or overwrite the existing table if it exists
table_schema = bq.Schema.from_data(df)
table.create(schema = table_schema, overwrite = True)

# Write the DataFrame to a BigQuery table
table.insert(df)

