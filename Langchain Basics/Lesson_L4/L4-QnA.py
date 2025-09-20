#!/usr/bin/env python
# coding: utf-8

# # LangChain: Q&A over Documents
# 
# An example might be a tool that would allow you to query a product catalog for items of interest.

# In[ ]:


#pip install --upgrade langchain


# In[1]:


import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


# Note: LLM's do not always produce the same results. When executing the code in your notebook, you may get slightly different answers that those in the video.

# In[2]:


# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"


# In[3]:


from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.llms import OpenAI


# In[4]:


file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)


# In[5]:


from langchain.indexes import VectorstoreIndexCreator


# In[ ]:


#pip install docarray


# In[6]:


index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])


# In[7]:


query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."


# **Note**:
# - The notebook uses `langchain==0.0.179` and `openai==0.27.7`
# - For these library versions, `VectorstoreIndexCreator` uses `text-davinci-003` as the base model, which has been deprecated since 1 January 2024.
# - The replacement model, `gpt-3.5-turbo-instruct` will be used instead for the `query`.
# - The `response` format might be different than the video because of this replacement model.

# In[8]:


llm_replacement_model = OpenAI(temperature=0, 
                               model='gpt-3.5-turbo-instruct')

response = index.query(query, 
                       llm = llm_replacement_model)


# In[9]:


display(Markdown(response))


# ## Step By Step

# In[10]:


from langchain.document_loaders import CSVLoader
loader = CSVLoader(file_path=file)


# In[11]:


docs = loader.load()


# In[12]:


docs[0]


# In[13]:


from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()


# In[14]:


embed = embeddings.embed_query("Hi my name is Harrison")


# In[15]:


print(len(embed))


# In[16]:


print(embed[:5])


# In[17]:


db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)


# In[18]:


query = "Please suggest a shirt with sunblocking"


# In[19]:


docs = db.similarity_search(query)


# In[20]:


len(docs)


# In[21]:


docs[0]


# In[22]:


retriever = db.as_retriever()


# In[23]:


llm = ChatOpenAI(temperature = 0.0, model=llm_model)


# In[24]:


qdocs = "".join([docs[i].page_content for i in range(len(docs))])


# In[25]:


response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 


# In[27]:


display(Markdown(response))


# In[28]:


qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)


# In[29]:


query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."


# In[30]:


response = qa_stuff.run(query)


# In[31]:


display(Markdown(response))


# In[32]:


response = index.query(query, llm=llm)


# In[33]:


index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])


# Reminder: Download your notebook to you local computer to save your work.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




