#!/usr/bin/env python
# coding: utf-8

# # Lesson 1 - Your first generations with Amazon Bedrock

# Welcome to Lesson 1. 
# 
# You'll start with using Amazon Bedrock to prompt a model and customize how it generates its response.
# 
# **Note:** To access the `requirements.txt` file, go to `File` and click on `Open`. Here, you will also find all helpers functions and datasets used in each lesson.
#  
# I hope you enjoy this course!

# ### Import all needed packages

# In[1]:


import boto3
import json


# ### Setup the Bedrock runtime

# In[2]:


bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')


# In[3]:


prompt = "Write a one sentence summary of Las Vegas."


# In[4]:


kwargs = {
    "modelId": "amazon.titan-text-lite-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body": json.dumps(
        {
            "inputText": prompt
        }
    )
}


# In[5]:


response = bedrock_runtime.invoke_model(**kwargs)


# In[6]:


response


# In[7]:


response_body = json.loads(response.get('body').read())


# In[8]:


print(json.dumps(response_body, indent=4))


# In[9]:


print(response_body['results'][0]['outputText'])


# ### Generation Configuration

# In[10]:


prompt = "Write a summary of Las Vegas."


# In[11]:


kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body" : json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 100,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
    )
}


# In[12]:


response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

generation = response_body['results'][0]['outputText']
print(generation)


# In[13]:


print(json.dumps(response_body, indent=4))


# In[14]:


kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body" : json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 500,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
    )
}


# In[15]:


response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

generation = response_body['results'][0]['outputText']
print(generation)


# In[16]:


print(json.dumps(response_body, indent=4))


# ### Working with other type of data

# In[17]:


from IPython.display import Audio


# In[18]:


audio = Audio(filename="dialog.mp3")
display(audio)


# In[19]:


with open('transcript.txt', "r") as file:
    dialogue_text = file.read()


# In[20]:


print(dialogue_text)


# In[21]:


prompt = f"""The text between the <transcript> XML tags is a transcript of a conversation. 
Write a short summary of the conversation.

<transcript>
{dialogue_text}
</transcript>

Here is a summary of the conversation in the transcript:"""


# In[22]:


print(prompt)


# In[23]:


kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body": json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0,
                "topP": 0.9
            }
        }
    )
}


# In[24]:


response = bedrock_runtime.invoke_model(**kwargs)


# In[25]:


response_body = json.loads(response.get('body').read())
generation = response_body['results'][0]['outputText']


# In[26]:


print(generation)


# In[ ]:





# In[ ]:




