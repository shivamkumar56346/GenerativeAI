#!/usr/bin/env python
# coding: utf-8

# # Lesson 1: Your first agent with Amazon Bedrock

# ## Preparation 
# <p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access <code>requirements.txt</code> and <code>helper.py</code> and other files:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix - Tips and Help"</em> Lesson.</p>

# In[1]:


# Before you start, please run the following code to set up your environment.
# This code will reset the environment (if needed) and prepare the resources for the lesson.
# It does this by quickly running through all the code from the previous lessons.

get_ipython().system('sh ./ro_shared_data/reset.sh')

import os

roleArn = os.environ['BEDROCKAGENTROLE']


# ## Start of the lesson

# In[2]:


import boto3


# In[3]:


bedrock_agent = boto3.client(service_name='bedrock-agent', region_name='us-west-2')


# In[4]:


create_agent_response = bedrock_agent.create_agent(
    agentName='mugs-customer-support-agent',
    foundationModel='anthropic.claude-3-haiku-20240307-v1:0',
    instruction="""You are an advanced AI agent acting as a front line customer support agent.""",
    agentResourceRoleArn=roleArn
)


# In[5]:


create_agent_response


# In[6]:


agentId = create_agent_response['agent']['agentId']


# In[7]:


from helper import *


# In[8]:


wait_for_agent_status(
    agentId=agentId, 
    targetStatus='NOT_PREPARED'
)


# In[9]:


bedrock_agent.prepare_agent(
    agentId=agentId
)


# In[10]:


wait_for_agent_status(
    agentId=agentId, 
    targetStatus='PREPARED'
)


# In[11]:


create_agent_alias_response = bedrock_agent.create_agent_alias(
    agentId=agentId,
    agentAliasName='MyAgentAlias',
)

agentAliasId = create_agent_alias_response['agentAlias']['agentAliasId']

wait_for_agent_alias_status(
    agentId=agentId,
    agentAliasId=agentAliasId,
    targetStatus='PREPARED'
)


# In[12]:


bedrock_agent_runtime = boto3.client(service_name='bedrock-agent-runtime', region_name='us-west-2')


# In[13]:


import uuid


# In[14]:


message = "Hello, I bought a mug from your store yesterday, and it broke. I want to return it."

sessionId = str(uuid.uuid4())

invoke_agent_response = bedrock_agent_runtime.invoke_agent(
    agentId=agentId,
    agentAliasId=agentAliasId,
    inputText=message,
    sessionId=sessionId,
    endSession=False,
    enableTrace=True,
)


# In[15]:


event_stream = invoke_agent_response["completion"]


# In[16]:


for event in event_stream:
    print(event)


# In[17]:


message = "Hello, I bought a mug from your store yesterday, and it broke. I want to return it."

sessionId = str(uuid.uuid4())


# In[18]:


invoke_agent_and_print(
    agentAliasId=agentAliasId,
    agentId=agentId,
    sessionId=sessionId,
    inputText=message,
    enableTrace=True,
)


# In[ ]:




