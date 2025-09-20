#!/usr/bin/env python
# coding: utf-8

# # Lesson 2: Summarize an audio file

# ### Import all needed packages

# In[1]:


import os
from IPython.display import Audio
import boto3
import uuid
import time
import json
from jinja2 import Template


# ### Let's start with transcribing an audio file

# In[2]:


audio = Audio(filename="dialog.mp3")
display(audio)


# In[3]:


s3_client = boto3.client('s3', region_name='us-west-2')


# In[4]:


bucket_name = os.environ['BucketName']


# In[5]:


file_name = 'dialog.mp3'


# In[6]:


s3_client.upload_file(file_name, bucket_name, file_name)


# In[7]:


transcribe_client = boto3.client('transcribe', region_name='us-west-2')


# In[8]:


job_name = 'transcription-job-' + str(uuid.uuid4())


# In[9]:


job_name


# In[10]:


response = transcribe_client.start_transcription_job(
    TranscriptionJobName=job_name,
    Media={'MediaFileUri': f's3://{bucket_name}/{file_name}'},
    MediaFormat='mp3',
    LanguageCode='en-US',
    OutputBucketName=bucket_name,
    Settings={
        'ShowSpeakerLabels': True,
        'MaxSpeakerLabels': 2
    }
)


# In[12]:


print(response)


# In[11]:


while True:
    status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
    if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
        break
    time.sleep(2)
print(status['TranscriptionJob']['TranscriptionJobStatus'])


# In[13]:


if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
    
    # Load the transcript from S3.
    transcript_key = f"{job_name}.json"
    transcript_obj = s3_client.get_object(Bucket=bucket_name, Key=transcript_key)
    transcript_text = transcript_obj['Body'].read().decode('utf-8')
    transcript_json = json.loads(transcript_text)
    
    output_text = ""
    current_speaker = None
    
    items = transcript_json['results']['items']
    
    for item in items:
        
        speaker_label = item.get('speaker_label', None)
        content = item['alternatives'][0]['content']
        
        # Start the line with the speaker label:
        if speaker_label is not None and speaker_label != current_speaker:
            current_speaker = speaker_label
            output_text += f"\n{current_speaker}: "
            
        # Add the speech content:
        if item['type'] == 'punctuation':
            output_text = output_text.rstrip()
            
        output_text += f"{content} "
        
    # Save the transcript to a text file
    with open(f'{job_name}.txt', 'w') as f:
        f.write(output_text)


# ### Now, let's use an LLM

# In[14]:


bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')


# In[15]:


with open(f'{job_name}.txt', "r") as file:
    transcript = file.read()


# In[16]:


print(transcript)


# In[17]:


get_ipython().run_cell_magic('writefile', 'prompt_template.txt', '\nI need to summarize a conversation. The transcript of the \nconversation is between the <data> XML like tags.\n\n<data>\n{{transcript}}\n</data>\n\nThe summary must contain a one word sentiment analysis, and \na list of issues, problems or causes of friction\nduring the conversation. The output must be provided in \nJSON format shown in the following example. \n\nExample output:\n{\n    "sentiment": <sentiment>,\n    "issues": [\n        {\n            "topic": <topic>,\n            "summary": <issue_summary>,\n        }\n    ]\n}\n\nWrite the JSON output and nothing more.\n\nHere is the JSON output:\n')


# In[18]:


with open('prompt_template.txt', "r") as file:
    template_string = file.read()


# In[19]:


print(template_string)


# In[20]:


data = {
    'transcript' : transcript
}


# In[21]:


template = Template(template_string)


# In[22]:


prompt = template.render(data)


# In[23]:


print(prompt)


# In[29]:


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


# In[30]:


response = bedrock_runtime.invoke_model(**kwargs)


# In[31]:


response_body = json.loads(response.get('body').read())


# In[33]:


print(response_body)


# In[34]:


generation = response_body['results'][0]['outputText']


# In[35]:


print(generation)


# In[ ]:





# In[ ]:




