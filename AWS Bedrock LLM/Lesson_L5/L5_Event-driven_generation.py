#!/usr/bin/env python
# coding: utf-8

# # Lesson 5: Event-driven generation

# ### Import all needed packages

# In[1]:


import boto3, os

from helpers.Lambda_Helper import Lambda_Helper
from helpers.S3_Helper import S3_Helper

lambda_helper = Lambda_Helper()
s3_helper = S3_Helper()

bucket_name_text = os.environ['LEARNERS3BUCKETNAMETEXT']
bucket_name_audio = os.environ['LEARNERS3BUCKETNAMEAUDIO']


# ### Deploy your lambda function

# In[2]:


get_ipython().run_cell_magic('writefile', 'lambda_function.py', '\n#############################################################\n#\n# This Lambda function is written to a file by the notebook \n# It does not run in the notebook!\n#\n#############################################################\n\nimport json\nimport boto3\nimport uuid\nimport os\n\ns3_client = boto3.client(\'s3\')\ntranscribe_client = boto3.client(\'transcribe\', region_name=\'us-west-2\')\n\ndef lambda_handler(event, context):\n    # Extract the bucket name and key from the incoming event\n    bucket = event[\'Records\'][0][\'s3\'][\'bucket\'][\'name\']\n    key = event[\'Records\'][0][\'s3\'][\'object\'][\'key\']\n\n    # One of a few different checks to ensure we don\'t end up in a recursive loop.\n    if key != "dialog.mp3": \n        print("This demo only works with dialog.mp3.")\n        return\n\n    try:\n        \n        job_name = \'transcription-job-\' + str(uuid.uuid4()) # Needs to be a unique name\n\n        response = transcribe_client.start_transcription_job(\n            TranscriptionJobName=job_name,\n            Media={\'MediaFileUri\': f\'s3://{bucket}/{key}\'},\n            MediaFormat=\'mp3\',\n            LanguageCode=\'en-US\',\n            OutputBucketName= os.environ[\'S3BUCKETNAMETEXT\'],  # specify the output bucket\n            OutputKey=f\'{job_name}-transcript.json\',\n            Settings={\n                \'ShowSpeakerLabels\': True,\n                \'MaxSpeakerLabels\': 2\n            }\n        )\n        \n    except Exception as e:\n        print(f"Error occurred: {e}")\n        return {\n            \'statusCode\': 500,\n            \'body\': json.dumps(f"Error occurred: {e}")\n        }\n\n    return {\n        \'statusCode\': 200,\n        \'body\': json.dumps(f"Submitted transcription job for {key} from bucket {bucket}.")\n    }\n\n')


# In[3]:


lambda_helper.lambda_environ_variables = {'S3BUCKETNAMETEXT' : bucket_name_text}
lambda_helper.deploy_function(["lambda_function.py"], function_name="LambdaFunctionTranscribe")


# In[4]:


lambda_helper.filter_rules_suffix = "mp3"
lambda_helper.add_lambda_trigger(bucket_name_audio, function_name="LambdaFunctionTranscribe")


# In[5]:


s3_helper.upload_file(bucket_name_audio, 'dialog.mp3')


# In[6]:


s3_helper.list_objects(bucket_name_audio)


# #### Restart kernel if needed.
# - If you run the code fairly quickly from start to finish, it's possible that the following code cell `s3_helper.list_objects(bucket_name_text)` will give a "Not Found" error.  
# - If waiting a few seconds (10 seconds) and re-running this cell does not resolve the error, then you can restart the kernel of the jupyter notebook.
# - Go to menu->Kernel->Restart Kernel.
# - Then run the code cells from the start of the notebook, waiting 2 seconds or so for each code cell to finish executing.

# In[9]:


s3_helper.list_objects(bucket_name_text)


# #### Re-run "download" code cell as needed
# - It may take a few seconds for the results to be generated.
# - If you see a `Not Found` error, please wait a few seconds and then try running the `s3_helper.download_object` again.

# In[10]:


s3_helper.download_object(bucket_name_text, 'results.txt')


# In[11]:


from helpers.Display_Helper import Display_Helper


# In[12]:


display_helper = Display_Helper()


# In[13]:


display_helper.text_file('results.txt')


# Extra resources:
# 
# * [Generative AI code](https://community.aws/code/generative-ai)
# 
# * [Generative AI](https://community.aws/generative-ai)
# 
