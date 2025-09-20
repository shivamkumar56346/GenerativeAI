#!/usr/bin/env python
# coding: utf-8

# # Lesson 4: Deploy an AWS Lambda function

# ### Import all needed packages

# In[1]:


import boto3, os


# In[2]:


from helpers.Lambda_Helper import Lambda_Helper
from helpers.S3_Helper import S3_Helper
from helpers.Display_Helper import Display_Helper


# In[3]:


lambda_helper = Lambda_Helper()
# deploy_function
# add_lambda_trigger


# In[4]:


s3_helper = S3_Helper()
# upload_file
# download_object 
# list_objects


# In[5]:


display_helper = Display_Helper()
# text_file
# json_file


# In[6]:


bucket_name_text = os.environ['LEARNERS3BUCKETNAMETEXT']


# In[7]:


get_ipython().run_cell_magic('writefile', 'prompt_template.txt', '\nI need to summarize a conversation. The transcript of the conversation is between the <data> XML like tags.\n\n<data>\n{{transcript}}\n</data>\n\nThe summary must contain a one word sentiment analysis, and a list of issues, problems or causes of friction\nduring the conversation. The output must be provided in JSON format shown in the following example. \n\nExample output:\n{\n    "version": 0.1,\n    "sentiment": <sentiment>,\n    "issues": [\n        {\n            "topic": <topic>,\n            "summary": <issue_summary>,\n        }\n    ]\n}\n\nAn `issue_summary` must only be one of:\n{%- for topic in topics %}\n - `{{topic}}`\n{% endfor %}\n\nWrite the JSON output and nothing more.\n\nHere is the JSON output:\n')


# In[8]:


display_helper.text_file('prompt_template.txt')


# ### Create the Lambda function

# In[9]:


get_ipython().run_cell_magic('writefile', 'lambda_function.py', '\n\n#############################################################\n#\n# This Lambda function is written to a file by the notebook \n# It does not run in the notebook!\n#\n#############################################################\n\nimport boto3\nimport json \nfrom jinja2 import Template\n\ns3_client = boto3.client(\'s3\')\nbedrock_runtime = boto3.client(\'bedrock-runtime\', \'us-west-2\')\n\ndef lambda_handler(event, context):\n    \n    bucket = event[\'Records\'][0][\'s3\'][\'bucket\'][\'name\']\n    key = event[\'Records\'][0][\'s3\'][\'object\'][\'key\']\n    \n    # One of a few different checks to ensure we don\'t end up in a recursive loop.\n    if "-transcript.json" not in key: \n        print("This demo only works with *-transcript.json.")\n        return\n    \n    try: \n        file_content = ""\n        \n        response = s3_client.get_object(Bucket=bucket, Key=key)\n        \n        file_content = response[\'Body\'].read().decode(\'utf-8\')\n        \n        transcript = extract_transcript_from_textract(file_content)\n\n        print(f"Successfully read file {key} from bucket {bucket}.")\n\n        print(f"Transcript: {transcript}")\n        \n        summary = bedrock_summarisation(transcript)\n        \n        s3_client.put_object(\n            Bucket=bucket,\n            Key=\'results.txt\',\n            Body=summary,\n            ContentType=\'text/plain\'\n        )\n        \n    except Exception as e:\n        print(f"Error occurred: {e}")\n        return {\n            \'statusCode\': 500,\n            \'body\': json.dumps(f"Error occurred: {e}")\n        }\n\n    return {\n        \'statusCode\': 200,\n        \'body\': json.dumps(f"Successfully summarized {key} from bucket {bucket}. Summary: {summary}")\n    }\n        \n        \n        \ndef extract_transcript_from_textract(file_content):\n\n    transcript_json = json.loads(file_content)\n\n    output_text = ""\n    current_speaker = None\n\n    items = transcript_json[\'results\'][\'items\']\n\n    # Iterate through the content word by word:\n    for item in items:\n        speaker_label = item.get(\'speaker_label\', None)\n        content = item[\'alternatives\'][0][\'content\']\n        \n        # Start the line with the speaker label:\n        if speaker_label is not None and speaker_label != current_speaker:\n            current_speaker = speaker_label\n            output_text += f"\\n{current_speaker}: "\n        \n        # Add the speech content:\n        if item[\'type\'] == \'punctuation\':\n            output_text = output_text.rstrip()  # Remove the last space\n        \n        output_text += f"{content} "\n        \n    return output_text\n        \n\ndef bedrock_summarisation(transcript):\n    \n    with open(\'prompt_template.txt\', "r") as file:\n        template_string = file.read()\n\n    data = {\n        \'transcript\': transcript,\n        \'topics\': [\'charges\', \'location\', \'availability\']\n    }\n    \n    template = Template(template_string)\n    prompt = template.render(data)\n    \n    print(prompt)\n    \n    kwargs = {\n        "modelId": "amazon.titan-text-express-v1",\n        "contentType": "application/json",\n        "accept": "*/*",\n        "body": json.dumps(\n            {\n                "inputText": prompt,\n                "textGenerationConfig": {\n                    "maxTokenCount": 2048,\n                    "stopSequences": [],\n                    "temperature": 0,\n                    "topP": 0.9\n                }\n            }\n        )\n    }\n    \n    response = bedrock_runtime.invoke_model(**kwargs)\n\n    summary = json.loads(response.get(\'body\').read()).get(\'results\')[0].get(\'outputText\')    \n    return summary\n    \n    \n')


# In[10]:


lambda_helper.deploy_function(
    ["lambda_function.py", "prompt_template.txt"],
    function_name="LambdaFunctionSummarize"
)


# In[11]:


lambda_helper.filter_rules_suffix = "json"
lambda_helper.add_lambda_trigger(bucket_name_text)


# In[ ]:


# display_helper.json_file('demo-transcript.json')


# In[12]:


s3_helper.upload_file(bucket_name_text, 'demo-transcript.json')


# #### Restart kernel if needed.
# - If you run the code fairly quickly from start to finish, it's possible that the following code cell `s3_helper.list_objects(bucket_name_text)` will give a "Not Found" error.  
# - If waiting a few seconds (10 seconds) and re-running this cell does not resolve the error, then you can restart the kernel of the jupyter notebook.
# - Go to menu->Kernel->Restart Kernel.
# - Then run the code cells from the start of the notebook, waiting 2 seconds or so for each code cell to finish executing.

# In[13]:


s3_helper.list_objects(bucket_name_text)


# #### Re-run "download" code cell as needed
# - It may take a few seconds for the results to be generated.
# - If you see a `Not Found` error, please wait a few seconds and then try running the `s3_helper.download_object` again.

# In[14]:


s3_helper.download_object(bucket_name_text, "results.txt")


# In[15]:


display_helper.text_file('results.txt')


# In[ ]:





# In[ ]:




