# Introduction
This file contains scripts used to give instruction to a coding LLM.  Aider-chat is used, usualy with DeepSeek R1 as LLM.

## Port of SmolAgentss GUI from Gradio to Streamlit

####
/web https://raw.githubusercontent.com/huggingface/smolagents/refs/heads/main/src/smolagents/gradio_ui.py 
####
smolagents_streamlit.py is a port of gradio_ui.py  (from the smolgents package) to StreamLit.
Howerver, smolgents packages version has changed, and gradio_ui.py has changed accordingly. 
There are npotablu issues in the display of code.

I ask you to to report the changes in that files smolagents_streamlit.py.

Note what already been done for the port:
    Use the 'to_raw' method to use AgentImage in st.image().
    Replace relative import by absolute import from smolagents.
    Add st.expander when displaying code.
    Add type annotation whenever possible (hint: agent is of type smolagents.MultiStepAgent, images  is of type PIL.Image.Image). 
    Keep the licence text.


## Create a Pydantic Scheme from documents
#### 
/read-only  ...  
#### 
I want to extract structured information from Markdown file using an LLM with structured output. Analyse the files I've given, and select everything is relevant to understand and the project success or failure (for further data analytics)  and project similarity search   : project description, people involved and their role, delivery information (business line, location, warnings, ...) , risks, technologies used, competitors, partnersips, biding strategy, etc.  Also gather key metrics, such as TCV, nome of customer, references of the RFQ, etc.  These are juste example,  do your  own analyis. In second step, write a file with Pydantic classes representing that structured information.  Use  'Field(description=...)' to precisely describe what is extracted (by an LLM). Ignore Compliance Checks. Just create the file.  You don't need other information on existing code.  

#### 
/ask write a short sentences wrapping-up  the information  extracted by that Pydantic Model, to be be inserted in an LLM prompt  