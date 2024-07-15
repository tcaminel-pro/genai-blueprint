# GenAI Training 


## Install
* Use 'poetry' to install and manage project
  * run 'poetry shell'
  * run 'poetry update'  

* Application settings are in file : app_conf.yaml ; Should likely be edited (and improved...)

* Run 'make test' to check it API works locally (ignore warnings)
* Run 'python python/main_cli.py echo "hello"  ' to check CLI
* Run 'make fast_api'  to launch FastAPI locally
  * Swagger API is testable from: http://localhost:8000/docs 



## Files
### Code
* fastapi_app.py : Entry point of FastAPI app.  API definitions are there
* langserve_app.py : Entry point for langserve
* GenAI_Lab : Entry point for Streamlit webapp
* main_cli.cpp : Entry point for the Command line interface (nice for dev.). 
  * example usage : poetry run python cmd 

See Makefile for examples
### Test Data
* 

# Create Docker
* Use 'make' to build Docker image, run it locally,..


## Various
