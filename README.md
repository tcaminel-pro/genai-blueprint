# GenAI Training 


## Install
* Use 'poetry' to install and manage project
  * run 'poetry shell'
  * run 'poetry update'  

* Application settings are in file : app_conf.toml ; Should likely be edited (and improved...)

* Run 'make test' to check it API works locally (ignore warnings)
* Run 'python python/main_cli.py echo "hello"  ' to check CLI
* Run 'make fast_api'  to launch FastAPI locally
  * Swagger API is testable from: http://localhost:8000/docs 


# Create Docker
* Use 'make' to build Docker image, run it locally,..

## Files
### Code
* fastapi_app.cpp : Entry point of FastAPI app.  API definitions are there
* main_cli.cpp : Entry point for the Command line interface (nice for dev.). 
  * example usage : poetry run python cmd 

### Test Data
* 

## Various