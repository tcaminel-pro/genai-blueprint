# GenAI Training 


## Install
* Use 'poetry' to install and manage project
** if Poert not intalled : run 'poetry shell'
* run 'poetry update'  
* Run 'pytest' to check it works  (ignore warnings)
* Use 'make' to build image, run it locally, run Python directly, etc.  See Makefile for details
* Application settings are in file : app_conf.toml ; Should likely be editet (and improved...)
* When running localy, Swagger API is testable from: http://localhost:8000/docs 

## Files
### Code
* main_api.cpp : Entry point of FastAPI app.  API definitions are there
* main_cli.cpp : Entry point for the Command line interface (nice for dev.). 
  * exemple usage : poetry run python cmd 

### Test Data
* 

## Various