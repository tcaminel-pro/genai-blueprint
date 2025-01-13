# GenAI Training 


## Install
* Use 'poetry' to install and manage project
  * run 'poetry shell'
  * run 'poetry update'  

* Application settings are in file : app_conf.yaml ; Should likely be edited (and improved...)

* Run 'make test' - But there some issues with several tests in //. You might need to change section 'pytest' app_conf.yaml too
* Run 'python python/main_cli.py echo "hello"  ' to check CLI
* Run 'python python/main_cli.py run joke  for a quick end-to-end test. add '--help' to see the different options



## Files
### Code
* fastapi_app.py : Entry point of FastAPI app.  API definitions are there
* langserve_app.py : Entry point for langserve
...

# Complete the README with a lost and description of the files  AI!