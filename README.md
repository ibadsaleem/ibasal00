Coding Challenge Demonstration

## Steps to install

 ### Using Docker:
    build the docker image
        docker build . -t [docker-image-name] -f Dockerfile .
    run docker image
        docker run [docker-image-name] -p8000:8000

### Folder structure
    /app - contains REST API code
        /models - contains DTOs for request and reponse
        /routers - contains logic for each route
        /test - contains test cases
        /utils - contains helper functions
        dependencies.py - in case of auth, can be configured to provide middleware support
        main.py - application startup file

    /training - contains the training code
        /config - contains configutration
        /data - contains dataset
        /model_files - contains model output files
        /preprocess_data - cleaning data scripts before training
        /utils - helper functions for training e.g (model)
        train.py - training script
        test.py - testing script
    /requirements - contains requiremnts file accroding to different environemnts

 
### Env variables
    if running code locally , set ENV variables manually
        MODEL_DIR_PATH : path to hte model_files folder
        ROOT_FOLDER_PATH : path to the folder containing DockerFile

# Way Solving this Problem:
 <p>The Problem is solved using GPT Model 2 and connected to API using Flask (Python). It currently gives the accuracy of around 70% and with more fine tuning of hyper parameters the accuracy can be improved. Above mentioned steps provides the way running the code to run FAST API</p>
