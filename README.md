FAST API DEMONSTATION

steps to install

 using Docker:
    build the docker image
        docker build . -t [docker-image-name] -f Dockerfile .
    run docker image
        docker run [docker-image-name] -p8000:8000

folder structure
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

 
env variables
    if running code locally , set ENV variables manually
        MODEL_DIR_PATH : path to hte model_files folder
        ROOT_FOLDER_PATH : path to the folder containing DockerFile
