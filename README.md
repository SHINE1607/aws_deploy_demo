# aws_deploy_demo

Hackathon project hosted by ineuron during the year 2020.The project has fundamenatally 3 
machine learnining models based on XGboot. the project has cardio-vascular, dota game prediction winner, 
and house rent prediction. All models are trained and binded to a single UI sysytem,  and integrated using flask 
server and deployed on AWS EC2 instance.


# Project directory description
## Folders
* templates - Contains all the html files
* static -  Stores all the predicted files as csv
* source_code - contains the source pyhton script of all vertical models 
  * cardio - folder conatins directories for code and  project documentation
  * dota  - foilder containing directories for code and project  documentation
  * houserent - folder containg directories for code and project documentation
## Files
* app.py - python script containing flask server
* cluster_classifier - Kmeans classifier to classify the lat-long variabes to different clusters in houserent predictionn
* Preprocess.py - a single binded python sciprt for all machine models containing the preprocesing models
* xgbregressor_house - pre-trained model for predicting house rent
* xgbclassifier_cardio  - pre-trained model for predicting the cardiovascular disease 
* xgbclassiier_dota - pre-trained model for predicting the dota game winner

# Steps to host the application in local server
* open terminal and enter "pip install -r  requirements.txt"
* then enter "python app.py 
* ctrl + click on the localhost link showing ion the terminal 

### Public url link
* ec2-3-6-92-45.ap-south-1.compute.amazonaws.com:8080
* PS: will only be accessible only if hosted from the admin system


