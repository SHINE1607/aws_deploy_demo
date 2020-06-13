# aws_deploy_demo

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

