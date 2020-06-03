import numpy as np
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for, send_file, send_from_directory
import joblib
import pandas as pd 
from preprocess import *
import io
import csv
from csv import reader

import threading 

UPLOAD_FOLDER = '/data'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
#counters for file save
c = 0
d = 0
h = 0

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/cardio")
def cardio():
    return render_template("cardio.html")


@app.route("/dota")
def dota():
    return render_template("dota.html")

    
@app.route("/house")
def house():
    return render_template("house_rent.html")
df_pred = pd.DataFrame()


@app.route('/cardio/predict', methods=[ 'POST'])
def predict_cardio():
    '''
    For rendering results on HTML GUI
    '''
    global c
    #loading the model
    model = joblib.load("xgbclassifier_cardio")
    #columns to  beaaded in the dataframe
    cols = ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
       'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
    cols.remove("id")
    cols.remove("cardio")
    
    #checking the input file is file or form values
    if "file" in  request.files:

        arr_pred = []
        print("the input is csv file")
        # taking file from the uploaded 
        file = request.files["file"]
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        #converting the uploaded file csv reader file
        csv_input = csv.reader(stream)
        m = 0
        #sending each row for prediction from csv
        for row in csv_input:
            if (m > 0):
                row = row[0].split(";")
                row = [(float(x)) for x in row]
                if(len(row) > 12):
                    row = row[1:12]
                
                df = pd.DataFrame(data = [row], columns = cols )
                df = preprocess_cardio(df.copy())   
                final_features = df.to_numpy()
                #storing the prediction and appending it to arr_pred
                prediction = model.predict((final_features))[0]
                arr_pred.append(prediction)


            m = 1
        df_pred = pd.DataFrame(data = arr_pred, columns = ["prediction"])
        df_pred.to_csv( "./predictions/cardio_prediction{}.csv".format(c))
        #case e=when the input is file 
        c= c + 1
        path = "./predictions/cardio_prediction{}.csv".format(c-1)
        
        return send_file('predictions/cardio_prediction{}.csv'.format(c-1),
                     mimetype='text/csv',
                     attachment_filename='cardio_prediction{}.csv'.format(c-1),
                     as_attachment=True)
    else:
        print("the input is form values")
        #case when input is form values 
        arr = []
        for i in request.form.values():
            arr.append(float(i))
        print(arr)
        #converting input form values to array
        int_features = [(float(x)) for x in request.form.values()]
        df = pd.DataFrame(data = [int_features], columns = cols )
        print(df, "is the dataframe")
        df = preprocess_cardio(df.copy())
        # converting the prepocessed dataframe to numpy array for predictionn 
        final_features = df.to_numpy()
        print(final_features)
        prediction = model.predict((final_features))[0]
        output = " " if(prediction == 1) else "Not"
        print(output , "is the output ")
        if(prediction == 1):
            return render_template('cardio.html', prediction_text_safe ='You are diagonised with Cardio disese')
        else:
            return render_template("cardio.html", prediction_text_safe = "You are safe!!")
        # check if the post request has the file part
    
    return render_template("cardio.html")
    
# if __name__ == "__main__":
#     app.run(host = "0.0.0.0", port = 8080)


@app.route('/dota/predict',methods=['POST'])
def predict_dota():
    '''
    For rendering results on HTML GUI
    '''
    #loadin the model

    global d 
    model = joblib.load("xgbclassifier_dota")
    def checkIfDuplicates_1(listOfElems):
        ''' Check if given list contains any duplicates '''
        if len(listOfElems) == len(set(listOfElems)):
            return False
        else:
            return True


    cols = [x for x in range(0, 116)]
    #condition for file upload
    if "file" in  request.files:
        print("uploaded a csv file")
        arr_pred = []
        # taking file from the uploaded 
        file = request.files["file"]
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        #converting the uploaded file csv reader file
        csv_input = csv.reader(stream)
        m = 0   #flag value for dropping header
        #sending each row for prediction from csv
        for row in csv_input:
            if (m > 0):
                
                row = [(float(x)) for x in row]
                row = row[-116:]
                if(checkIfDuplicates_1(row[3:]) == True):
                    return render_template('dota.html', prediction_text_safe='A hero can be selected by one player, heroes cannot clone!!')
                
                df = pd.DataFrame(data = [row], columns = cols )
                df = preprocess_dota(df.copy())   
                final_features = df.to_numpy()
                #storing the prediction and appending it to arr_pred
                prediction = model.predict((final_features))[0]
                arr_pred.append(prediction)


            m = 1
        df_pred = pd.DataFrame(data = arr_pred, columns = ["prediction"])
        df_pred.to_csv( "./predictions/dota_prediction{}.csv".format(d))
        #case e=when the input is file 
        d= d + 1
        path = "./predictions/dota_prediction{}.csv".format(d-1)
        
        return send_file('predictions/dota_prediction{}.csv'.format(d-1),
                     mimetype='text/csv',
                     attachment_filename='dota_prediction{}.csv'.format(c-1),
                     as_attachment=True)
    else:
        print("uploaded form values ")
        int_features = [int(float(x)) for x in request.form.values()]
        print(int_features)
        if(checkIfDuplicates_1(int_features[3:]) == True):
            return render_template('dota.html', prediction_text_safe='A hero can be selected by one player, heroes cannot clone!!')
        
        #list to map 113 heroes with players
        player_map = [0]*113
        for i in range(4, len(int_features)-1):
            player_map[int_features[i]-1] = 1
            player_map[int_features[i+1]-1] = -1
        print(player_map)
        final_data = int_features[0:3] + player_map
        df = pd.DataFrame(data = [final_data] , columns = cols)
        print(df.shape)
        df = preprocess_dota(df.copy())
        final_features = df.to_numpy()
        prediction = model.predict((final_features))[0]
        print(prediction, "is the prediction")
        return render_template("dota.html", prediction_text_safe = "Team {} won the game!!!".format(prediction))
   

    
@app.route('/house/predict',methods=['POST'])
def predict_house():
    '''
    For rendering results on HTML GUI
    '''
    #loadin the model
    model = joblib.load("regressor_house")
    global h

    cols = ['id', 'url', 'region', 'region_url', 'type', 'sqfeet', 'beds',
       'baths', 'cats_allowed', 'dogs_allowed', 'smoking_allowed',
       'wheelchair_access', 'electric_vehicle_charge', 'comes_furnished',
       'laundry_options', 'parking_options', 'image_url', 'description', 'lat',
       'long', 'state']

    #condition if the upload csv
    if "file" in  request.files:
        print("uploaded a csv file")
        arr_pred = []
        # taking file from the uploaded 
        file = request.files["file"]
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        #converting the uploaded file csv reader file
        csv_input = csv.reader(stream)
        m = 0   #flag value for dropping header
        #sending each row for prediction from csv
        for j, row in enumerate(csv_input):
            if(j <= 2000):
                if (m >= 1):
                    for i in range(len(row)):
                        try:
                            row[i] = float(row[i])
                        except :
                            pass

                    df = pd.DataFrame(data = [row], columns = cols )
                    try:
                        df = preprocess(df.copy()) 
                        # print(type(df.iloc[0]["sqfeet"]))  
                        prediction = model.predict(df)[0]  
                        # # # #storing the prediction and appending it to arr_pred
                        arr_pred.append(np.exp(prediction) - 1)
                    except:
                        arr_pred.append("{}th row has corrupted data".format(j))
                
            m = m + 1

        #     print("GGGGGGGGGGGGGGGGGG")
        #     print(m)
        # # return render_template("house_rent.html")
        df_pred = pd.DataFrame(data = arr_pred, columns = ["HousePrice"])
        df_pred.to_csv( "./predictions/house_prediction{}.csv".format(h))
        print(df_pred)
        h  = h + 1 
        path = "./predictions/house_prediction{}.csv".format(h-1)
        
        return send_file('predictions/house_prediction{}.csv'.format(h-1),
                     mimetype='text/csv',
                     attachment_filename='house_prediction{}.csv'.format(h-1),
                     as_attachment=True)
    #condition for form value uploaded
    else:
        int_features = []
        for i in request.form.values():
            try:
                int_features.append(float(i))
        
            except:
                int_features.append(i)
            
        



        df = pd.DataFrame(data = [int_features], columns = cols)
        df = preprocess(df)

        prediction = model.predict(df)[0]
        # print(prediction, "is the prediction")
        print(prediction )
        return render_template("house_rent.html", prediction_text_safe = "The predicted house price is ${}".format(np.exp(prediction) - 1))


@app.route("/cardio/cardio_readme")
def route_cardio_readme():
    return render_template("cardio_readme.html")

@app.route("/dota/dota_readme")
def route_dota_readme():
    return render_template("dota_readme.html")

@app.route("/house/house_readme")
def route_house_readme():
    return render_template("house_readme.html")

if __name__ == "__main__":
    app.secret_key = 'ineuron_secret_key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug = True)
   
# if __name__ == "__main__":
#     app.secret_key = 'ineuron_secret_key'
#     app.config['SESSION_TYPE'] = 'filesystem'
#     app.run(host = "0.0.0.0", port = 8080)