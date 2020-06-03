#python file to preprocess data
from sklearn.cluster import KMeans
import pandas as pd
import joblib 
def preprocess_cardio(df):

    try:
        df.drop(["id"], inplace = True, axis = 1)
    except:
        pass
    df["age"] = df["age"].div(365)
    df[df["gender"] == 2] = 0
    df["height"] = df["height"].div(100)
    return df

def preprocess_dota (df):
    # df[df[0] == -1] = 0
    df[1] = df[1].div(100)
    return df

def preprocess(df):
    kmeans = joblib.load("./cluster_classifier")
    #filling null values
    df["laundry_options"].fillna(df["laundry_options"].mode()[0], inplace = True)
    df["parking_options"].fillna(df["parking_options"].mode()[0], inplace = True)
    df["state"].fillna(df["state"].mode(), inplace = True)
    kmeans = joblib.load("cluster_classifier")
    lat = df[~df["lat"].isna()]["lat"]
    long = df[~df["long"].isna()]["long"]
    lat_long = pd.DataFrame()
    lat_long["lat"] = lat
    lat_long["long"] = long
    
    labels = kmeans.predict(lat_long[lat_long.columns[0:2]])
    df["cluster"] = labels
    df.drop(["id", "url", "image_url","region", "description", "region_url","state"], inplace = True, axis = 1)   
    
    #ordinal data
    df["laundry_encoded"] = [0]*df.shape[0]
    df["parking_encoded"] = [0]*df.shape[0]
    for i, el in enumerate(df["laundry_options"].unique()[::-1]):
        df[df["laundry_options"] == el]["laundry_options"]= i
    arr = ["no_parking", "off_street_parking", "street parking", "valetparking", "detached garage", "carport", "attached garage"]
    for i, el in enumerate(arr):
        df[df["parking_options"] == el]["parking_options"]= i
    df.drop(["laundry_options", "parking_options"], inplace = True, axis = 1)
    #dummy values for nominal data
    dummy_cols = ['type_assisted living', 'type_condo', 'type_cottage/cabin',
       'type_duplex', 'type_flat', 'type_house', 'type_in-law', 'type_land',
       'type_loft', 'type_manufactured', 'type_townhouse']
    arr = [0]*len(dummy_cols)
    for i in range(len(dummy_cols)):
        if df.loc[0]["type"] in dummy_cols[i]:
            arr[i] = i
    df_dummy = pd.DataFrame(data =  [arr], columns = dummy_cols)
    df.drop(["type", "lat", "long"], axis = 1, inplace = True)
    df = df.join(df_dummy)    
    return df




