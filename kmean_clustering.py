from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import shutil
import argparse

# Function to Extract features from the images
def kmean(args):
    input_path = args.input_folder_path
    img_list= os.listdir(input_path)
    output_path = args.output_folder_path
    
    model = InceptionV3(weights='imagenet', include_top=False)
    features = [];
    img_name = [];
    for i in tqdm(img_list):
        fname=os.path.join(input_path,i)
        img=image.load_img(fname,target_size=(224,224))
        x = img_to_array(img)
        print(x)
        x=np.expand_dims(x,axis=0)
        print(x)
        x=preprocess_input(x)
        print(x)
        feat=model.predict(x)
        feat=feat.flatten()
        features.append(feat)
        img_name.append(i)


    image_cluster = pd.DataFrame(img_name,columns=['image'])
    print(image_cluster)

    #Creating Clusters
    k = args.number_mean_points
    clusters = KMeans(k, random_state = 40)
    clusters.fit(features)

    # KMeans(n_clusters=2, random_state=40)
    image_cluster["clusterid"] = clusters.labels_ # To mention which image belong to which cluster
    print(image_cluster) 

    # Images will be seperated according to cluster they belong
    for j in range(k):
        for i in range(len(image_cluster)):
            if image_cluster['clusterid'][i]==j:
                cluster_path = os.path.join(output_path,'cluster_'+str(j))
                os.makedirs(cluster_path,exist_ok=True)
                shutil.move(os.path.join(input_path, image_cluster['image'][i]), cluster_path)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Kmaen images classiffications')
    parser.add_argument('--input_folder_path', action = 'store', type = str, required = True, help = 'Input folder path where all images present')
    parser.add_argument('--output_folder_path', action = 'store', type = str, required = True, help = 'Output folder path where all clustered images stored')
    parser.add_argument('--number_mean_points', action = 'store', type = int, required = False, default = 224, help = 'Enter of number kmaen points')
    args = parser.parse_args()

    kmean(args)