
pip install tensorflow

!pip install opencv-python

import pandas as pd
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt

from glob import glob

import shutil
import tensorflow as tf
from skimage.io import imread
from google.colab import drive
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

drive.mount('/content/drive', force_remount=True)

path = '/content/drive/MyDrive/Sentiment classifier'

# get the folders inside the path
os.listdir(os.path.join(path))

"""# Splitting in folders"""

import random
random.seed(42)
happy_images = glob(os.path.join(path,'happy','/*'))
random.shuffle(happy_images)
sad_images = glob(os.path.join(path,'sad','/*'))
random.shuffle(sad_images)

"""Splitting data into train and test"""

len_train = int(0.7*(len(happy_images) + len(sad_images)))
len_validation = int(0.15*(len(happy_images) + len(sad_images)))
len_test = int(0.15*(len(happy_images) + len(sad_images)))
total_len = len(happy_images) + len(sad_images)

import math
def train_validation_test(classes_names, classes_images, len_train, len_validation, len_test, total_len, path):
    l_l = [len_train, len_test, len_validation]
    p,a = 0,0
    for folder_name, len_list in zip(['train', 'test', 'validation'], [len_train, len_test, len_validation]):
        for class_name, images in zip(classes_names, classes_images):
            os.makedirs(os.path.join(path, 'Images_split', folder_name, class_name), exist_ok=True)

            start_index = int(len(images) / total_len * (sum(l_l[:a])))
            end_index = int(len(images) / total_len * (sum(l_l[:a+1])))

            for i in range(start_index, end_index):
                if os.path.exists(images[i]):
                    dest_path = os.path.join(path, 'Images_split', folder_name, class_name, os.path.basename(images[i]))
                    shutil.copyfile(images[i], dest_path)
                else:
                    print(f"File not found: {images[i]}")
            p += 1
            a = math.floor(p/2)
            print(f'p:{p},a:{a}')

train_validation_test(['happy','sad'],[happy_images,sad_images],len_train,len_validation,len_test,total_len,path)

"""# Citire imagini"""

images_training = tf.keras.utils.image_dataset_from_directory(os.path.join(path,'Images_split','train'), label_mode='binary', seed = 42)
images_validation = tf.keras.utils.image_dataset_from_directory(os.path.join(path,'Images_split','validation'), label_mode='binary', seed = 42)
images_testing = tf.keras.utils.image_dataset_from_directory(os.path.join(path,'Images_split','test'), label_mode='binary', seed = 42)

# normalization of images
X_training = images_training.map(lambda x, y: x/255)
y_training = images_training.map(lambda x, y: y)
X_validation = images_validation.map(lambda x, y: x/255)
y_validation =  images_validation.map(lambda x, y: y)
X_testing = images_testing.map(lambda x, y: x/255)
y_testing = images_testing.map(lambda x, y: y)

batch = images_training.as_numpy_iterator().next()

batch[0].shape

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(8,4))
ax1.imshow(batch[0][0].astype("int16"))
ax2.imshow(batch[0][28].astype("int16"))
fig.suptitle(f'Labels are: left {int(batch[1][0])}, right {int(batch[1][28])}')
plt.show()

"""# Type 1 feature: pretrained VGG model"""

images = tf.keras.utils.image_dataset_from_directory('/content/drive/MyDrive/Sentiment classifier_data', label_mode='binary', seed = 42)

X_img = images.map(lambda x,y:x/255)
y_img = images.map(lambda x,y:y)

def get_y(y_tensor):
  y_labels = []
  for y in y_tensor:
    y_labels.append(y)
  return np.vstack(y_labels)

y_labels = get_y(y_img)

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

# include_top=False to exclude the fully connected layers at the top.
vgg_model = VGG16(input_shape = (256, 256, 3), include_top = False, weights = 'imagenet', pooling='max')
vgg_model.trainable = False

vgg_model.summary()

features_train,features_validation,features_test = vgg_model.predict(X_training),vgg_model.predict(X_validation),vgg_model.predict(X_testing)
features_data = np.concatenate((features_train,features_validation),axis=0)
y_train = get_y(y_training)
y_valid = get_y(y_validation)
y_test = get_y(y_testing)
y_data = np.concatenate((y_train,y_valid),axis=0)

"""Random Forest supervised learning"""

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100,random_state = 42,class_weight = 'balanced')
rf.fit(features_train, y_train)
y_pred = rf.predict(features_validation)

print(classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
rf_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight':['balanced']
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=rf_grid, cv=5, scoring='recall')
grid_search.fit(features_data, np.squeeze(y_data))

print('Best parametrs:', grid_search.best_params_)

best_model = grid_search.best_estimator_
best_model.fit(features_data, y_data)
y_pred = best_model.predict(features_test)

print(classification_report(np.squeeze(get_y(y_test)), y_pred))

r = recall_score(y_test,y_pred,average=None)
f = f1_score(y_test,y_pred,average=None)
p = precision_score(y_test,y_pred,average=None)
original_dict = {'recall':r,'f1_score':f,'precision':p}

fig, ax = plt.subplots(figsize=(8,4))
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
cmd.plot(values_format='d',ax=ax)
ax.set_title('Confusion Matrix for VGG features extractor')
plt.show()

"""# SMOTE algorithm for the imbalanced data problem"""

from collections import Counter
from imblearn.over_sampling import SMOTE
count_l = Counter(np.squeeze(y_data))
smote = SMOTE()
features_smote, y_smote = smote.fit_resample(features_data,y_data)
count_smote_l = Counter(np.squeeze(y_smote))
print(count_l,count_smote_l)

# Random Forest with the updated dataset oversampled with SMOTE
best_model.fit(features_smote, y_smote)
y_pred = best_model.predict(features_test)

print(classification_report(y_test, y_pred))

fig, ax = plt.subplots(figsize=(8,4))
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
cmd.plot(values_format='d',ax=ax)
ax.set_title('Confusion Matrix for VGG features extractor with SMOTE technique')
plt.show()

r = recall_score(y_test,y_pred,average=None)
f = f1_score(y_test,y_pred,average=None)
p = precision_score(y_test,y_pred,average=None)
smote_dict = {'recall':r,'f1_score':f,'precision':p}

original_dict

smote_dict

new_dict = {}
for key,value in original_dict.items():
  if key in smote_dict:
    new_dict[key] = np.concatenate((value[1:], smote_dict[key][1:]))
new_dict

score_df_smote = pd.DataFrame(new_dict)
score_df_smote['resample'] = ['original','smote']

score_df_smote

sns.set(font_scale=1.2)
#g = sns.FacetGrid(score_df_smote, height=5)
sns.barplot(x="resample", y="recall", data=score_df_smote, palette='viridis', order=["original", "smote"])
plt.xticks(rotation=30)

plt.xlabel(' ')
plt.ylabel('Recall', fontsize=14)

plt.title('Recall of the minority class', fontsize=16)

plt.show()

"""# Dummy Classifier"""

# random classifier
from sklearn.dummy import DummyClassifier
dc = DummyClassifier(strategy ='stratified')
dc.fit(features_data, y_data)
y_pred = dc.predict(features_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm)

cmd.plot(values_format='d')

"""# Agglomerative clustering"""

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import KNeighborsClassifier

agg_cluster = AgglomerativeClustering(n_clusters=2)
agg_cluster = agg_cluster.fit_predict(features_smote)

knn = KNeighborsClassifier()
knn.fit(features_smote,np.squeeze(agg_cluster))
y_pred = knn.predict(features_test)

silhouette_avg = silhouette_score(features_smote, agg_cluster)
silhouette_avg

len(y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

"""Hyperparameter tuning"""

def find_best_parameters(features, param):
    max_score = 0
    best_parameters = None
    score = []

    for n_clusters in param:
      agg_cluster = AgglomerativeClustering(n_clusters=n_clusters)
      c_labels = agg_cluster.fit_predict(features)

      silhouette_score1 = silhouette_score(features, c_labels)
      print(f'N_Clusters: {n_clusters}, silhouette score= {silhouette_score1}')
      score.append(silhouette_score1)

      if silhouette_score1 > max_score:
          max_score = silhouette_score1
          best_parameters = n_clusters

    return best_parameters, max_score,score

from sklearn.metrics import silhouette_score
params = [2, 3, 4, 5, 6, 7, 8, 9, 10]
param_vgg,best_score_vgg,score_vgg = find_best_parameters(features_smote,params)

"""# HDBScan"""

!pip install hdbscan

import hdbscan
def find_best_parameters_hdbscan(features, params):
    max_score = 0
    best_min_cluster_size = None
    score = []

    for param in params:
      hdb = hdbscan.HDBSCAN(min_cluster_size=param)

      c_labels = hdb.fit_predict(features)

      silhouette_score1 = silhouette_score(features, c_labels)
      print(f"min_cluster_size: {param}; silhouette score= {silhouette_score1}")
      score.append(silhouette_score1)

      if silhouette_score1 > max_score:
          max_score = silhouette_score1
          best_min_cluster_size = param
    return best_params, max_score, score

params = [3,5,10]
best_params, best_score, score = find_best_parameters_hdbscan(features_smote,params)

l_hdb = hdbscan.HDBSCAN(min_cluster_size=3).fit_predict(features_data)
unique_labels, label_counts = np.unique(l_hdb, return_counts=True)
for label, count in zip(unique_labels, label_counts):
    print(f"Label {label}: {count} occurrences")

mask = l_hdb != -1
features_filtered = features_data[mask]
labels_filtered = l_hdb[mask]

silhouette_score(features_data, l_hdb)

knn = KNeighborsClassifier()
knn.fit(features_filtered,np.squeeze(labels_filtered))
y_pred = knn.predict(features_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

"""# Type 2 : HOG feature extraction"""

from skimage import io
from sklearn.utils import shuffle
from skimage import io, transform,color,feature
from skimage.feature import hog

def load_images_and_labels(directory, label):
  images = []
  labels = []

  for filename in os.listdir(directory):
    image_path = os.path.join(directory, filename)
    image = io.imread(image_path)
    resized_image = transform.resize(image, (200,200))
    if len(image.shape) == 3:
      images.append(resized_image)
      labels.append(label)

  return images, labels

def shuffle_images(sad_directory,happy_directory):
  sad_images, sad_labels = load_images_and_labels(sad_directory, label=1)
  happy_images, happy_labels = load_images_and_labels(happy_directory, label=0)
  all_images = np.array(sad_images + happy_images)
  all_labels = np.array(sad_labels + happy_labels)

  shuffled_images, shuffled_labels = shuffle(all_images, all_labels, random_state=42)
  return shuffled_images,shuffled_labels

train_images,train_labels = shuffle_images(os.path.join(path,'Images_split/train/sad/'),os.path.join(path,'Images_split/train/happy/'))
val_images,val_labels = shuffle_images(os.path.join(path,'Images_split/validation/sad/'),os.path.join(path,'Images_split/validation/happy/'))
test_images,test_labels = shuffle_images(os.path.join(path,'Images_split/test/sad/'),os.path.join(path,'Images_split/test/happy/'))

train_val_images,train_val_labels = np.concatenate([train_images,val_images]),np.concatenate([train_labels,val_labels])

def get_hog_features(dataset):
  images_hog = []
  for image in dataset:
    image = color.rgb2gray(image)
    features= hog(image, orientations=15, pixels_per_cell=(15, 15),
                        cells_per_block=(2, 2), visualize=False, block_norm="L1", transform_sqrt=False,multichannel=False)
    images_hog.append(features)
  return images_hog

hog_images = get_hog_features(train_val_images)

hog_images_test = get_hog_features(test_images)

import matplotlib.pyplot as plt
resized_image = transform.resize(train_val_images[188], (200,200))
image1 = color.rgb2gray(resized_image)
hog_feature, hog_image= hog(image1, orientations=15, pixels_per_cell=(15, 15), cells_per_block=(2, 2), visualize=True, block_norm="L2", transform_sqrt=False)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

ax1.imshow(resized_image, cmap=plt.cm.gray)
ax1.set_title('Image')

ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title('HOG Image')

ax1.grid(False)
ax2.grid(False)
plt.show()

from collections import Counter
from imblearn.over_sampling import SMOTE
count_l = Counter(np.squeeze(train_val_labels))
smote = SMOTE()
features_smote_hog, y_smote_hog = smote.fit_resample(hog_images,train_val_labels)
count_smote_l = Counter(np.squeeze(y_smote_hog))

# Random Forest with the updated dataset oversampled with SMOTE
best_model = grid_search.best_estimator_
best_model.fit(features_smote_hog, y_smote_hog)
y_pred = best_model.predict(hog_images_test)

print(classification_report(test_labels, y_pred))

fig, ax = plt.subplots(figsize=(8,4))
cm = confusion_matrix(test_labels, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
cmd.plot(values_format='d',ax=ax)
ax.set_title('Confusion Matrix for HOG features with SMOTE technique')
plt.show()

"""# Dummy Classifier"""

# random classifier
from sklearn.dummy import DummyClassifier
dc = DummyClassifier(strategy ='stratified')
dc.fit(features_smote_hog, y_smote_hog)
y_pred = dc.predict(hog_images_test)

from sklearn.metrics import classification_report
print(classification_report(test_labels, y_pred))

fig, ax = plt.subplots(figsize=(8,4))
cm = confusion_matrix(test_labels, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
cmd.plot(values_format='d',ax=ax)
ax.set_title('Confusion Matrix for HOG features with SMOTE technique')
plt.show()

"""# Agglomerative clustering"""

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import KNeighborsClassifier


agg_cluster = AgglomerativeClustering(n_clusters=2)
agg_cluster = agg_cluster.fit_predict(features_smote_hog)

knn = KNeighborsClassifier()
knn.fit(features_smote_hog,np.squeeze(agg_cluster))
y_pred = knn.predict(hog_images_test)

silhouette_avg = silhouette_score(features_smote_hog, agg_cluster)
silhouette_avg

from sklearn.metrics import classification_report
print(classification_report(test_labels, y_pred))

"""Hyperparameter tuning"""

def find_best_parameters(features, param):
  max_score = 0
  best_parameters = None
  score=[]

  for n_clusters in param:

      agg_cluster = AgglomerativeClustering(n_clusters=n_clusters)

      c_labels = agg_cluster.fit_predict(features)

      silhouette_score1 = silhouette_score(features, c_labels)
      score.append(silhouette_score1)
      print(f'N_Clusters: {n_clusters}, silhouette score= {silhouette_score1}')

      if silhouette_score1 > max_score:
          max_score = silhouette_score1
          best_parameters = n_clusters

  return best_parameters, max_score,score

from sklearn.metrics import silhouette_score
params = [2, 3, 4, 5, 6, 7, 8, 9, 10]
param,best_score_hog, score_hog = find_best_parameters(features_smote_hog,params)

n_clusters_list = [2,3,4,5,6,7,8,9,10]
plt.scatter(n_clusters_list, score_hog, c="blue", label="HOG")
plt.scatter(n_clusters_list, score_vgg, c="green", label="VGG")

plt.plot(n_clusters_list, score_hog, linestyle="--", marker="o", color="blue")
plt.plot(n_clusters_list, score_vgg, linestyle="--", marker="o", color="green")

plt.legend()
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

plt.show()

"""# HDBScan"""

import hdbscan
def find_best_parameters_hdbscan(features, params):
  max_score = 0
  best_min_cluster_size = None
  score = []

  for param in params:

      hdb = hdbscan.HDBSCAN(min_cluster_size=param)

      c_labels = hdb.fit_predict(features)

      silhouette_score1 = silhouette_score(features, c_labels)
      print(f"min_cluster_size: {param}; silhouette score= {silhouette_score1}")
      score.append(silhouette_score1)

      if silhouette_score1 > max_score:
          max_score = silhouette_score1
          best_min_cluster_size = param
  return best_params, max_score, score

params = [3,5,10]
best_params_hog, best_score_hog, score_hog = find_best_parameters_hdbscan(features_smote_hog,params)

l_hdb = hdbscan.HDBSCAN(min_cluster_size=17).fit_predict(features_smote_hog)
unique_labels, label_counts = np.unique(l_hdb, return_counts=True)
for label, count in zip(unique_labels, label_counts):
    print(f"Label {label}: {count} occurrences")

knn = KNeighborsClassifier()
knn.fit(features_smote_hog,np.squeeze(l_hdb))
y_pred = knn.predict(hog_images_test)

from sklearn.metrics import classification_report
print(classification_report(test_labels, y_pred))

params = [3,5,10]
plt.scatter(params, score_hog, c="blue", label="HOG")
plt.scatter(params, score, c="green", label="VGG")

plt.plot(params, score_hog, linestyle="--", marker="o", color="blue")
plt.plot(params, score, linestyle="--", marker="o", color="green")

plt.legend()
plt.title('Silhouette Scores for Different Minimum Cluster Size')
plt.xlabel('Minimum cluster size')
plt.ylabel('Silhouette Score')

plt.show()