# %%
from google.colab import drive
drive.mount('/content/gdrive')

# %%
!pip install tensorflow
import tensorflow
print(tensorflow.__version__)
#!pip install tensorflow==1.15.3

# %%
!pip install Pillow
!pip install scipy==1.1.0

# %%
! pip install SimpleITK

# %%
import SimpleITK as sitk

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns 

import os
import glob

##import SimpleITK as sitk

from PIL import Image

from scipy.misc import imread

%matplotlib inline
from IPython.display import clear_output
pd.options.mode.chained_assignment = None

# %%
annotations = pd.read_csv('/content/gdrive/My Drive/Dataset_LUNA_16/annotations.csv')
candidates = pd.read_csv('/content/gdrive/My Drive/Dataset_LUNA_16/candidates.csv')

# %%
annotations.head()

# %%
candidates['class'].sum()

# %%
candidates.info()

# %%
print(len(candidates[candidates['class'] == 1]))
print(len(candidates[candidates['class'] == 0]))

# %%
import multiprocessing
num_cores = multiprocessing.cpu_count()
print(num_cores)

# %%
class CTScan(object):
    def __init__(self, filename = None, coords = None):
        self.filename = filename
        self.coords = coords
        self.ds = None
        self.image = None

    def reset_coords(self, coords):
        self.coords = coords

    def read_mhd_image(self):
        path = glob.glob('/content/gdrive/My Drive/Dataset_LUNA_16/seg-lungs-LUNA16/*/'+ self.filename + '.mhd')
        self.ds = sitk.ReadImage(path[0])
        self.image = sitk.GetArrayFromImage(self.ds)

    def get_resolution(self):
        return self.ds.GetSpacing()

    def get_origin(self):
        return self.ds.GetOrigin()

    def get_ds(self):
        return self.ds

    def get_voxel_coords(self):
        origin = self.get_origin()
        resolution = self.get_resolution()
        voxel_coords = [np.absolute(self.coords[j]-origin[j])/resolution[j] \
            for j in range(len(self.coords))]
        return tuple(voxel_coords)
    
    def get_image(self):
        return self.image
    
    def get_subimage(self, width):
        self.read_mhd_image()
        x, y, z = self.get_voxel_coords()
        subImage = self.image[z, y-width/2:y+width/2, x-width/2:x+width/2]
        return subImage   
    def normalizePlanes(self, npzarray):
        maxHU = 400.
        minHU = -1000.
        npzarray = (npzarray - minHU) / (maxHU - minHU)
        npzarray[npzarray>1] = 1.
        npzarray[npzarray<0] = 0.
        return npzarray
    
    def save_image(self, filename, width):
        image = self.get_subimage(width)
        image = self.normalizePlanes(image)
        Image.fromarray(image*255).convert('L').save(filename)

# %%
positives = candidates[candidates['class']==1].index  
negatives = candidates[candidates['class']==0].index

# %%
scan = CTScan(np.asarray(candidates.iloc[negatives[600]])[0], \
              np.asarray(candidates.iloc[negatives[600]])[1:-1])
scan.read_mhd_image()
x, y, z = scan.get_voxel_coords()
image = scan.get_image()
dx, dy, dz = scan.get_resolution()
x0, y0, z0 = scan.get_origin()

# %%
filename = '1.3.6.1.4.1.14519.5.2.1.6279.6001.100398138793540579077826395208'
coords = (70.19, -140.93, 877.68)#[877.68, -140.93, 70.19]
scan = CTScan(filename, coords)
scan.read_mhd_image()
x, y, z = scan.get_voxel_coords()
image = scan.get_image()
dx, dy, dz = scan.get_resolution()
x0, y0, z0 = scan.get_origin()

# %%
positives

# %%
np.random.seed(42)
negIndexes = np.random.choice(negatives, len(positives)*5, replace = False)

# %%
candidatesDf = candidates.iloc[list(positives)+list(negIndexes)]

# %%
from sklearn.model_selection import train_test_split
X = candidatesDf.iloc[:,:-1]
y = candidatesDf.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

# %%
X_train.size

# %%
y_train

# %%
y_test

# %%
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = 42)

# %%
X_train.size

# %%
X_train

# %%
y_train

# %%
len(X_train)

# %%
# for faster computation, min max Scaler
def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

# %%
print('number of positive cases are ' + str(y_train.sum()))
print('total set size is ' + str(len(y_train)))
print('percentage of positive cases are ' + str(y_train.sum()*1.0/len(y_train)))

# %%
#  for balancing we created two new variables
tempDf = X_train[y_train == 1]
tempDf = tempDf.set_index(X_train[y_train == 1].index + 1000000)
X_train_new = X_train.append(tempDf)
tempDf = tempDf.set_index(X_train[y_train == 1].index + 2000000)
X_train_new = X_train_new.append(tempDf)

ytemp = y_train.reindex(X_train[y_train == 1].index + 1000000)
ytemp.loc[:] = 1
y_train_new = y_train.append(ytemp)
ytemp = y_train.reindex(X_train[y_train == 1].index + 2000000)
ytemp.loc[:] = 1
y_train_new = y_train_new.append(ytemp)

print(len(X_train_new), len(y_train_new))

# %%
X_train_new

# %%
y_train_new

# %%
X_train_new.index

# %%
import scipy.misc

# %%
import cv2

# %%
#from scipy.misc import imresize
from PIL import ImageEnhance

#  like image compression
class PreProcessing(object):
    def __init__(self, image = None):
        self.image = image
    
    def subtract_mean(self):
        self.image = (self.image/255.0 - 0.25)*255
        return self.image
    
    def downsample_data(self):
        self.image = cv2.resize(self.image, (40,40), interpolation = cv2.INTER_AREA)
        return self.image

    def upsample_data(self):
        self.image = cv2.resize(self.image, (224, 224), interpolation = cv2.INTER_CUBIC)
        return self.image

# %% [markdown]
# Imported the preprocessed and segmented dataset

# %%
import matplotlib.pyplot as plt
dirName = '/content/gdrive/My Drive/Dataset_LUNA_16/data/train/'
plt.figure(figsize = (10,10))
inp = imread(dirName + 'image_'+ str(30517) + '.jpg')
print ("Original shape of input image: ", inp.shape)
plt.subplot(221)
plt.imshow(inp, cmap='gray')
plt.grid(False)

inp = PreProcessing(inp).upsample_data()
Pp = PreProcessing(inp)

inp2 = Pp.subtract_mean()
plt.subplot(222)
plt.imshow(inp2, cmap='gray')
plt.grid(False)

inp3 = ImageEnhance.Contrast(Image.fromarray(inp))
contrast = 1.5
inp3 = inp3.enhance(contrast)
plt.subplot(223)
plt.imshow(inp3, cmap='gray')
plt.grid(False)

inp4 = Pp.downsample_data()
plt.subplot(224)
plt.imshow(inp4,cmap='gray')
plt.grid(False)


# %%
inp.shape

# %%
inp.dtype

# %%
#!pip install tflearn==0.3.2
#import tflearn

# %%
y_train_new.values.astype(int)

# %% [markdown]
# Training

# %%
dirName

# %%
#  saving training images in directory
train_filenames =\
X_train_new.index.to_series().apply(lambda x:\
                                    '/content/gdrive/My Drive/Dataset_LUNA_16/data/train/image_'+str(x)+'.jpg')
train_filenames.values.astype(str)

# %%
train_filenames.values.astype(str)

# %%
#@title Default title text
dataset_file = 'traindatalabels.txt'
filenames = train_filenames.values.astype(str)
train_filenames =X_train_new.index.to_series().apply(lambda x:filenames)
labels = y_train_new.values.astype(int)
traindata = np.zeros(filenames.size,\
                     dtype=[('var1', 'S36'), ('var2', int)])
traindata['var1'] = filenames
traindata['var2'] = labels

np.savetxt(dataset_file, traindata, fmt="%10s %d")

# %%
traindata

# %%
#import tflearn

# %%
# Build a HDF5 dataset (only required once)
#from tflearn.data_utils import build_hdf5_image_dataset
#build_hdf5_image_dataset(dataset_file, image_shape=(50, 50), mode='file', output_path='traindataset.h5', categorical_labels=True, normalize=True)

# %%
# Load HDF5 dataset
#  file datatype
import h5py
h5f = h5py.File('/content/gdrive/My Drive/Dataset_LUNA_16/data/traindataset.h5', 'r')
X_train_images = h5f['X']
Y_train_labels = h5f['Y']

h5f2 = h5py.File('/content/gdrive/My Drive/Dataset_LUNA_16/data/valdataset.h5', 'r')
X_val_images = h5f2['X']
Y_val_labels = h5f2['Y']

# %%
X_train_images

# %%
Y_train_labels

# %% [markdown]
# Testing_image_processing

# %%
X_train_processing = np.array(X_train_images)
X_val_processing = np.array(X_val_images)

# %%
print(X_train_processing.shape)
print(X_val_processing.shape)

# %%
print(X_train_processing.dtype)
print(X_val_processing.dtype)

# %%
img_test = X_train_processing[1000]
img_test_downsampled = PreProcessing(img_test).downsample_data()
img_test_upsampled = PreProcessing(img_test).upsample_data()

# %%
print(Y_train_labels[1000])

# %%
plt.subplot(121)
plt.imshow(img_test,cmap='gray')
plt.subplot(122)
plt.imshow(img_test_downsampled,cmap='gray')

# %%
plt.subplot(121)
plt.imshow(img_test,cmap='gray')
plt.subplot(122)
plt.imshow(img_test_upsampled,cmap='gray')

# %%
img_test_subtracted_mean = PreProcessing(img_test_upsampled).subtract_mean()
plt.subplot(121)
plt.imshow(img_test_upsampled,cmap='gray')
plt.subplot(122)
plt.imshow(img_test_subtracted_mean,cmap='gray')

# %% [markdown]
# testing cv2 BRISK algorithm

# %% [markdown]
# need some work and research

# %%
import cv2
brisk = cv2.BRISK_create(50)
img_BRISK_1 = img_test_upsampled.copy()
rgb = cv2.cvtColor(img_BRISK_1, cv2.COLOR_GRAY2RGB)
img_BRISK_2 = rgb*1000
img_BRISK_3 = np.array(img_BRISK_2, dtype=np.uint8)
img_contr_copy = img_BRISK_3.copy()

plt.subplot(221)
plt.imshow(img_BRISK_3, cmap='gray')
kpts_1, des_1 = brisk.detectAndCompute(img_BRISK_3, None)

image_BRISK_1 = cv2.drawKeypoints(image=img_BRISK_3, outImage=img_BRISK_3, keypoints=kpts_1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.subplot(222)
plt.imshow(image_BRISK_1, cmap='gray')


# with enhanced contrast

inp_contr = ImageEnhance.Contrast(Image.fromarray(img_contr_copy))
contrast = 1.5
img_contr = inp_contr.enhance(contrast)
plt.subplot(223)
plt.imshow(img_contr, cmap='gray')
img_BRISK_4 = np.array(img_contr)
kpts_2, des_2 = brisk.detectAndCompute(img_BRISK_4, None)

image_BRISK_2 = cv2.drawKeypoints(image=img_BRISK_4, outImage=img_BRISK_4, keypoints=kpts_2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.subplot(224)
plt.imshow(image_BRISK_2, cmap='gray')

# %% [markdown]
# test both with other CV2 algorithms and then implement all those algorithms with DL model self defined.

# %% [markdown]
# Converting to training dataset

# %%
# to detect grayscale image we have expanded the dimensions
X_train_images_np = np.expand_dims(np.array(X_train_images), axis = 3)
y_train_labels_np = np.array(Y_train_labels)
X_val_images_np = np.expand_dims(np.array(X_val_images), axis = 3)
y_val_labels_np = np.array(Y_val_labels)

# %%
print(X_train_images_np.shape)
print(y_train_labels_np.shape)
print(X_val_images_np.shape)
print(y_val_labels_np.shape)

# %%
print(X_train_images_np.dtype)
print(y_train_labels_np.dtype)
print(X_val_images_np.dtype)
print(y_val_labels_np.dtype)

# %% [markdown]
# processing for test dataset

# %%
h5f2 = h5py.File('/content/gdrive/My Drive/Dataset_LUNA_16/data/testdataset.h5', 'r')
X_test_images = h5f2['X']
Y_test_labels = h5f2['Y']

# %%
X_test_images_np = np.expand_dims(np.array(X_test_images), axis = 3)
y_test_labels_np = np.array(Y_test_labels)

# %%
print(X_test_images_np.shape)
print(y_test_labels_np.shape)

# %%
print(X_test_images_np.dtype)
print(y_test_labels_np.dtype)

# %% [markdown]
# Model Making and Model Training

# %%
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,BatchNormalization,AveragePooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# %% [markdown]
# model based on deep cnn

# %%
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(50, 50, 1)))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.1))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.3))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.4))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'));return model

mobile = define_model()

# %%
mobile.summary()

# %%
op_layer = mobile.output
final_layer = Dense(128,activation='relu',kernel_initializer='he_uniform')(op_layer)
final_layer = Dense(128,activation='relu',kernel_initializer='he_uniform')(final_layer)
final_layer = Dense(2,activation= 'softmax')(final_layer)
from keras.models  import Model
# Define model input and output
model = Model(inputs = mobile.input , outputs = final_layer)
opt = SGD(lr=0.001, momentum=0.9)
import keras
optimizer_sgd = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer_sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# %%
model.summary()

# %%
# plot diagnostic learning curves
import matplotlib.pyplot as plt
def summarize_diagnostics(hist):
	plt.plot(hist.history["accuracy"])
	plt.plot(hist.history['val_accuracy'])
	plt.plot(hist.history['loss'])
	plt.plot(hist.history['val_loss'])
	plt.subplot(244, grayscale)
	plt.title("model accuracy")
	plt.ylabel("Accuracy")
	plt.xlabel("Epoch")
	plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
	plt.imshow()


# %%
#  overfitting avoid
callbacks = [
    keras.callbacks.ModelCheckpoint('classification_deep_conv_model.h5', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau()]

# %%
# run the test harness for evaluating a model
#  multiple copies of the image at diff angles
def run_test_harness():
	# create data generator
	datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,zoom_range=0.1)
	# prepare iterator
	it_train = datagen.flow(X_train_images_np, y_train_labels_np, batch_size=64)
	# fit model
	steps = int(X_train_images_np.shape[0] / 64)
	hist = model.fit_generator(it_train, steps_per_epoch=steps, epochs=500, validation_data=(X_val_images_np, y_val_labels_np), verbose=1, callbacks = callbacks)
	# evaluate model
	_, acc = model.evaluate(X_val_images_np, y_val_labels_np, verbose=1)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(hist)

# entry point, run the test harness
run_test_harness()

# %%
# model saved and load again
model_new_1 = keras.models.load_model("classification_deep_conv_model.h5")

# %%
model_new_1.summary()

# %%
from sklearn.metrics import confusion_matrix,classification_report
y_pred = model_new_1.predict(X_test_images_np)
y_p = np.argmax(y_pred,axis=1)
y_true = np.argmax(y_test_labels_np,axis=1)
print('confusion matrix')
print(confusion_matrix(y_true,y_p))


# %%
print('Classification report')
print(classification_report(y_true,y_p))

# %%
#  z score normalisation technique
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
op_layer = model_new_1.get_layer('dense_3').output
model_new = Model(inputs = model_new_1.input,outputs = op_layer)
train_new = sc.fit_transform(model_new.predict(X_train_images_np))
val_new = sc.fit_transform(model_new.predict(X_val_images_np))
test_new = sc.fit_transform(model_new.predict(X_test_images_np))

# %%
def gaussian_kernel(distances):
  kernel_width=2
  weights=np.exp(-(distances ** 2)/ kernel_width)
  return weights

# %%
from sklearn.svm import SVC
svm = SVC(kernel='rbf')
svm.fit(train_new,np.argmax(y_train_labels_np,axis=1))
svm_train = svm.score(train_new,np.argmax(y_train_labels_np,axis=1))
svm_test = svm.score(test_new,np.argmax(y_test_labels_np,axis=1))
print('Training score of svm',svm.score(train_new,np.argmax(y_train_labels_np,axis=1)))
print('validation score of svm :',svm.score(val_new,np.argmax(y_val_labels_np,axis=1)))
print('testing score of svm :',svm.score(test_new,np.argmax(y_test_labels_np,axis=1)))

# %%
from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(train_new,np.argmax(y_train_labels_np,axis=1))
xg_train = xg.score(train_new,np.argmax(y_train_labels_np,axis=1))
xg_test = xg.score(test_new,np.argmax(y_test_labels_np,axis=1))
print('Training score of xgb',xg.score(train_new,np.argmax(y_train_labels_np,axis=1)))
print('validation score of xgb :',xg.score(val_new,np.argmax(y_val_labels_np,axis=1)))
print('testing score of xgb :',xg.score(test_new,np.argmax(y_test_labels_np,axis=1)))

# %%
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(train_new,np.argmax(y_train_labels_np,axis=1))
knn_train = classifier.score(train_new,np.argmax(y_train_labels_np,axis=1))
knn_test = classifier.score(test_new,np.argmax(y_test_labels_np,axis=1))
print('Training score of KNN',classifier.score(train_new,np.argmax(y_train_labels_np,axis=1)))
print('validation score of KNN :',classifier.score(val_new,np.argmax(y_val_labels_np,axis=1)))
print('testing score of KNN :',classifier.score(test_new,np.argmax(y_test_labels_np,axis=1)))

# %%
def knn(X_train, y_train, X_test, k):
    distances = []
    targets = []
    
    for i in range(len(X_train)):
        distance = np.sqrt(np.sum(np.square(X_test - X_train[i, :])))
        distances.append([distance, i])
        
    distances = sorted(distances)
    
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])
        
    return np.argmax(np.bincount(targets))

# Number of nearest neighbors
k = 3

# Predict the labels for the test data
y_pred = [knn(X_train, y_train, x, k) for x in X_test]
print(y_pred)

# %%
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'manhattan', algorithm="kd_tree", p = 2, weights=gaussian_kernel, leaf_size=1)
classifier.fit(train_new,np.argmax(y_train_labels_np,axis=1))
mknn_train = classifier.score(train_new,np.argmax(y_train_labels_np,axis=1))
mknn_test = classifier.score(test_new,np.argmax(y_test_labels_np,axis=1))
print('Training score of KNN',classifier.score(train_new,np.argmax(y_train_labels_np,axis=1)))
print('validation score of KNN :',classifier.score(val_new,np.argmax(y_val_labels_np,axis=1)))
print('testing score of KNN :',classifier.score(test_new,np.argmax(y_test_labels_np,axis=1)))

# %%
plt.plot(np.array([svm_train, svm_test]), label="SVM")
plt.plot(np.array([xg_train, xg_test]), label="Gradient Boosting")
plt.plot(np.array([knn_train, knn_test]), label="KNN")
plt.plot(np.array([mknn_train, mknn_test]), label="Modified KNN")
plt.xlabel('Training score')
plt.ylabel('Test score')
plt.title('Comparison of algorithms')
plt.legend()
plt.show()

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, GridSearchCV
from sklearn.datasets import make_classification

X,y = make_classification(n_samples=1000, n_features=8,n_classes=3, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42, stratify=y)

# %%
#base model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

#predict
y_pred=knn.predict(X_test)

#summarize results
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
#print('Training score of KNN',classifier.score(train_new,np.argmax(y_train_labels_np,axis=1)))
#print('validation score of KNN :',classifier.score(val_new,np.argmax(y_val_labels_np,axis=1)))
#print('testing score of KNN :',classifier.score(test_new,np.argmax(y_test_labels_np,axis=1)))

# %%
knn_params = {
    "n_neighbors": range(1, 30, 2),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"],
    # "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "leaf_size": range(1, 50, 5)
}

knn = KNeighborsClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
grid_search = GridSearchCV(estimator=knn, param_grid=knn_params, n_jobs=1, cv=cv, scoring="accuracy", error_score=0)
grid_results = grid_search.fit(X_train, y_train)

final_model = knn.set_params(**grid_results.best_params_)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

#summarize results
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(grid_results.best_params_)

# %%



