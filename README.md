# Lung-Nodule-Detection
Lung Nodule Detection using Modified KNN algorithm. 
Cancer is a condition as soon as a few of the body's tissues 
get larger and are out of control and migrate to other bodily 
regions. In the many cells that generate up the human body, 
cancer can develop practically anywhere . Individual cells 
often divide to create new cells as the body needs them. New 
cells return old ones when they die as a consequence of 
ageing or damage .
Lung cancer starts in the lungs and can extend to the 
lymph nodes or even other bodily functions, like the brain. 
The lungs may potentially become infected with cancer from 
other tissues . Metastases are the term used to describe the 
spread of cancer cells from one tissue to the next. Lung 
nodules in CT images can now be seen using computer-aided 
detection (CAD) techniques, which have recently been 
created which might result in incorrect descriptive statistical 
and incorrect diagnostics by doctors sometimes .
By tuning the hyperparameters, the modified KNN 
algorithm can improve its performance compared to the 
traditional KNN algorithm. This is because the optimal 
values of the hyperparameters can be different for different 
data sets and problems. By tuning the hyperparameters, the 
modified KNN algorithm can adapt to the specific 
characteristics of the data, resulting in improved 
performance.
In this system, the aim is to decrease the number of 
hyperparameters in order to shorten the execution time for 
lung nodule detection. By simplifying the algorithm and 
leaving out certain hyperparameters such as k, the study aims 
to improve the performance of the detection method. The 
accuracy and time detection of the optimized algorithm is 
found to be quite low.


<b>Steps Involved are:<b><br>
1.The first step is image acquisition, which involves obtaining an image dataset.<br>
2. The dataset is then divided into two parts: training and testing.<br>
3. The dataset is further normalized using the Min-Max scalar algorithm and is divided into negative and positive cases.<br>
4. Image preprocessing is done using the standard scalar algorithm, which standardizes the hyperparameters by subtracting the mean and scaling to unit variance.<br>
5. The model then downsamples the training and testing data.<br>
6. Apply CV2 BRISK Algorithm. CV2 BRISK algorithm is used to select features of an image in a given dataset which is turned into grayscale. And then random images are taken to obtain binary features.<br>
7. Applying the modified K-NN algorithm , In this model 5 hyperparameters are used which are , n-neighbours , weight , leaf-size, algorithm , metric. The Gaussian kernel is designed in weight.Algorithm can be auto, kd-tree , ball-tree , brute. <br>
8. This 5 parameters are converted to dictionary. It is provided to grid search then grid search will select the best parameters. <br>
9. Now this will fit the model and train the dataset after that Model can calculate the validation and training , testing score.<br>
