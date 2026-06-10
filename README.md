Big Data Machine Learning algorithm 

Introduction 

This project consists of a dataset of 28,000 images of 10 different animals and documents the creation and design process of a CNN model that learns how to identify the different animals present in the dataset. There are 10 different animals represented in this dataset: Dog, cat, sheep, cow, chicken, spider, butterfly, squirrel and elephant. A dataset of this siz means there must be careful consideration  between accuracy, computational efficiency and time. There are 4 main phases to this ML pipeline:

1 - Data collection and manipulation: splitting raw assets and safeguarding data stream pipeline integrity by checking the incoming data
2 - Augmentations and loading: Artificially expanding the training pool and normalizing inputs.
3 - Model creation: Feature extraction through deep network layers, using between 2 and 4 convolution layers for this model. 
4 - Diagnostic evaluation: Comprehensive assessment of model performance and error tracking.

Project Purpose: The purpose of this project is to create a CNN model that can as accurately a possible identify an image of an animal fom a dataset containing 10 different types of animal image. 

Phases of the ML Pipeline:

Phase 1 - Data partitioning: pipeline isolates the data to ensure rigorous, unbiased testing.
Phase 2 - Ingestion, validation and augmentation: This stage is the gatekeeper, cleaning and transforming data on the fly before giving it to the GPU.
Phase 3 - Feature extraction and architecture: The model uses a 2-4-tier Convolutional Neural Network designed to dynamically break down complex images into simple numerical features.
Phase 4 - Optimisation and regularisation: The model takes batches of images makes a guess, calculates how wrong it was using categorical_crossentropy, and updates its millions of internal weights using the Adam optimizer. It monitors this over 20 epochs with a patience of 3-5
Phase 5 - Diagnostic evaluation: Once training finishes, the pipeline runs a rigorous check across all three splits (Train, Validation, and Test) to see exactly where the model succeeds or struggles. allowing the programmer to act to change the model based on teh results received. 

1) Data Collection- I have 2 versions of this code, one on github which uses the kagglehub library to download the dataset directly from Kaggle and the second which uses a local download of the dataset. I had to use this second method as the processing times when running the code on github could take up to over 30 minutes depending on the setting used, In order to try to speed this up I made alterations to allow the model to run locally on my laptop's onboard GPU in order to decrease processing time. Due to having to use an older version of python and tensorflow I encountered issues that I could not ix in a timely manner when trying use the kagglehub library to downlaod he dataset straight from Kaggle. Therefore, I decided to just download the dataset and run it locally as I decided to continue trying to get kagglehub to work was not a worthwhile way to spend the time.

This is the local version used: 
datapath = r"C:\Mechanical engineering\Big Data\BigData\raw-img" 
print("using dataset path:", datapath) 
print("Contents:", os.listdir(datapath))

This is the download version used:
path = kh.dataset_download("alessiocorrado99/animals10") # downloads the Animals-10 dataset from kaggle 
datapath = os.path.join(path,"raw-img") # builds correct path to images 
print ("using dataset path:",datapath) # prints dataset lecation 
print ("Contents:", os.listdir(datapath)) # lists all class folders 

The code also possesses data validation to prevent errors or corrupt inputs. The following code loops batch by batch through a data generator, verifying that every file is a proper NumPy array, strictly matches the dimension rule (depending on what the img_size has been set through for each run), verifies pixel scaling resides inside th [0,1] range and checks for corupted NaN values. 

The code below:
def check_images(generator, dataset_name):
    invalid_count = 0  
    valid_count = 0    
    num_batches = len(generator)
    print(f"Checking {dataset_name} ({generator.n} total images across {num_batches} batches)...")

    for batch_idx in range(num_batches):
        images, _ = generator[batch_idx]
        
        for img_idx, image in enumerate(images):
            if not isinstance(image, np.ndarray):
                print(f"{dataset_name} - Batch {batch_idx}, Img {img_idx}: Not a valid image array")
                invalid_count += 1
                continue

            if image.shape != (128, 128, 3):
                print(f"{dataset_name} - Batch {batch_idx}, Img {img_idx}: Incorrect shape {image.shape}")
                invalid_count += 1
                continue

            if not (image.min() >= 0.0 and image.max() <= 1.0):
                print(f"{dataset_name} - Batch {batch_idx}, Img {img_idx}: Invalid scaled pixel values (Min: {image.min()}, Max: {image.max()})")
                invalid_count += 1
                continue

            if np.isnan(image).any():
                print(f"{dataset_name} - Batch {batch_idx}, Img {img_idx}: Contains NaN values")
                invalid_count += 1
                continue

            valid_count += 1

    print(f"=== {dataset_name} Check Complete ===")
    print(f"Valid images: {valid_count} | Invalid images: {invalid_count}\n")

Finally the dataset is split into 3 components: 70% training, 15% validation, and 15% testing 
Uses a seed of 42 to fix the psuedo-random generator so this identical split can be replicated everytime the code runs.
The code used:

output_folder = r"C:\Mechanical engineering\Big Data\BigData\output_images"
splitfolders.ratio(datapath, output=output_folder, seed=42, ratio=(.7, .15, .15))


2) EDA 

The class names were all originally in italian, therefore I had to change them to english. This was achieved by extracting the subfolder class indices, it maps them cleanl to english, grabs a single localised batch of images and renders teh first item onscreen alongside its mappedc ategory title 
the code used below:

italian_to_english = { 
    "cane": "dog", "gatto": "cat", "cavallo": "horse", "ragno": "spider",
    "farfalla": "butterfly", "gallina": "chicken", "pecora": "sheep",
    "mucca": "cow", "scoiattolo": "squirrel", "elefante": "elephant"
}
original_classes = list(train_data.class_indices.keys()) 
class_names = [italian_to_english[c] for c in original_classes] 

images, labels = next(train_data) 

plt.imshow(images[0]) 
plt.title(f"Label: {class_names[np.argmax(labels[0])]}") 
plt.axis('off') 
plt.show()



The total of each different category of image also needs to be totalled and displayed. It uses np.unique to scan the absolute image frequencies assigned to each category inside the folder directories. It appends these values into a Pandas DataFrame, then builds a grouped Seaborn bar plot showing how labels are distributed across training, validation, and testing scopes.

the following code is how it was achieved:

df_freq = pd.DataFrame(columns=['Set', 'Label', 'Frequency']) 

def labelcount(generator, dataset_name): 
    global df_freq 
    counts = generator.classes 
    unique, freq = np.unique(counts, return_counts=True) 
    for u, f in zip(unique, freq): 
        df_freq = pd.concat([ 
            df_freq,
            pd.DataFrame([{'Set': dataset_name, 'Label': class_names[u], 'Frequency': f}])
        ], ignore_index=True) 

labelcount(train_data, "Train")
labelcount(val_data, "Validation")
labelcount(test_data, "Test")

plt.figure(figsize=(10, 6))
sns.barplot(data=df_freq, x='Set', y='Frequency', hue='Label')
plt.xticks(rotation=45)
plt.title("Label Distribution")
plt.show()

The code also allows the model to train off of images viewed at different angles:

Train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,      
    width_shift_range=0.2,  
    height_shift_range=0.2, 
    shear_range=0.2,        
    zoom_range=0.2,         
    horizontal_flip=True,   
    fill_mode='nearest'
)
train_data = Train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    shuffle=True
)

3) Model Building:

Very basic initial parameters were used, similar to what was used in lecture slides. The following are the basic charcateristics used for the first version of the model:
28,28 image resolution
32 batch size 
16 and 16 filter layer model with 128 neuron density
Patience = 3
Epochs = 20 (13 used)

Version 1's model code:

def build_tf_model(input_shape, n_labels): # creates the neural network 
    model = Sequential()

    # First Convolutional Block 
    model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=input_shape, activation='relu')) #learns 16 different filters to detect basic features like edges and textures, the kernel size of (3,3) means it looks at 3x3 pixel areas at a time, the activation function 'relu' helps the model learn complex patterns by introducing non-linearity.
    model.add(MaxPool2D(pool_size=(2, 2))) # reduces the image dimension by taking the maximum value in each 2x2 region 

    # Second Convolutional Block to extract more advanced features
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu')) # learns 16 different filters
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Flattening the feature maps, converts the 2D feature maps into a 1D vector  and prepares data for the dense layers
    model.add(Flatten())

    model.add(Dense(128, activation='relu')) # standard neural network layer, 128 neurons and it learns the combinations of extracted features.
    model.add(Dropout(0.25))  # Randomly turns off 25% of neurons during training to prevent overfitting by forcing the model to learn more robust features that are not reliant on specific neurons.

    model.add(Dense(n_labels, activation='softmax')) # produces probabilities for each class, softmax converts outputs into probabilities 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # compiles the model with categorical crossentropy loss (suitable for multi-class classification), the Adam optimizer (An efficiency focused optimizer), and tracks accuracy as a performance metric.

    return model


The main aim of the first version was just to get a model mostly working and ideally quickly therefore basic and simple parameters were selected to ensure a quick execution of th model. Once information on how the model was interacting with and interpreting this first version then I could improve the areas that the model showed needed improvement hopefully reducing wasting processing time testing changes that don't have an effect. Eventually as the model progressed and the resolution increased, a deeper model was needed resulting in the use of the following as the final model code:

def build_tf_model(input_shape, n_labels): # creates the neural network 
    model = Sequential()

    # First Convolutional Block 
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation='relu')) #learns 32 different filters to detect basic features like edges and textures, the kernel size of (3,3) means it looks at 3x3 pixel areas at a time, the activation function 'relu' helps the model learn complex patterns by introducing non-linearity.
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2))) # reduces the image dimension by taking the maximum value in each 2x2 region 

    # Second Convolutional Block to extract more advanced features
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')) # learns 64 different filters
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
     # Third Convolutional Block to extract more advanced features
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu')) # learns 64 different filters
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu')) # learns 64 different filters
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Flattening the feature maps, converts the 2D feature maps into a 1D vector  and prepares data for the dense layers
    model.add(Flatten())

    model.add(Dense(256, activation='relu')) # standard neural network layer, 128 neurons and it learns the combinations of extracted features.
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dropout(0.5))  # Randomly turns off 50% of neurons during training to prevent overfitting by forcing the model to learn more robust features that are not reliant on specific neurons.

    model.add(Dense(n_labels, activation='softmax')) # produces probabilities for each class, softmax converts outputs into probabilities 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # compiles the model with categorical crossentropy loss (suitable for multi-class classification), the Adam optimizer (An efficiency focused optimizer), and tracks accuracy as a performance metric.

    return model

As can be seen 2 more convolution layers were added along with filters starting at 32 and increasing for each layer. Normalisation was also added and neuron density was increased to 256. This was all added in response to increasing the resolution to 128,128 as the increased detail in the image mean a deeper convolution network was needed to analyse it successfully. 

4) Model Evaluation:

Version 1: 

Key Characteristics:
28,28 image resolution
32 batch size 
16 and 16 filter layer model with 128 neuron density
Patience = 3
Epochs = 20 (13 used)

The result of 54% accuracy on the test set, 54% on the validation, which is better than random guessing, however it is barely guessing more than half successfully indicating that it is struggling.
Training accuracy only reaches about 6%, and the training loss levels out at about 1.13. Due to this low training performance the model is underfitting as it doesn't have sufficient capacity or enough information to learn teh training data properly. 
Therefore it stopped at epoch 12 as the validation stopped improving around epoch 9.

The model performs the worst at guessing cats and could accurately guess roughly 10% of them and many were misclassified as dogs, this is likely due to the very low resolution images and that they share very similar characteristics. Dogs are also slightly over represented in the data set so if the model needs to guess then it is more likely to guess dog. 

The model managed to successfully guess many spiders successfully, however this was likely only due to the fact that it predicted spider for many of the images. It likely did this as the spider's distinctive features survived the downscaling relatively well and the model overly relied on those patterns. 

The data set is also unbalanced with dog and spiders appearing many more times thn other animals, this unbalanced dataset could lead to a bias towards those 2 animals as it guesses them more often or if it is not sure as statistically they are the most likely what it is. However ,by doing that the model is memorising the data set and not actually learning how to identify the animal from its features.  

The main problem with this model is that it is too simple as the network only has 2 convolution layers with 16 filters, which is not deep enough to effectively capture complex features. The downscaling process also removes most of the animals key features. To improve the model, the next iteration will increase the image dimensions to 64,64 and increase the 

Version 2:

Key Characteristics:
Version 2
64,64 image resolution
32 batch size 
32 and 64 filter model with 128 neuron density
Patience = 3
Epochs 20 (9 used)

Increasing the image size and convolution layer filters has increased the test accuracy from 56% to 62%, the validation accuracy to 61% and the training accuracy has reached 82%. The cat accuracy has also increased to 24%. All this means that th changes were positive and that the model is moving in the right direction. 

The problem now is overfitting, the model is now memorising the data set as opposed to learning the general traits of each animal needed to identify them. Around epoch 3-5 the training loss keeps dropping toward 0.6, however the validation loss bottoms out at 1.16 and then starts climbing back up. The training and validation accuracy shows a similar story,with training at 82% while validation is only at 62%

The next iteration to combat these problems I will introduce a 3rd convolution layer to aid in picking out finer details, the training set data will also show the model images from many different angles to prevent it from memorising the data set. 

Version 3: 

Key Characteristics:
64,64 image resolution
32 batch size 
32 and 64 and 128 filter model with 128 neuron density 
Patience = 5
Also added a training generator to change the viewing angles of the images
Batch normalisation was added
Epochs 20 (17 used)

All the changes have increased the model's accuracy: The training has reached 69%, the validation 64% and the test 65%. 
These values indicate that the model being used is in a healthy position and that the changes made have had a positive impact. Having such a small gap between the training and test accuracy indicates that the model is learning the features of the animals and not memorising the dataset itself like previous iterations. 

At this stage the main bottleneck preventing higher accuracy is likely the image quality, at the current level of detail one 4 legged animal likely looks practically identical to another. The next iteration will increase it to 128 to help make finer details more obvious to help distinguish physiologically similar animals.

Version 4:

Key Characterisics:
128,128 image resolution
batch size 32
32, 64 and 128 filter model with 128 neurons
patience 5 
Has batch normalisation 
training generator which creates different views of the images for training 
Epochs 20 (14 used)

Unfortunately it seems adding resolution and increasing the image size to 128,128 has had the opposite effect of what was intended. The training accuracy has dropped down to 50%, the validation down to 48% and the test down to 49%. 

The massive increase in trainable connections means the dense layer cannot map out the logic correctly, with so many parameters and a small dataset the model either overfits or stalls out. A 3 layer network using small 3x3 filters can see a large percentage of a 64x64 at once, however for a 128,128 the layers are too shallow. 

Therefore a 4th convolution layer will be added and the neuron density increased to 256 to help map out the increased number of trainable connections

Version 5: 

Key Characteristics: 
128,128 image size 
batch size 32
32, 64, 128 and 256 filter model with 256 neurons
patience 5 
Has batch normalisation 
has training generator which creates different views of the images for training 
Epochs 20 (20 used)

Adding the extra convolution layer and increasing the neuron density has completely reversed the results of increasing the resolution to 128,128 and accuracy overall shows a massive improvement. The Training has increased to 85%, validation to 78% and the test to 79% these are by far the best results obtained so far and only having a 6% gap between the test and training data is also very good. 

Final changes to make to the model would be increasing the data augmentation and balancing the dataset 


5) Predictions

The test, validation and training accuracy (weighted average f1-score) for each model:

Version 1 - 54% , 54% , 61% 
Version 2 - 62% , 61% , 82% 
Version 3 - 65% , 64% , 69% 
Version 4 - 49% , 48% , 50% 
Version 5 - 79% , 78% , 85% 

Jupyter notebook structure:

Each separate version has its own code section, with a markdown above it explaining:
Each version's key characteristics 
An analysis of the results, what might have caused them and what to do to improve the model
What I plan on changing in the next iteration of the model

Future Work:

If I had more time or access to more resources, I would like to try increasing the resolution further and also increasing the number of convolution layers and their filters as well as the amount of neurons used. I believe this would allow me to increase the accuracy of my model even further as many animals have small and detailed features that can get damaged or removed entirely when the images are downscaled. 

This approach would have been completely unpractical in my current circumstances as version 5 took about 50 minutes to run on my laptop and would have taken even longer if running it off of github. Increasing the detail of the model and increasing the trainable parameters would have resulted in a model that does not finish in a reasonable length of time, if it even finished at all. 

Another thing that I would have liked to include would be dataset balancing, as dogs and spiders appeared in the data set many more times than cats and elephants. This imbalance can lead to biases on the model's part and due to this it would be more likely to guess spider or dog if it is not sure as statistally is is more likely to be one of those. However, doing it that way then the model does not learn how to recognise the animals only teh dataset, balancing the dataset wold have hopefully fixed this bias. 

Libraries and modules:

kagglehub - kagglehub provides a simple way to interact with kagglehub resources such as datasets, models and notebook outputs in python. The kagglehgub library is used in this project to download the dataset from kaggle 

Splitfolders - library used to automatically split the dataset into distinct data sets whi keeping the subfolder category intact. I used it in my code to splt the single raw image director into 3 clean folders: train 70%, val(15%) and test(15%)

OS - It provides a way to interact directly with the operating system, allowing you to manipulate file paths, list folder contents, and configure system variables.
In the script it was used to silence background warnings from tensorflow and to read and print out the raw animal names in the directory

Tensorflow and keras - AN open-source machine learning platform. Keras is the high-level API inside TensorFlow that buildS, compileS, and trainS deep learning networks with simple, readable layers. In my code it was used to:
scale image pixels to [0,1] and pply image modifications 
used to construct my CNN layer by layer 
runs model.predict to generate probabilities on unseen images

NumPy - The fundamental package for scientific computing in Python. It handles large, multi-dimensional arrays and matrices, along with an extensive collection of high-level mathematical functions. It is used to:
Data validation: in check_images verifies if incoming inputs are Numpy array and calculates min/max pixel boundaries 
EDA counting: In labelcount parses data blocks and counts how many images exist per animal type 
Class extaraction: runs np.argmax() to scan across the raw probability arrays and extract the index of the highest prediction value.

Pandas - A data manipulation tool built around DataFrames, used to:
Frequency tracking: initialises an mpty structrured table to compile the visual label distributio log. 
History logging: After training finishes, it wraps the raw dictionary logs returned by Keras into a structured dataframe pd.DataFrame(history.history) for easy column plotting.

Maplotlib - A comprehensive library for creating static, animated, and interactive visualizations in Python. It acts as the structural canvas for almost every graph drawn in the script. Used to:
Image Inspection: Used during EDA (plt.imshow) to physically render raw pixel arrays on the screen, add labels, and strip out background graph ticks.
Performance Curves: Used right at the end of the script to set layout sizes and display the finalized training vs. validation loss and accuracy line graphs.

Seaborn - A statistical data visualization library built directly on top of Matplotlib. It provides an easier, high-level interface for drawing highly attractive, polished charts. Used to: 
Styling: Overhauls global plotting aesthetics using sns.set_style('white')
EDA Bar Charts: Generates the color-coded bar chart (sns.barplot) to visually contrast label quantities across the train, validation, and test datasets.
Confusion Matrix Formatting: Generates the color-mapped grid layouts (sns.heatmap) used to draw the training, validation, and test performance matrices.

Scikit-Learn - A robust machine learning library focused on data mining, modeling, and statistical evaluation tools. USed to:
confusion_matrix calculates the raw mathematical grid cross-referencing true values against predicted classes.
classification_report calculates the precision, recall, and F1-score for every one of the 10 animal categories. your


Acknowledgements:

AI was used to aid in identifying errors given. I also used it to work out how to get this code to run locally on my GPU 


Conclusions:

The final model can predict the animal from an image with an accuracy of 79%. I consider this to be a success as I have managed to increase the accuracy by 25%, from 54% in the first version to 79% in the final. If i had more time or a more powerful computer I would have liked to push the resolution and convultiuon layers further. 

My main takeaway from this is that you need to increase the depth of the convolution layers in conjuction with increasing resolution. As if you just increase resolution too far without also inreasing convolution depth then accuracy suffers(as demonstrated by version 4) and if you also just increase the convolution depth and not the resolution then you get diminshing returns, as the image is to blurry and indistinct for the increased depth to pick out features and have an effect.