import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#for GPU
from tensorflow.keras import mixed_precision

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print(tf.__version__)

# Define your image dimensions and number of classes
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 128, 256, 3  # Resized dimensions for input
NUM_CLASSES = 34  # Cityscapes has 34 classes

# Visualize the prediction
def visualize_prediction(image, label, prediction):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Display the input image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    # Display the true label
    axes[1].imshow(np.argmax(label, axis=-1), cmap='nipy_spectral')
    axes[1].set_title("True Label")
    axes[1].axis('off')
    
    # Display the predicted label
    axes[2].imshow(np.argmax(prediction, axis=-1)[0], cmap='nipy_spectral')  # Taking the first prediction
    axes[2].set_title("Predicted Label")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Function to plot training and validation loss and IoU
def plot_training_history(history):
    # Summarize history for loss and accuracy
    plt.figure(figsize=(12, 6))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

def visualize_prediction(image, prediction): #add label to see label too
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Display the input image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    # Display the true label
    #axes[1].imshow(np.argmax(label, axis=-1), cmap='nipy_spectral')  # Use proper colormap for segmentation
    #axes[1].set_title("True Label")
    #axes[1].axis('off')
    
    # Display the predicted label
    axes[1].imshow(np.argmax(prediction, axis=-1)[0], cmap='nipy_spectral')  # Taking the first prediction
    axes[1].set_title("Predicted Label")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def preprocess_image_label(image_path, label_path):
    # Read and decode the image from the file path tensor
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=IMG_CHANNELS)  # Assuming RGB image
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0  # Normalize to [0, 1] range
    
    # Read and decode the label from the file path tensor (grayscale)
    label = tf.io.read_file(label_path)
    label = tf.image.decode_image(label, channels=1)  # Grayscale image for segmentation masks
    label = tf.image.resize(label, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    # One-hot encode the label
    label_one_hot = tf.one_hot(tf.squeeze(label, axis=-1), depth=NUM_CLASSES, axis=-1)

    #Get unique labels from one-hot encoded labels
    #unique_labels = tf.unique(tf.reshape(tf.argmax(label_one_hot, axis=-1), [-1]))[0]

    #print(f"Processed image shape: {image.shape}, Label shape: {label_one_hot.shape}, Unique labels: {unique_labels.numpy()}")

    return image, label_one_hot


def data_generator(image_paths, label_paths, batch_size):
    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
    dataset = dataset.shuffle(buffer_size=100)
    
    # Apply preprocessing function
    def map_func(x, y):
        image, label = tf.py_function(func=preprocess_image_label, inp=[x, y], Tout=[tf.float32, tf.float32])
        image.set_shape([IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
        label.set_shape([IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES])
        return image, label
    
    dataset = dataset.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

# Load and sort image and label paths for train, val, and test sets
image_paths_train = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk('cityscapes/leftImg8bit/train') for f in filenames if f.endswith('.png')])
label_paths_train = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk('cityscapes/gtFine/train') for f in filenames if f.endswith('labelIds.png')])

image_paths_val = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk('cityscapes/leftImg8bit/val') for f in filenames if f.endswith('.png')])
label_paths_val = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk('cityscapes/gtFine/val') for f in filenames if f.endswith('labelIds.png')])

image_paths_test = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk('cityscapes/leftImg8bit/test') for f in filenames if f.endswith('.png')])
label_paths_test = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk('cityscapes/gtFine/test') for f in filenames if f.endswith('labelIds.png')])

# Set batch size and create the ttrain, val, and test datasets
batch_size = 64
train_dataset = data_generator(image_paths_train, label_paths_train, batch_size).repeat()
val_dataset = data_generator(image_paths_val, label_paths_val, batch_size).repeat()
test_dataset = data_generator(image_paths_val, label_paths_val, batch_size)

inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

c1 = tf.keras.layers.Conv2D(4, (5, 5), kernel_initializer='he_normal', padding='same')(inputs)
#4 -> number of filters (kernels), 5,5 -> kernel size matrix, kernel init. -> starting values of weights, normal distribution centered around zeros, same -> output has the same size as the input
c1 = tf.keras.layers.BatchNormalization()(c1)  # Add Batch Normalization
c1 = tf.keras.layers.Activation('relu')(c1)  # Activation after Batch Norm
c1 = tf.keras.layers.Dropout(0.1)(c1) #to prevent overfitting %10
c1 = tf.keras.layers.Conv2D(4, (5, 5), kernel_initializer='he_normal', padding='same')(c1)
c1 = tf.keras.layers.BatchNormalization()(c1)
c1 = tf.keras.layers.Activation('relu')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1) #to keep the highest value from every 2x2, resulting reducing the size

# Second block
c2 = tf.keras.layers.Conv2D(8, (5, 5), kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.BatchNormalization()(c2)
c2 = tf.keras.layers.Activation('relu')(c2)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(8, (5, 5), kernel_initializer='he_normal', padding='same')(c2)
c2 = tf.keras.layers.BatchNormalization()(c2)
c2 = tf.keras.layers.Activation('relu')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

# Third block
c3 = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.BatchNormalization()(c3)
c3 = tf.keras.layers.Activation('relu')(c3)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c3)
c3 = tf.keras.layers.BatchNormalization()(c3)
c3 = tf.keras.layers.Activation('relu')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

# Fourth block
c4 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.BatchNormalization()(c4)
c4 = tf.keras.layers.Activation('relu')(c4)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c4)
c4 = tf.keras.layers.BatchNormalization()(c4)
c4 = tf.keras.layers.Activation('relu')(c4)
p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

# Fifth block
c5 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.BatchNormalization()(c5)
c5 = tf.keras.layers.Activation('relu')(c5)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c5)
c5 = tf.keras.layers.BatchNormalization()(c5)
c5 = tf.keras.layers.Activation('relu')(c5)

# Going upwards
u6 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
#transpose conv2D -> to increse dimension opposite of conv2D(upsample), 128 -> number of filters (kernels), (2,2)-> kernel size, strides-> how much filter moves and means output size will be doubled (upsamples)
u6 = tf.keras.layers.concatenate([u6, c4]) #concatenate to provide richer information by combining features from different stages, channel sizes are added together
c6 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.BatchNormalization()(c6)
c6 = tf.keras.layers.Activation('relu')(c6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c6)
c6 = tf.keras.layers.BatchNormalization()(c6)
c6 = tf.keras.layers.Activation('relu')(c6)

u7 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.BatchNormalization()(c7)
c7 = tf.keras.layers.Activation('relu')(c7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c7)
c7 = tf.keras.layers.BatchNormalization()(c7)
c7 = tf.keras.layers.Activation('relu')(c7)

u8 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(8, (5, 5), kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.BatchNormalization()(c8)
c8 = tf.keras.layers.Activation('relu')(c8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(8, (5, 5), kernel_initializer='he_normal', padding='same')(c8)
c8 = tf.keras.layers.BatchNormalization()(c8)
c8 = tf.keras.layers.Activation('relu')(c8)

u9 = tf.keras.layers.Conv2DTranspose(4, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3) #concatenate along the channel axis, default value '-1' -> last dimension
c9 = tf.keras.layers.Conv2D(4, (5, 5), kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.BatchNormalization()(c9)
c9 = tf.keras.layers.Activation('relu')(c9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(4, (5, 5), kernel_initializer='he_normal', padding='same')(c9)
c9 = tf.keras.layers.BatchNormalization()(c9)
c9 = tf.keras.layers.Activation('relu')(c9)

outputs = tf.keras.layers.Conv2D(34, (1, 1), activation='softmax')(c9)  #1 -> 34 feature maps for 34 classes, (1,1)-> pointwise convolution, sigmoid -> binary classification


model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #adam -> backpropagation algorithm, loss -> what optimizer wants to minimize cathegorical for multiclasses, metric -> to measure model
model.summary()

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_autonomus_driving.keras', verbose=1, save_best_only=True)
#filepath to save model, verbose will write a messageeach time model is being saved, save model if only it improves
callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=6, monitor='val_loss'),
  #  tf.keras.callbacks.TensorBoard(log_dir='logs') #!tensorboard --logdir=logs/ --host localhost --port 8088 
    #copy this to console to see the graphics of the training process
]

steps_per_epoch = len(image_paths_train) // batch_size
validation_steps = len(image_paths_val) // batch_size


# Training the model with train and validation datasets
history = model.fit(train_dataset, 
                    validation_data=val_dataset, 
                    steps_per_epoch=steps_per_epoch, 
                    validation_steps=validation_steps, 
                    epochs=17, 
                    callbacks=callbacks)

# Evaluate the model on the test dataset
test_results = model.evaluate(test_dataset)
print(f"Test results - Loss: {test_results[0]}, Accuracy: {test_results[1]}")

'''
def visualize(label_path):
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(label, cmap='nipy_spectral')
    plt.title("Raw Label Image")
    plt.axis('off')
    plt.show()
'''

for idx in [0, 10, 20, 40, 65, 80, 100, 120]:  # Adjust indices to test different images   
    test_image, test_label = preprocess_image_label(image_paths_test[idx], label_paths_test[idx]) 
    print(f"Testing image: {image_paths_test[idx]}, Label: {label_paths_test[idx]}")
    print(f"Unique labels for Test Image: {np.unique(np.argmax(test_label, axis=-1))}")
    prediction = model.predict(np.expand_dims(test_image, axis=0))
    visualize_prediction(test_image, prediction)

# Plot the training history
plot_training_history(history)
