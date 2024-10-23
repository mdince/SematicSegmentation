import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#for GPU
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16') 
mixed_precision.set_policy(policy)



print(tf.__version__)

# Define your image dimensions and number of classes
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 256, 512, 3  # Resized dimensions for input
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

# Example function to visualize an image and its label
def visualize_image_and_label(image, label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display the image
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis('off')
    
    # Display the label (using argmax to get class for each pixel)
    axes[1].imshow(np.argmax(label, axis=-1), cmap='nipy_spectral')  # You can change the colormap for better visualization
    axes[1].set_title("Label")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Preprocessing function for image and label
def preprocess_image_label(image_path, label_path):
    image_path = image_path.numpy().decode('utf-8')
    label_path = label_path.numpy().decode('utf-8')
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image / 255.0
    
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.resize(label, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    
    # One-hot encode the labels
    label_one_hot = np.zeros((IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        label_one_hot[:, :, c] = (label == c).astype(int)
    
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
batch_size = 16
train_dataset = data_generator(image_paths_train, label_paths_train, batch_size).repeat()
val_dataset = data_generator(image_paths_val, label_paths_val, batch_size).repeat()
test_dataset = data_generator(image_paths_test, label_paths_test, batch_size)


inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#going downwards
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
#16 -> number of filters (kernels), 3,3 -> kernel size matrix, kernel init. -> starting values of weights, normal distribution centered around zeros, same -> output has the same size as the input
c1 = tf.keras.layers.Dropout(0.1)(c1) #to prevent overfitting %10
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1) #to keep the highest value from every 2x2, resulting reducing the size

c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#going upwards
u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
#transpose conv2D -> to increse dimension opposite of conv2D(upsample), 128 -> number of filters (kernels), (2,2)-> kernel size, strides-> how much filter moves and means output size will be doubled (upsamples)
u6 = tf.keras.layers.concatenate([u6, c4]) #concatenate to provide richer information by combining features from different stages, channel sizes are added together
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3]) 
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2]) 
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3) #concatenate along the channel axis, default value '-1' -> last dimension
c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(34, (1,1), activation='softmax')(c9) #1 -> 34 feature maps for 34 classes, (1,1)-> pointwise convolution, sigmoid -> binary classification


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
                    epochs=2, 
                    callbacks=callbacks)

# Evaluate the model on the test dataset
test_results = model.evaluate(test_dataset)
print(f"Test results - Loss: {test_results[0]}, Accuracy: {test_results[1]}")

# Preprocess and predict for a test image
test_image, test_label = preprocess_image_label(image_paths_test[0], label_paths_test[0])
prediction = model.predict(np.expand_dims(test_image, axis=0))

# Visualize the result
visualize_prediction(test_image, test_label, prediction)

# Plot the training history
plot_training_history(history)


#SADELEŞTİRRRR