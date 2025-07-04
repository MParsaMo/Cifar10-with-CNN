# CIFAR-10 Convolutional Neural Network 🧠🖼️

This repository contains a Convolutional Neural Network (CNN) implementation using Keras to classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). CIFAR-10 consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

---

The CIFAR-10 dataset
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here are the classes in the dataset, as well as 10 random images from each:
airplane										
automobile										
bird										
cat										
deer										
dog										
frog										
horse										
ship										
truck										

---

## 🚀 Features

- Data normalization and one-hot encoding
- Train-validation split for performance monitoring
- Image data augmentation with rotation, shifting, and flipping
- Deep CNN architecture with Batch Normalization and Dropout
- Early stopping callback to prevent overfitting
- Performance evaluation on test data

## 🧰 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn

## 🧠 Model Architecture

## 🧠 Model Architecture

```text
Input: 32x32x3
└── Conv2D(32) + BatchNorm
└── Conv2D(32) + BatchNorm
└── MaxPooling + Dropout
└── Conv2D(64) + BatchNorm
└── Conv2D(64) + BatchNorm
└── MaxPooling + Dropout
└── Conv2D(128) + BatchNorm
└── Conv2D(128) + BatchNorm
└── MaxPooling + Dropout
└── Flatten
└── Dense(128) + BatchNorm + Dropout
└── Dense(10) [Softmax Output]
```


📊 Training Details
Optimizer: SGD with momentum (0.95)

Loss Function: Categorical Crossentropy

Epochs: Up to 50 with early stopping (patience=10)

Batch Size: 64

Validation Split: 20%

🧪 Results
Final evaluation on test data:

Test Loss: Displayed at runtime

Test Accuracy: Displayed at runtime

📷 Sample Output
Displays the first image from the dataset with its label.
plt.imshow(x_train_full[0])
print(y_train_full[0])


🙌 Contributing
Pull requests are welcome! If you'd like to improve the architecture, optimize training, or enhance documentation, feel free to submit suggestions.


Let me know if you'd like me to tweak the style, add visuals, or set up the repo structure with folders like `/data`, `/models`, and `/notebooks`.
