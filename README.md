# Face Mask Detection and Segmentation using ML / DL


 - Niranjan Gopal (IMT2022543)
 - Yash Sengupta  (IMT2022532)
 - Teerth Bhalgat (IMT2022586)

 > - Complete Report :- [Download/View PDF Report](./assets/FaceMaskDetection_and_Segmentation_Report.pdf)
 > - All Model weights, checkpoints, Model Traning history :- [Google Drive Link to download weights](https://drive.google.com/drive/folders/155lCKD1fDNq2Kq1ZPGncunSYnTp_jycS?usp=drive_link)

## Introduction
This project aims to develop robust methods for detecting face masks in images and segmenting the mask regions. We research the most state of the art DL methodes and cost effective traditional methodes for this task.
The work is divided into four parts:
1. Binary classification using handcrafted features and machine learning
2. Binary classification using CNN
3. Mask segmentation using traditional techniques
4. Mask segmentation using U-Net architectures

## Dataset

For face-mask detection the dataset consists of facial images labeled as "with mask" or "without mask". Key details:
- Source: 
- Contains RGB images of varying resolutions
- Split into training (80%) and testing (20%) sets

For segmentation tasks
- source:
- 

## Methodology

### Part A: Binary Classification Using Handcrafted Features
1. **Feature Extraction**:
   - Used Histogram of Oriented Gradients (HOG) with:
     - 8 orientations
     - 8x8 pixels per cell
     - 2x2 cells per block
2. **Model Training**:
   - Evaluated 9 classifiers (SVM, Decision Tree, Random Forest, etc.)
   - Performed hyperparameter tuning using GridSearchCV/RandomizedSearchCV
   - Implemented Stacking and Voting ensemble which utilizes top 4 best tuned models from HPO 

### Part B: CNN Classification
1. **Architecture**:
   - 3 convolutional layers (32, 64, 128 filters)
   - Max pooling after each conv layer
   - Dense layer (128 units) with dropout (0.5)
   - Sigmoid output
2. **Training**:
   - Adam optimizer
   - Binary cross-entropy loss
   - 10 epochs, batch size 32

### Part C: Traditional Segmentation
1. **Techniques**:
   - Thresholding (global threshold = 127)
   - Canny edge detection (thresholds = 50, 150)
2. **Evaluation**:
   - Dice Coefficient and IoU metrics
   - Contour extraction for refinement

### Part D: U-Net Segmentation
1. **Architectures**:
   - U-Net and U-Net++ variants
   - Backbones: EfficientNet-b7, ResNet50, VGG19
2. **Training**:
   - Standard U-Net training protocol
   - Evaluated multiple backbones

## Hyperparameters and Experiments

### CNN Classification
- Learning rate: Default Adam (≈0.001)
- Batch size: 32
- Activation: ReLU (hidden), Sigmoid (output)
- Dropout: 0.5

### U-Net Models
| Model    | Backbone     | Key Hyperparameters           |
|----------|--------------|-------------------------------|
| U-Net    | Efficient-b7 | Pretrained weights, Adam opt  |
| U-Net++  | ResNet50     | Pretrained weights, Adam opt  |
| U-Net    | VGG19        | Pretrained weights, Adam opt  |

## Results

### Classification Performance
| Model               | Accuracy |
|---------------------|----------|
| CNN                 | 97.0%    |
| Stacking Classifier | 95.0%    |
| Voting Classifier   | 94.8%    |
| CatBoost            | 94.5%    |

### Segmentation Performance
| Method          | Dice   | IoU    |
|-----------------|--------|--------|
| Thresholding    | 0.2562 | 0.3080 |
| U-Net (Eff-b7)  | 0.9447 | 0.8952 |

![Unet Segmentation](./assets/results_of_best_unet.png)

## Observations and Analysis
1. **Classification**:
   - CNN outperformed traditional ML methods by ~2-9%
   - Ensemble methods provided good alternatives to CNN
2. **Segmentation**:
   - Traditional methods were limited by:
     - Fixed threshold values
     - Sensitivity to lighting/colors
   - U-Net achieved superior results (Dice > 0.94)
3. **Challenges**:
   - Variability in mask colors/textures
   - Occlusions and unusual angles

## How to Run the Code
1. **Requirements**:
```bash
    pip install -r requirements.txt 
```
And run the jupyter notebooks provided.

## Challenges we faced
When working on optimizing machine learning workflows, we encountered several challenges that required methodical troubleshooting to resolve effectively. Below, we share the issues we faced and the steps we took to overcome them.

Dataset too Large
- was not able to load entire 9000 images, the function train_test_split() would not work
- This causes RAM Overflow and Linux Kernal automatically kills the process


We experienced a classic case of overfitting and being stuck at a local minima while gradient descent :
- Training metrics (dice_coef) steadily improve from 0.8269 to 0.9186
- Meanwhile, validation metrics hover around 0.81-0.82 and don't improve
The learning rate reduction at epochs 8 and 13 didn't help break through the platea
![Overfilling when we trained on entire Dataset](./assets/OverFitting.png)

```log
Epoch 6: val_dice_coef did not improve from 0.81955
939/939 ━━━━━━━━━━━━━━━━━━━━ 137s 146ms/step - binary_accuracy: 0.9188 - dice_coef: 0.8756 - iou_coef: 0.7745 - loss: 0.1244 - val_binary_accuracy: 0.8772 - val_dice_coef: 0.8126 - val_iou_coef: 0.6868 - val_loss: 0.1873 - learning_rate: 1.0000e-04
Epoch 7/50
939/939 ━━━━━━━━━━━━━━━━━━━━ 0s 140ms/step - binary_accuracy: 0.9260 - dice_coef: 0.8871 - iou_coef: 0.7922 - loss: 0.1129
Epoch 7: val_dice_coef did not improve from 0.81955
939/939 ━━━━━━━━━━━━━━━━━━━━ 140s 149ms/step - binary_accuracy: 0.9260 - dice_coef: 0.8871 - iou_coef: 0.7922 - loss: 0.1129 - val_binary_accuracy: 0.8767 - val_dice_coef: 0.8161 - val_iou_coef: 0.6913 - val_loss: 0.1838 - learning_rate: 1.0000e-04
Epoch 8/50
939/939 ━━━━━━━━━━━━━━━━━━━━ 0s 140ms/step - binary_accuracy: 0.9319 - dice_coef: 0.8964 - iou_coef: 0.8065 - loss: 0.1036
Epoch 8: val_dice_coef did not improve from 0.81955

Epoch 8: ReduceLROnPlateau reducing learning rate to 9.999999747378752e-06.
939/939 ━━━━━━━━━━━━━━━━━━━━ 140s 149ms/step - binary_accuracy: 0.9319 - dice_coef: 0.8964 - iou_coef: 0.8065 - loss: 0.1036 - val_binary_accuracy: 0.8765 - val_dice_coef: 0.8152 - val_iou_coef: 0.6904 - val_loss: 0.1848 - learning_rate: 1.0000e-04
Epoch 9/50
939/939 ━━━━━━━━━━━━━━━━━━━━ 0s 140ms/step - binary_accuracy: 0.9383 - dice_coef: 0.9060 - iou_coef: 0.8214 - loss: 0.0940
Epoch 9: val_dice_coef did not improve from 0.81955
939/939 ━━━━━━━━━━━━━━━━━━━━ 140s 149ms/step - binary_accuracy: 0.9383 - dice_coef: 0.9060 - iou_coef: 0.8214 - loss: 0.0940 - val_binary_accuracy: 0.8788 - val_dice_coef: 0.8145 - val_iou_coef: 0.6900 - val_loss: 0.1855 - learning_rate: 1.0000e-05
Epoch 10/50
939/939 ━━━━━━━━━━━━━━━━━━━━ 0s 139ms/step - binary_accuracy: 0.9415 - dice_coef: 0.9105 - iou_coef: 0.8292 - loss: 0.0895
Epoch 10: val_dice_coef did not improve from 0.81955
939/939 ━━━━━━━━━━━━━━━━━━━━ 139s 148ms/step - binary_accuracy: 0.9415 - dice_coef: 0.9105 - iou_coef: 0.8292 - loss: 0.0895 - val_binary_accuracy: 0.8786 - val_dice_coef: 0.8136 - val_iou_coef: 0.6891 - val_loss: 0.1864 - learning_rate: 1.0000e-05
Epoch 11/50
939/939 ━━━━━━━━━━━━━━━━━━━━ 0s 138ms/step - binary_accuracy: 0.9433 - dice_coef: 0.9136 - iou_coef: 0.8350 - loss: 0.0864
Epoch 11: val_dice_coef did not improve from 0.81955
939/939 ━━━━━━━━━━━━━━━━━━━━ 138s 147ms/step - binary_accuracy: 0.9433 - dice_coef: 0.9136 - iou_coef: 0.8350 - loss: 0.0864 - val_binary_accuracy: 0.8787 - val_dice_coef: 0.8139 - val_iou_coef: 0.6892 - val_loss: 0.1861 - learning_rate: 1.0000e-05
Epoch 12/50
939/939 ━━━━━━━━━━━━━━━━━━━━ 0s 140ms/step - binary_accuracy: 0.9458 - dice_coef: 0.9171 - iou_coef: 0.8416 - loss: 0.0829
Epoch 12: val_dice_coef did not improve from 0.81955
939/939 ━━━━━━━━━━━━━━━━━━━━ 140s 149ms/step - binary_accuracy: 0.9458 - dice_coef: 0.9171 - iou_coef: 0.8416 - loss: 0.0829 - val_binary_accuracy: 0.8785 - val_dice_coef: 0.8145 - val_iou_coef: 0.6896 - val_loss: 0.1854 - learning_rate: 1.0000e-05
Epoch 13/50
939/939 ━━━━━━━━━━━━━━━━━━━━ 0s 139ms/step - binary_accuracy: 0.9470 - dice_coef: 0.9186 - iou_coef: 0.8438 - loss: 0.0813
Epoch 13: val_dice_coef did not improve from 0.81955

Epoch 13: ReduceLROnPlateau reducing learning rate to 9.999999747378752e-07.
939/939 ━━━━━━━━━━━━━━━━━━━━ 139s 148ms/step - binary_accuracy: 0.9470 - dice_coef: 0.9186 - iou_coef: 0.8438 - loss: 0.0813 - val_binary_accuracy: 0.8782 - val_dice_coef: 0.8143 - val_iou_coef: 0.6897 - val_loss: 0.1858 - learning_rate: 1.0000e-05
```



CUDA Out of Memory Errors
- One of the initial challenges was managing CUDA memory limitations, which often occurred during model training on large datasets. This issue disrupted the training process and required immediate attention. To address it:
- Reduced Batch Size: we decreased the batch size to --batch_size 16, which lowered memory consumption without compromising training efficiency.
![Memory Problems ](./assets/CudeMemOverflow_WHILE_TRAINING.png)

Poor Segmentation Results
- While implementing segmentation models, particularly U-Net, we observed suboptimal performance in the form of inaccurate outputs and poor prediction quality. To improve results:
- Adjusted Threshold Values: Fine-tuning the threshold parameters allowed better discrimination between segmented regions.
- Experimented with Different Backbones for U-Net: Switching to alternative architectures enhanced feature extraction and improved overall accuracy.
- Ensured Proper Image Normalization: Correct normalization processes were crucial to maintaining consistent input quality, ultimately boosting model performance.

Problems while Model Saving and Inferring from Trained Models
- Another significant challenge arose when saving trained models and using them for inference, especially in cases involving custom architectures. These issues impacted the reproducibility and usability of the trained models. Here's how I resolved them:
- Model Saving: Ensured consistent serialization and deserialization by using reliable libraries such as torch.save() and torch.load() for PyTorch models. I also checked for version compatibility between the training and deployment environments.
- Inference Issues: Carefully exported the model to the appropriate format (e.g., ONNX or TensorFlow SavedModel) for compatibility with deployment platforms. This helped mitigate discrepancies during the inference phase.
- Dependency Conflicts: Verified that all required dependencies were correctly installed and aligned with the framework versions used during training. This step minimized errors due to mismatches in libraries.
- Input Shape Validation: To prevent runtime errors, I ensured that the input data during inference matched the expected shapes and preprocessing steps used during training.

Had to understand models from Tensorflow (Keras Application) in order to use them as a backbone for Unets
- Tensor reshaping because default BackBone models from Tensorflow had different spatial dimentions
- how keras Resize class was deprecated 

Installation Issues
- Setting up the environment presented difficulties due to conflicting dependencies or outdated versions.
- Pip Vs Conda: Tensorflow officially only releases on PyPi channels and does not support conda; meanwhile Ubuntu 24.04 had strange policies regarding system-wide python installation and suggested usage of conda.

## Important Advanced things that we are trying 

- GPU enabled training of scikit learn models ( according to 2025 Google's latest Colab ; Cuda now supports GPU acceleration for all Scikit learn traditional ML model :- RandomForest, etc )
- ABBANet ( the author of MSFD dataset provided a model that out-performs every other model )
- Google drive link of all the train model to run inference on. ( or provide a Docker Image for reporducing what we did )
