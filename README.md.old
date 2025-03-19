# VR_Project1_Niranjan-GopaL_IMT2022543
Mini Project


# EXPLORE A LOT ( cuz we got the freedome to, cuz of LLMs )


For DL :-
Hyperparameters :-
- Number of epochs
- Learning Rate
- Batch size
- Optimizer ( Adam, SGD, RMSprop, Adagrad, Adadelta, Ftrl, AdamW, Nadam  )
- Initial Learning Rate
- Number of Layers, Number of Nuerons in a layer ( if you are not using ResNet, FasterRCNN, etc)
- Activation function in the classification layer



0. Verify if using hog() in ML part where sir asked to do "Hand crafted feature" :- is this exactl what sir meant ?
1. tf and pytorch Both version are tried
2. What each hyper parameter resulted in 



## These are the Datasets THAT ARE STANDARD BENCHMARKS for ML models

Catboost argued it's performance superiority via these DatasetBenchmarks against LightGBM, XGBoost, H20.ai

- Adult
- Amazon
- Click prediction
- KDD appetency
- KDD churn
- KDD internet
- KDD upselling
- KDD 98
- Kick prediction



## Had to manually compile LightGBM from source code ( compile with flags ) in order to use GPU

```sh
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build
cd build
cmake -DUSE_GPU=1 ..
make -j4
cd ../python-package
python setup.py install
```

## 

However, you're now seeing many warnings about "No further splits with positive gain, best gain: -inf". 
This indicates that your model is unable to find meaningful splits in the data, which suggests potential issues with:
- Overfitting
- Imbalanced classes (which you have: 808 positive vs 295 negative)


Modification made :-
```py
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=100,           # Reduced from 200
        max_depth=4,                # Reduced from 6 
        learning_rate=0.05,         # Reduced from 0.1
        min_child_samples=20,       # Minimum samples in a leaf
        subsample=0.8,              # Use 80% of data for trees
        colsample_bytree=0.8,       # Use 80% of features per tree
        class_weight='balanced',    # Handle class imbalance
        reg_alpha=0.1,              # L1 regularization
        reg_lambda=0.1,             # L2 regularization
        random_state=42,            # For reproducibility
    ),
```


# Question D 
Make U-Net
Try to make ABANet (research paper that used MSFD)
> https://github.com/sadjadrz/ABANet-Attention-boundary-aware-network-for-image-segmentation





# Question C
