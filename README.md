# OCR data classification 

since dataset is balanced we can visualize tsne distribution on two dimension to check the over complexity of the data and in-class interference.

![test](https://i.ibb.co/0hgZTj1/photo-2022-11-08-18-34-18.jpg)


## ML approach

As typical machine learning task I used only **xgboost** on different variations of features:
| feature | acc  |
|--|--|
| raw images (100x300) | 0.85 |
|pojections (300) |0.85|
|pca (250)|0.87|
|tsne(only 2 dims)|0.75|
|hog (1500 dims)|**0.965**|

 - we can observe that the behavior of raw images and histogram projections is quite the same due to the fact that the classification is based on the fact that the similarity of spaces between letters is highly correlated in different members of the same class. Hence histograms tend to look similar in the same class: **drop histogram of two different images from the same class here**  
 - we can see that **hog** features outperform the rest with arelatively  low  computational complexity. 
 ![enter image description here](https://i.ibb.co/0hgZTj1/photo-2022-11-08-18-34-18.jpg)


## ML approach
we trained 2 CNNs and a vision transformer on on different variety of image transformations: 
Note: all models are initialized with **imagenet** weights and optimized whit **adabelief** optimizer on cuda gpus.

| transformation-model | acc  |
|--|--|
| raw_images (33x100) - resent 18 | 0.996 |
| raw_images (33x100) - efficientNetB0 | 0.9982 |
| raw_images (224x224) - efficientNetB0 | **1** |
| raw_images (224x224) - ViT small 16 | 0.975 |

 - we can see that although multihead attention mechanism is way more efficient than convolution cnns outperform the vision transformer due to the fact that transformers require large amounts of data.
 - we can see that the efficientNet on square images outperformed the one with images of shape 33x100 due to the fact that the the actual model was pretrained on imagenet dataset in which images have square shape.
 -  visualisation of the efficientnet features on a 2d space: 
 ![enter image description here](https://i.ibb.co/zR2BmXL/effcient-Netb0-features.png)
 - Learning curve of the efficientnet b0
 ![enter image description here](https://i.ibb.co/PDcBgxt/photo-2022-11-08-18-36-39.jpg)
 - confusion matrix
 
 ![enter image description here](https://i.ibb.co/3hCdc5B/photo-2022-11-08-18-36-43.jpg)


## Over all
![enter image description here](https://i.ibb.co/hcFC9sv/summary.png)


## One more thing we can try is binarizing the images
