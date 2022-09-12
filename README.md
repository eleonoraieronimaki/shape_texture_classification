# Shape and Texture features for Classification

### Description 
In this repository, we are analysing the shape and texture features of embryos of a mutant of the plant Arabidopsis thaliana in order to assess a combination of them that performs well in a classification.

In the set of images provided, there are about three classes that express the mutant form differently which can be expressed by their size and shape. In order to classify the embryos as accurately as possible, various image processing and clustering techniques have been tested.
The images shown below are taken from the dataset as an example. The blue areas in the embryo represent the expression of one particular gene; in some cases, it is abundantly present while in other the gene expression seems to be absent.

<p float="left">
<img src="embryo1.png" width="400" height="300">
<img src="embryo2.png" width="400" height="300">
</p>

The main steps followed in order to achieve this are: 
- ## Segmentation and Preprocessing : 
Develop and apply a segmentation technique to prepare the embryos for further processing. 

- ## Shape feature:
Compute the size and shape for the embryos in all images; including the amount of gene-expression
- ## Texture feature:
Compute the texture (standard deviation and uniformity )  in the blue part as well as in the remaining part of the embryo 
- ## HOG feature: 
Evaluation of a handcrafted feature: Histogram of Oriented Gradient (HOG) of each the embryos for just the gene-expression part
as well as for the whole for the whole embryo
- ## Classification:
Use of Support Vector Machine (SVM) to classify the images so as to see how well the classes can be separated on the basis of the features determined:  HOG, Texture and Shape features
