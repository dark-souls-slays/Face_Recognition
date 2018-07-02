<b>Claudia Espinoza, Asunción Martínez <br>
National Dong Hwa Univerisity <br>
Final Project - Machine Learning </b>

------------------------------------------------------------------------------------------------------------------------------------------
<br>
Image Classification System

Data consists of 650 images, 13 images for each different person. 
Training -> 10 images * 50 different faces
Testing -> 3 images * 50 different faces

IMPORT DATA

Read directory "training_data" and extract each image in greyscale. Since, the images are of different sizes, we resize all of them to (128, 128). Y contains an array of  500 labels 1-50, used to identify the person in dataset[x].