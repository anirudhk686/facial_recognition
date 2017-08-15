---
layout: default
---
# [](#header-2)About

In this project, i have built a facial recognition system using a convolutional neural network. This project was a part of my internship at [NCAOR](http://www.ncaor.gov.in/), India under the guidance of [Mr. Sakthivel Samy](http://www.ncaor.gov.in/profiles/details/14).I have used the VGG-16 Net architecture and this has been implemented using keras library in python.

In this page, i have described the implementation details and references used.
If you want to just know how to use this project and not the details, please refer to project's [readme](https://github.com/anirudhk686/facial_recognition/blob/master/README.md). 

## [](#header-2)Project overview

Facial recognition problem is approached using the following steps:
1. Detect faces in an image - image may contain background which is not needed for face recognition. hence we need to crop out face.
for this, I have used OpenCV's frontal face haar cascade. 
2. Calculate unknown face encoding - this is the heart of the project. details are discussed in next section.
3. Comparing the face encodings - compare the unknown face encoding with all the known encodings and return the name of the most similar face encoding.

## [](#header-2)Encoding Faces

What we need is a way to extract a few basic measurements from each face. Then we could measure our unknown face the same way and find the known face with the closest measurements. For example, we might measure the size of each ear, the spacing between the eyes, the length of the nose, etc. But we do not know which of the measurements exactly encode a face. Hence we use Convolutional Neural Network which can automatically learn the encoding from the given image.

For this I have followed the approach as suggested in this paper:

>Deep face recognition, O. M. Parkhi and A. Vedaldi and A. Zisserman, Proceedings of the British Machine Vision Conference (BMVC), 2015 [paper](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf).

Here is the brief discussion of the method proposed in the above paper.<br>
The basic objective is to obtain a network which outputs similar feature vectors for faces of the same person.Based on this objective function the authors have proposed to train the Convolutional network with the help of training set that contains three face images at a time. However, two of the three images is of the same person where as third image is of a different person. In order to train the network to give similar encoding for the first two images and a different one for the third image, they have proposed a triplet loss function. This loss function aims to minimize the distance between first two images and maximize distance from the third image.

## [](#header-2)VGG-16 Net 
The convnet architecture used here is VGG-16 as shown below:<br>
<img src="https://raw.githubusercontent.com/anirudhk686/facial_recognition/master/images/vgg16.png" width="300" height="600">
<br>
[image source](http://book.paddlepaddle.org/03.image_classification/)

## [](#header-2)Implementation
* Keras library with TensorFlow backend has been used to implement the above network.however due device constraints I could to train the network. Hence I used the trained weights after some modifications. the weights were downloaded from [here](http://www.vlfeat.org/matconvnet/pretrained/#face-recognition).They had also used the same procedure as described above. 
* But the downloaded weights were trained to identify 2622 pre-defined specific faces and hence I had to generalize it. For this, i removed the last softmax layer and now the network would give out 2622 dimension feature vector for each face. since the network was trained using the triplet loss function it would output similar feature vector for faces of the same person. 
* Now all the faces to be recognized in future are placed in a folder labeled with the person's name. The feature vectors for all those faces are obtained by passing them through the modified network. These feature vectors are stored in a dictionary along with their person names.
* Later when need to recognized a face, its feature vector is generated and then compared with the known feature vectors.Name of the most similar vector is obtained. for this, i have used cosine similarity from scipy library.

>### Code
* The entire project is implemented in Python and available [here](https://github.com/anirudhk686/facial_recognition).
* the images to be recognized later must be placed in the known folder.
* the code to generate vectors of the known folder is present in vgg.py.
* the code to compare unknown face with known is present in compare.py.
* Image pre-processing and code to crop out the faces is implemented in both files hence any image containing one face can be passed.

## [](#header-2)Usage
Information on code usage and requirements is avaliable in project [readme](https://github.com/anirudhk686/facial_recognition/blob/master/README.md).

## [](#header-2)References
Apart from the paper mentioned above i have used the following resources:
* Facenet: A unified embedding for face recognition and clustering,F Schroff, D Kalenichenko, J Philbin - Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,2015.[paper](https://arxiv.org/abs/1503.03832)
* Stanford [course](http://cs231n.stanford.edu/) on convnets.
* [Blog](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78) by Adam Geitgey.
* [Blog](https://aboveintelligent.com/face-recognition-with-keras-and-opencv-2baf2a83b799) by M.Zaradzki.
