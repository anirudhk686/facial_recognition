---
layout: default
---
# [](#header-2)About

In this project i have built a facial recognition system using convolutional neural network. This project was a part of my internship at [NCAOR,India](http://www.ncaor.gov.in/).I have used the VGG-16 Net architecture and this has been implemented using keras library in python.

In this page i have described the implementation details and references used.
If you want to just know how to use this project and not the details, please refer to project's [readme](https://github.com/anirudhk686/facial_recognition/blob/master/README.md) 

## [](#header-2)Project overview

facial recognition problem is approached using the following steps:
1. Detect faces in an image - image may contain background which is not needed for face recognition. hence we need to crop out face.
for this i have used opencv's frontalface haarcascade. 
2. Calculate unknown face encoding - this is the heart of the project. details are disscussed in next section.
3. Comparing the face encodings - compare the unknown face encoding with all the known encodings and return the name of the most similar face encoding.

## [](#header-2)Encoding Faces

What we need is a way to extract a few basic measurements from each face. Then we could measure our unknown face the same way and find the known face with the closest measurements. For example, we might measure the size of each ear, the spacing between the eyes, the length of the nose, etc. But we do not know which of the measurements exactly encode a face. Hence we use Convolutional Neural Network which can automatically learn the encoding from the given image.

For this i have followed the approach as suggested in this paper:

>Deep face recognition, O. M. Parkhi and A. Vedaldi and A. Zisserman, Proceedings of the British Machine Vision Conference (BMVC), 2015 [paper](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf).

Here is the brief discussion of the method proposed in the above paper.
The basic objective is to obtain a network which outputs similar feature vectors for faces of the same person.Based on this objective function the authors have proposed to train the Convolutional network with the help of training set that contains three face images at a time. However two of the three images is of the same person where as third image is of a different person. In order to trian the network to give samilar encoding for the first two images and a differnt one for the third image, they have proposed a triplet loss function. This loss function aims to minimize distance between first two images and maximize distance from the third image.

## [](#header-2)VGG-16 Net
The convnet architecture used here is VGG-16 as shown below:
      <img src="https://raw.githubusercontent.com/anirudhk686/facial_recognition/master/images/vgg16.png" width="300" height="600"><br>
[image source](http://book.paddlepaddle.org/03.image_classification/)





*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.



###### [](#header-6)Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![](https://assets-cdn.github.com/images/icons/emoji/octocat.png)

### Large image

![](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
