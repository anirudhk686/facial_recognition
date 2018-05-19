# facial_recognition
## About
In this project i have built a facial recognition system using convolutional neural network. I have used the VGG-16 Net architecture and this has been implemented using keras library in python.<br>
Here I have described only about the usage and requirements. For implementation details ,please refer to the [project_blog](https://anirudhk686.github.io/facial_recognition/)
## Requirements:
- python (version 3.5 or greater)
- pandas,scipy,matplotlib,shelve
- opencv (version 2)
- TensorFlow (version 1.2 or greater) 
- keras (version 2)
#### Additional downloads:
- vgg net weights - vgg-face.mat - [link](www.vlfeat.org/matconvnet/models/vgg-face.mat)
- this was 1gb file hence have not uploaded on git.
- after downloading it place this in the project main directory.

## Usage:
- after the above steps, place the images you want to recognize in 'known' directory that is present in projects main directory. also label the images with the person's name.
- now run vgg.py. 
- after successful run three more file will appear in the main directory:
    - known_vectors.shlf - cointains the encodings of all the images in known folder.
    - model.json - stores the modified vgg model
    - model.h5 - stores the weights
    the last two files are for faster information retrival which is used in next step
- now place the image to be recognised in unknown directory.
- run compare.py
- this will load the three files created by vgg.py compare the unknown face encoding with the known encodings and return the name of most similar face among those in known directory.
