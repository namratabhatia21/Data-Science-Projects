# Facial Recognition System with Transfer Learning

Using the Inception ResNet V1 model from FaceNet as a base, the system is fine-tuned with a custom dataset for improved accuracy in identifying individuals from both images and live video feeds. The implementation is done using the PyTorch framework, making it accessible and modifiable for further research or practical applications.

## DSAIS Face Recognition Application
In this project, I developed a facial recognition system to recognize classmates from the Masters program.

### Problem Overview
Face recognition problems typically fall into two categories:

**Face Verification**: Building a model to distinguish your face from others. This is a binary classification problem.

**Face Recognition**: Building a model to identify multiple faces. This is a multi-class classification problem.

### Scenario
Imagine a system for the Emlyon building where we want to provide face recognition to allow students to enter the building. The goal is to fine-tune the classification head of FaceNet to classify all your friends.

### Project Phases
Face Detection:
In the first phase, I prepared the datasets by using the **MTCNN** model to create cropped face images from raw images and stored them in new directories.

Face Recognition:
In the second phase, I modified the classifier head of **FaceNet** and trained it on the cropped face images.

I built a Python application that displays recognized faces of DSAIS students on video feeds (or real-time images from the camera).


