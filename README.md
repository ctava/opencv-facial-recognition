# opencv-facial-recognition

If you have xcode installed, this program should run out of the box.
What it does is:

- train a model based on 1.jpg and 99.jpg (see the generated file: ./1-trainedModel.yml)
- load that model and determine that identity for 1.jpg is 1, rather than 99.

You should see the following output:

  Compiled with OpenCV version 3.1.0

  preprocessFacesAndSaveModel 
  Finished training.

  loadModelAndRecognizeFace 
  Predicted identity = 1.

  Program ended with exit code: 0
