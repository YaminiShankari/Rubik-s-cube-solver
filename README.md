# Rubik's cube solver
In this project, using SVM (RBF kernel) and Kociemba algorithm the faces of each side of the cube is captured for color prediction and easy step-by-step instructions are given to solve the rubik's cube

1) input_dataset1.py: This file is used to create your own datasets. Place the <color> centered face as per the instructions and change the number of samples you want to take accordingly. This captures the hsv of each of the 9 stickers in the single face which will be stored as .npy files in the dataset folder. This is called the original dataset that will be used for model training.

3) input_dataset2.py: This file is used to create augmented datasets. As there are a particular range of HSV for all the 6 colors, the number of samples specified files are created and stored as .npy files. These are called the augmented dataset which will also be used for model training.

4) train.py: This is where the model is trained using SVM (Support Vector Machines) using RBF Kernel wherein the c and gamma values can be changed accordingly in such a way that the value of c should be less and the valye of gamma to be more. This depends on the mean hsv of your dataset files.

5) color_prediction.py: In this file, the user is requested to show each face of the cube according to the <color> center just to make sure that the user gives each face. Make sure that the lightings and position of the cube is properly placed within the given gridlines, which will be the ROI (Region of Interest). The color prediction is shown to the user so that they can verify if the colors are predicted correctly and only when their consent is given, the capturing of the next face will be held. Once all the faces are predicted, the values are stored in captured_faces.npy (which will be created and saved in the same folder as this file exists in).

6) npy_file_checker.py: This is a simple file for user's convinence to check the colors saved in the captured_faces.npy.

7) solution.py: This is the heart of the project where the captured_faces.npy file is loaded and configured in such a way that it gives the Kociemba algorithm (used to give the optimal solution for solving the Rubik's cube) the expected way the input should be. Once the auto-configuration of the faces are done, the algorithm is applied and easy step-by-step instructions are given to the user and is successfully solved.
