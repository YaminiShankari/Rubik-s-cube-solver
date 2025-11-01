# 🧩 Rubik's cube solver
This project uses SVM (RBF kernel) for color prediction and the Kociemba algorithm to generate the optimal solution for solving a Rubik’s Cube.
Each face of the cube is captured through the webcam, analyzed for colors, and the program provides step-by-step instructions to solve it.
   
# 📂 Project Files Overview
1️⃣ input_dataset1.py:

-Used to create your own dataset.    
-Place the cube with the <color> center as instructed.   
-Adjust the number of samples to capture.   
-This script records the HSV values of the 9 stickers on that face and saves them as .npy files in the dataset folder.   
-These form the original dataset for model training.   

2️⃣ input_dataset2.py

-Used to create augmented datasets.  
-Based on HSV ranges for the six colors, this script generates additional samples and stores them as .npy files.   
-These serve as augmented data to improve the model’s accuracy.   

3️⃣ train.py

-Trains the SVM (Support Vector Machine) model with an RBF kernel.   
-You can tune the parameters C and gamma.   
-Typically, use a smaller C and a larger gamma.   
-The optimal values depend on the mean HSV values from your dataset.   

4️⃣ color_prediction.py

-Used to capture and predict cube colors in real-time.   
-The user is guided to show each cube face according to the <color> center.   
-Ensure proper lighting and cube alignment within the ROI grid.   
-After each face prediction, the user can confirm before moving to the next face.   
-Once all six faces are captured, the data is saved as captured_faces.npy.   

5️⃣ npy_file_checker.py

-A simple utility to view the colors stored in captured_faces.npy.   
-This helps verify that your captured data is correct before generating a solution.   

6️⃣ solution.py

The core of the project 💡

-Loads the captured_faces.npy file and automatically configures the data into the format expected by the Kociemba algorithm.   
-Generates an optimal move sequence to solve the cube.   
-Displays easy, step-by-step instructions for the user to follow and successfully solve the Rubik’s Cube.   

# ⚙️ Installation

To set up and run the project on your system:

1️⃣ Clone the repository
git clone https://github.com/<your-username>/RubiksCubeSolver.git
cd RubiksCubeSolver

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the color prediction or solution script
python color_prediction.py
python solution.py   
   
✅ Make sure Python (≥3.9) and pip are installed before running these commands.

# 🧠 Tech Highlights

Machine Learning: Support Vector Machines (RBF kernel)   
Algorithm: Kociemba’s two-phase optimal solver   
Languages/Libraries: Python, OpenCV, NumPy, scikit-learn   

# 🚀 Outcome

Once trained and executed, the program predicts cube colors accurately and provides clear solving steps — bringing your scrambled cube back to perfection!


# ✨ Credits

Developer: Yamini Shankari AJ 👩‍💻   
