
1. Install dependencies in a new seperate environment by running the below lines 1-by-1.(to avoid conflicts with pre-downloaded packages):

   python -m venv venv

   .\venv\Scripts\activate

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   pip install -r requirements.txt



2. Launch the agents by running launch_agents.bat. this will automatically activate the virtual environment and install all required packages. (First launch will take a while)

3. Run the project in Unity. (First launch will take a while)


// How to use this project

1. Open the 'Start Scene' and press Play. (make sure all 3 python agents from the .bat file are running before pressing play)
2. In the in-game Settings menu, set the paths for the 'in-game slideshow' and 'test bank' to their correct locations on your drive. (they light up green if valid, reference provided on github. make sure the pdf is converted to .pngs using the given script in LectureFood subfolder)
3. Click 'Start'. this loads the Classroom Scene.
4. In this Classroom scene, Click 'Toggle Lecture' to begin lecturing, then toggle it off when done. (Make sure you have a valid microphone connected before beginning to lecture).
5. Click 'Start Exam' to begin the evaluation.


// Controls (Mouse & Keyboard)

Right-click + move mouse = Rotate camera.

Left Shift + move mouse = Rotate left hand in-game.

Left Shift + Left-click = Click a button in-game.

