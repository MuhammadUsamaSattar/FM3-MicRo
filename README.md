# Requirements
python==3.12

cuda==12.4

All libraries in requirements.txt

**`NOTE: This description is for the summer job project. This needs to be updated once the thesis matures a bit.`**

# Particle Mainpulator
This repository is the code base for Micro- and Nano-robotics Course's Lab offered at by Robotic Instruments Lab at Aalto University, Finland. 

## Installation
There is no functionality to `pip install` the package right now. For installation, the user has to manually download the code and extract it in their desired repository.

The GUI part of the code can only be run on the lab computer since it needs the Arduinos and FLIR camera. Other components can be run from any computer.

The simulator is part of the complete Particle Manipulator package and uses many of the same dependencies as the experimental setup program (gui.py). Therefore, it is recommended to copy the entire code repository rather than individual files. Nonetheless, the instructions also detail which individual files need to be copied to allow just the simulator to run. 

The instructions are applicable for both Windows and Linux; where there is a difference between the two, it is clearly specified.

1.	Copy the code to your PC’s drive. You can either copy the entire package or just the required files for the simulator. 
If you want to copy just the required files, the hierarchy for this selective package is:
    - {Package Folder}
        -	Library
            - data_extractor.py
            - functions.py
            - pygame_recorder.py
        - Models
            - Predict_rdot_from_I_r_R2_0.916_No_0.0_to_0.15_shuffled
        - control_algorithm.py
        - initializations.py
        - path_plotter.py
        - simulator.py
2.	If you have python installed on your PC, you can skip this step.

    - **Python Windows Installation:**

        https://www.python.org/downloads/
    
        To verify if the install was successful, you can run the following command in command prompt: 
    
        `python3 -–version`

    - **Python Linux Installation:**

        Usually, python is pre-installed on Linux systems. You can verify if there is an installation by running the command in command prompt:

        `$ python3 –-version`

        If the command returns no valid python installation, you can install it by following this tutorial:

        https://docs.python-guide.org/starting/install3/linux/

3.  You can write and run python code using most simple text editor     programs. But it is better to have a proper IDE since they make writing code and debugging it easier. We recommend Microsoft Visual Studio Code (VS Code), but you can install any other IDE you want.

    https://code.visualstudio.com/download

4.  Run VS Code and go to the Extensions tab by clicking on the correct icon on the left side of the screen. Alternatively, you can use the shortcut *Ctrl+Shift+X*. Install Python extension which allows you to use python debugger and functionalities within your workspace. If you are using a different IDE, you will have to consult its documentation on how to use Python on it.
5.  *pip* is a package management tool for python which allows you to download, install, upgrade and uninstall various python packages. Generally, installing python automatically installs *pip* as well, but if you do not have *pip*, you can install it by following the tutorial under **get-pip** heading for your OS:

    https://pip.pypa.io/en/stable/installation/#get-pip-py

6.	You need the following libraries to run the simulator and path plotter:
    - pygame
    - numpy
    - scipy
    - scikit-learn
    - opencv
    - matplotlib

    You can install the required libraries using the `pip install`. If you are using Visual Studio Code, you can go to the Terminal tab on the bottom of the screen. Alternatively, you can use the shortcuts:
    - **Windows Shortcut:** *Ctrl+`* on English keyboard layout or *Ctrl+ö* on Finnish keyboard layout. 

    - **Linux Shortcut:** *Ctrl+`* on English keyboard layout or *Ctrl+Shift+´* on Finnish keyboard layout.
    
    Then run the following commands one-by-one to install the required libraries:

    ```
    pip install pygame
    pip install numpy
    pip install scipy
    pip install scikit-learn
    pip install opencv-python
    pip install matplotlib
    ```

    There might be older versions of the libraries already installed on the PC, but it is best to upgrade them to prevent running into issues. In such cases, add *-U* after the pip install command to upgrade the libraries.

    `pip install -U {Library}`

    For example:

    `pip install -U numpy`

7.	You can now run the *simulator.py* and *path_plotter.py* files using *Ctrl+F5*. You might be prompted to select an interpreter at the top of the screen. You can select the latest Python version. 


## Description
The setup consists of 8 solenoid arranged in a circular arrangement. A ferromagnetic particle is placed within the circle traced out by these solenoids. Application of current at any solenoid allows it to attract the particle towards itself. Simultaneous smart acutation of these solenoid can allow placement of the particle accurately.

![CAD model of the system](/assets/Description/CAD.png)

![Picture of the system](/assets/Description/Picture.png)

### System Mechanics
The particle is accelerated towards the solenoids through magnetic force. The fluid drag while passing through the water surface reduces the particle momentum. Additionally, the meniscus of the water surface pulls the particle to the center of the petri dish, but due to the relatively large size of our petri dish, this effect is negligible.

Therefore, the equation of motion only consists of the magnetic and drag force. The governing equation of this system is given in the following figure:

![Diagram of system mechanics](/assets/Description/System%20Mechanics.png)

## Code
Broadly, this project has 4 main components:

### GUI
The GUI consists of the code and its dependencies which allows the user to interact with the arduinos and the camera. Therefore, this part of the package allows user to work with the experimental setup. It is written on QT.

![Screenshot of the main window](/assets/GUI/Main%20Screen.png)

The buttons and their functionalities are as follows:
- **Video:** Toggles video.
- **Grid:** Toggles grid.
- **Recording:** Toggles recording. The video file is saved at *~/Data/Experiment_Data_{Timestamp of Start of Program}* where *~* is the directory of your *gui.py* file. Each recording is labelled *Recording_{number}.avi*.
- **Take Snapshot:** Takes a screenshot of the current camera view. The video file is saved at *~/Data/Experiment_Data_{Timestamp of Start of Program}* where *~* is the directory of your *gui.py* file. Each recording is labelled *Snapshot_{number}.jpg*.
- **Object Detection:** Toggles object detection. The program is initialized with object location at top-right of the screen.
- **Show Solenoid State:** Shows the reference positions of the solenoids and color according to the amount of current.
- **Show Goal:** Toggles goal position. The program is initialized with goal at center of the screen.
- **Show Goal Vector:** Toggles the vector from particle location to the goal location.
- **Manual Control:** Allows user to manually set current values in percentage.
- **Reset:** Resets the value of each solenoid to 0.0.
- **Automatic Control:** Toggles automatic control algorithm and turns on object detection. It uses the control algorithm in *control_algorithm.py* file.
- **Multiple Goals Mode/Execute Goal Motion/Cancel Motion:** Turning on *Multiple Goals Mode* allows user to define multiple goals which are then achieved through automatic control one by one. There is no limit on the number of goals that can be set by the user. 
After starting the *Multiple Goals Mode*, the button label changes to *Execute Motion* which allows the user to instruct the program to start executing the set goals
Once the Execute Motion button has been pressed, the control algorithm executes the goals one by one. If no goal was added, then the goal that was set before pressing the button is taken as the current goal and *Multiple Goals Mode* is cancelled
Once the execution of the motion has been started, the user can observe that the button label changes to *Cancel Motion* which allows the user to cancel the *Multiple Goals Motion*, if needed.
On completion or cancellation of the motion, the label changes back to *Multiple Goals Mode* which allows the user to set another series of goals.
- **Set Goal/Add Goal:** Sets/Adds the entered *X* and *Y* positions as goals depending upon whether *Multiple Goals Mode* is turned on or off.
- **Log Data:** Toggles data logging which records the time, particle position, current goal, solenoid current values and solenoid locations. The log data is stored at *~/Data/Experiment_Data_{Timestamp of Start of Program}* where *~* is the directory of your *gui.py* file. Each log is labelled *{number}_Log_Data.csv*.
- **Solenoid Calibration:** Toggles solenoid calibration scheme which measures the time, particle position and solenoid current for every solenoid to determine the mechanics of the system. The calibration data is stored in *~/Data/Experiment_Data_{Timestamp of Start of Program}* where *~* is the directory of your *simulator.py* file. Each calibration is labelled *{number}_Solenoid_Calibration_Data.csv*.
- **Demagnetize Solenoids:** Demagnetizes each solenoid by alternating between decreasing magnitudes of positive and negative current values.

### Simulator
The simulator consists of a modelled pygame program which imitates the experimental setup. The purpose is to allow the user to develop and test their own control algorithm in a controlled environment.

![Screenshot of the simulator main window](/assets/Simulator/Main%20Window.png)

The following interactions are availabe in the simulator:

- Move the particle with *left-click*. The particle cannot be moved outside the circle traced by the solenoids.
- Move the goal with *right-click*. The goal cannot be moved outside the circle formed by the solenoids.
- Left clicking on the position text of the particle and goal at the bottom-left of the screen allows user to type in the value for the respective parameter. The *X* and *Y* co-ordinates should be separated by a comma. The particle and goal cannot be moved outside the circle formed by the solenoids.
- Press *m* to start *Multiple Goals Mode*, which allows you to place multiple goals. Once you have placed all the goals, press *m* to start executing the motion.
- Press *l* to start logging data which stores the time, particle position, current goal, solenoid current values and solenoid locations. Press *l* again to stop logging data. The log data is stored at *~/Data/Simulator_Data_{Timestamp of Start of Program}* where *~* is the directory of your *simulator.py* file. Each log is labelled *{number}_Log_Data.csv*.
- Press *v* to start recording the simulator screen. Press *v* again to stop the recording and save it. The recording is stored at *~/Data/Simulator_Data_{Timestamp of Start of Program}* where *~* is the directory of your *simulator.py* file. Each recording is labelled *Recording_{number}.avi*.

### Data Visualization and Modelling
The step that links the Experimental setup and Simulator is the data modelling part of the code. This takes in the raw data recorded from experimental setup, cleans it up and fits a model to it.

Initially, SINDy approach was taken but it failed to provide good results.Currently, a Deep Learning model is fitted on the data which has 3 hidden layers with 10 nodes each. The R2 score for this DL model is ~0.92.

The results were mapped in a 3D plot which shows the predicted velocity depending upon the distance from the solenoid and the applied current.

![3D plot of the model over different current and position values](/assets/Modelling/3D%20Plot.png)

### Path Plotter
Path Plotter allows the user to convert the log data files from the simulator and GUI into 2D plots showing the path taken by the particle and the various associated goals.

![Screenshot of a sample output generated from recorded log data](/assets/Path%20Plotter/Plot.png)

To plot your own log data files, you need to open the *path_plotter.py* file that you copied. Then you need to put the path of your file in the filename variable.
The path can either be:

- Relative to the *path_plotter.py* file in which case you can have an example input of `Data/Simulator_Data/1_Log_Data.csv` which will look for a folder called *Data* in the same folder as *path_plotter.py*. Once found, it will look for the folder *Simulator_Data* and within it *1_Log_Data.csv*
- A global path like `C:/MyFiles/PlotFiles/1_Log_Data.csv`

Note that you need to include the extension *.csv* in the filename.

The output opens in a *matplotlib* window. You can pan and zoom in the window. Additionally, you can save the output by pressing *Ctrl+S* or clicking the save button on top of the screen.

## Contacts
- The PySpin Wrapper and Arduino code was written by [Jiangkun Yu](mailto:jiangkun.yu@aalto.fi).
- The orignal GUI and control algorithm code (the first commit of this project) was written by [Erik Skeel](mailto:erik.skeel@aalto.fi) in Summer 2023. He also modified the Arduino code to suit our application.
- All other modules were developed by [Usama Sattar](mailto:muhammad.sattar@aalto.fi) in summer 2024. He also heavily modifed the GUI file. Therefore, he is the main developer of the latest version of the code.
- The project was supervised by [Artur Kopitca](mailto:artur.kopitca@aalto.fi) during all development phases.
- The project was developed under general group level supervision of [Quan Zhou](mailto:quan.zhou@aalto.fi) during all development phases.

You should direct any questions or concerns towards Artur since other contributors are either not related to the project anymore or provided high level supervision.
