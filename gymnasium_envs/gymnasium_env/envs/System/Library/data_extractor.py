import csv
import os
import time


class data_extractor:
    """Class for extracting and saving data"""

    def __init__(self, coil_loc, coil_names):
        """Initializes class attributes

        Args:
            coil_loc : List of size 8 containing locations of coils where 0th element corresponds to Northern coil
            coil_names : List of size 8 containing names of coils where 0th element corresponds to Northern coil
        """

        self.coil_loc = coil_loc
        self.coil_names = coil_names
        self.file_counter = (
            1  # Tracks the number of file to write to so that all filenames are unique
        )

    def startDataSet(self, mode):
        """Resets the recorded data with field names according to mode

        Args:
            mode : String that determines whether the mode is "solenoid_calibration" or "log_data"
        """
        self.mode = mode

        # Creates the data fields that are relevant to the mode type
        if self.mode == "solenoid_calibration":
            print("Solenoid Calibration data recording started!")
            self.fields = ["t", "Particle Location", "Coil_Current", "Coil_Location"]

        elif self.mode == "log_data":
            print("Log data recording started!")
            coil_curr_field = ["Coil_Current_" + x for x in self.coil_names]
            coil_loc_field = ["Coil_Location_" + x for x in self.coil_names]

            self.fields = (
                ["t", "Particle Location", "Goal Location"]
                + coil_curr_field
                + coil_loc_field
            )

        self.data = []
        self.start_time = time.time()

    def startDataSeries(self):
        """Resets the time which is useful when running multiple trials"""
        self.start_time = time.time()

    def record_datapoint(self, particle_loc, coil_vals, goal_loc=None, coil_index=None):
        """Adds data to the self.data variable according to the mode set by startDataSet()

        Args:
            particle_loc : List of size 8 containing particle locations where 0th element corresponds to Northern coil
            coil_vals : List of size 8 containing coil currents where 0th element corresponds to Northern coil
            goal_loc : List of size 2 containing x and y co-ordinates of goal. Not needed for "solenoid_calibration" mode.
            coil_index : Index of coil which is being calibrated. Not needed for "log_data" mode.
        """
        # Records the data that is relevant to the mode type
        if self.mode == "solenoid_calibration":
            t = time.time() - self.start_time

            self.data.append(
                [
                    t,
                    list(map(int, particle_loc)),
                    coil_vals[coil_index],
                    self.coil_loc[coil_index],
                ]
            )

        elif self.mode == "log_data":
            t = time.time() - self.start_time

            self.data.append(
                [t, list(map(int, particle_loc)), goal_loc.copy()[0]]
                + coil_vals
                + self.coil_loc
            )

    def save_data(self, dir):
        """Saves data to a .csv file

        Args:
            dir : Directory at which the file is to be saved
        """
        # Sets filename according to the mode type
        if self.mode == "solenoid_calibration":
            filename = "Solenoid_Calibration_Data"

        elif self.mode == "log_data":
            filename = "Log_Data"

        local_path_string = dir + "/" + str(self.file_counter) + "_" + filename + ".csv"

        os.makedirs(dir, exist_ok=True)  # Creates the directory if it does not exist

        # Creates a .csv file and writes data to it
        with open(local_path_string, "w", newline="") as csvfile:
            csvsvwriter = csv.writer(csvfile)
            csvsvwriter.writerow(self.fields)
            csvsvwriter.writerows(self.data)

        print(
            "Data saved: ", os.getcwd() + "/" + local_path_string
        )  # Prints complete path of the file to the terminal

        self.file_counter += 1
