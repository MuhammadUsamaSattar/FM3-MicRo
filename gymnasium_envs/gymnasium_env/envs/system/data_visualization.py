import csv
import math
import os
import random

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pickle
import pysindy as ps
import scipy.integrate
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from gymnasium_env.envs.system.Library import functions


def printTrialsAndDataCount(data):
    """Prints number of trials and datapoints for each current value

    Args:
        data : List containing data separated by trials
    """
    min_global = float("inf")
    max_global = float("-inf")

    count_trials = [0 for _ in range(21)]
    count_datapoints = [0 for _ in range(21)]

    for trial in data:
        count_trials[int(math.ceil(float(trial[0]) / 0.05))] += 1

        n_trial = 0
        for point in trial:
            count_datapoints[int(math.ceil(float(trial[0]) / 0.05))] += 1
            n_trial += 1

        if trial[0] != 0:
            min_global = min(min_global, n_trial)
            max_global = max(max_global, n_trial)

    print("MIN:", min_global)
    print("MAX:", max_global)

    for i in range(len(count_trials)):
        print(i / 20.0, ":", count_trials[i], count_datapoints[i])

    print()


def predictorFromSingleFeature(model):
    """Returns the nested function predict() which allows user to predict from just one data point

    Args:
        model : scikit-learn predictor object
    """

    def predict(*argv):
        """Creates prediction of derivates of entered arguments. Argument must 
        be list of length 2 and consist of current and radial distance

        Returns:
            list: Derviatives of entered argument
        """
        return [0, model.predict([argv[1]])[0]]

    return predict


def flatten3DList(X):
    """Converts 3D list into 2D by flattening the first axis.

    Args:
        X : List containing data separated by trials. First axis is trials. 2nd axis is data points. 
        3rd axis consists of time, current and radial distance values.

    Returns:
        list: 2D list
    """
    X_temp = []
    for trial in X:
        X_temp.extend(trial)

    X = X_temp.copy()

    return X


def addBogusValues(current_vals, X_data, y_data, t_data):
    """Adds constant paths for each current in current_vals

    Args:
        current_vals : Current values for which to add bogus values
        X_data : Dataset separated by trials containing current and radial distance values
        y_data : Dataset separated by trials containing velocity values
        t_data : Dataset separated by trials containing time values

    Returns:
        list: X_data with bogus values
        list: y_data with bogus values
        list: t_data with bogus values
    """
    for current in current_vals:
        # The number within range() indicates how many paths are to be added
        for trial in range(12):
            X_data.append([])
            y_data.append([])
            t_data.append([])

            r = random.randrange(
                200, 600
            )  # The starting position is chosen randomly between the two bounds

            # The number within range() indicates how many datapoints are to be added in each path
            for i in range(100):
                X_data[-1].append([current, r])
                y_data[-1].append(0)
                t_data[-1].append(i * 5 / 100)

    return X_data, y_data, t_data


def normalizeData(X_data, y_data):
    """Normalizes data by scaling according to the maximum value in dataset

    Args:
        X_data : Dataset separated by trials containing current and radial distance values
        y_data : Dataset separated by trials containing velocity values

    Returns:
        list: Normalized X_data
        list: Normalized y_data
    """
    scaler = MinMaxScaler()
    r_data_flattened = [[point[1]] for point in flatten3DList(X_data)]
    r_data_flattened.append([640])
    scaler.fit(r_data_flattened)

    for trial in range(len(X_data)):
        r_scaled = scaler.transform([[point[1]] for point in X_data[trial]])
        X_data[trial] = [[X_data[trial][0][0], r[0]] for r in r_scaled]

        y_data[trial] = scaler.transform([[point] for point in y_data[trial]])
        y_data[trial] = flatten3DList(y_data[trial])

    return X_data, y_data


def calcScoreAndPlotTestData(model, X_test, y_test, t_test, number_to_show=None):
    """Calculates total R2 score of the model on the test values and also plots (optional) 
    some of the paths and their predictions along with the R2 score

    Args:
        model : scikit-learn predictor object
        X_data : Dataset separated by trials containing current and radial distance values
        y_data : Dataset separated by trials containing velocity values
        t_data : Dataset separated by trials containing time values
        number_to_show (optional): Number of plots to show. A value of -1 shows only plots
        with an R2 score < 0. A value of None shows no plots. Defaults to None.
    """
    model_PredictFromSingleFeature = predictorFromSingleFeature(model)

    predict_test = []
    for trial in range(len(X_test)):
        sim = scipy.integrate.solve_ivp(
            model_PredictFromSingleFeature,
            (t_test[trial][0], t_test[trial][-1]),
            X_test[trial][0],
            t_eval=t_test[trial],
        )  # Advances the input initial conditions until the last time value for that particular path
        predict_test.extend(sim.y[1])
        y_plot = sim.y[1]

        score = r2_score(
            [point[1] for point in X_test[trial]], y_plot
        )  # Calculates R2 score for the path

        # Conditions to determine what plots to create
        if number_to_show == None:
            condition = False

        elif number_to_show == -1:
            condition = score < 0

        else:
            condition = number_to_show >= 0

        # Creates the plots
        if condition:
            print(X_test[trial][0])
            plt.plot(t_test[trial], [point[1] for point in X_test[trial]], label="test")
            plt.plot(t_test[trial], y_plot, "--", label="predict")

            print("Individual Score:", score)

            print()

            plt.xlabel("Time (s)")
            plt.ylabel("Radial Distance (pixels)")
            plt.legend()
            plt.show()

            number_to_show = number_to_show - 1

    # Flatten all test data and print total R2 score
    X_test = flatten3DList(X_test)
    y_test = flatten3DList(y_test)

    print()
    print("Total score:", r2_score([point[1] for point in X_test], predict_test))
    print()


def createPredictor(data, current_vals_to_train_on, type):
    """Creates predictor object according to dataset

    Args:
        data : Dataset separated by trials containing time, current and radial distance values
        current_vals_to_train_on : Current values on which to train the model
        type : Type of model to create. "ML" and "SINDy" are available. "SINDy" does not give good results.

    Returns:
        scikit-learn predictor object: Trained model
    """
    # Code for Machine Learning (DL) model
    if type == "ML":
        X_data = []
        y_data = []
        t_data = []

        # Separtes data into features, label and time
        for trial in range(len(data["t"])):
            if str(data["I"][trial][0]) in current_vals_to_train_on:
                X_data.append([])
                y_data.append(
                    np.gradient(data["r"][trial], data["t"][trial])
                )  # Calculates velocity with 2nd order FDM
                t_data.append(data["t"][trial])

                for point in range(len(data["t"][trial])):
                    X_data[-1].extend(
                        [[data["I"][trial][point], data["r"][trial][point]]]
                    )

        # X_data, y_data, t_data = addBogusValues([0.0, 0.05, 0.1, 0.15], X_data, y_data, t_data)
        # print("After adding bogus data")
        # printTrialsAndDataCount([[point[0] for point in trial] for trial in X_data])

        # normalizeData(X_data, y_data)

        # np.random.seed() sets the seed for the initial random values assigned in the model.
        # Uncomment this line and keep the argument constant for consistency
        # np.random.seed(2)

        # random_state changes the shuffling of the data i.e., which paths are used to train the model
        # You can modify this value but do not remove the line, otherwise X_train and X_test values will be
        # incompatible with t_train and t_test
        random_state = 42
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.30, random_state=random_state
        )  # Split the data into train and test data sets for X and y
        t_train, t_test, y_train, y_test = train_test_split(
            t_data, y_data, test_size=0.30, random_state=random_state
        )  # Split the data into train and test data sets for t and y according to the split of X and y done previously

        # Flatten X and y so that they can be fed into the model
        X_train = flatten3DList(X_train)
        y_train = flatten3DList(y_train)

        model = MLPRegressor(
            hidden_layer_sizes=(10, 10, 10),
            activation="relu",
            solver="adam",
            max_iter=500,
            verbose=True,
        )
        model.fit(X_train, y_train)

        calcScoreAndPlotTestData(model, X_test, y_test, t_test, number_to_show=None)

        # Saves the model
        os.makedirs("Models/", exist_ok=True)
        with open(os.path.dirname(__file__) + "/Models/Predict_rdot_from_I_r.pkl", "wb") as f:
            pickle.dump(model, f)

        return model

    # Code for SINDy model. You can ignore this.
    elif type == "SINDy":
        X_data = []
        y_data = []
        for trial in range(len(data["t"])):
            X_data.append([])

            for point in range(len(data["t"][trial])):
                X_data[-1].append([data["I"][trial][point], (data["r"][trial][point])])

            X_data[-1] = np.array(X_data[-1])
            y_data.append(np.array(data["t"][trial]))

        X_data, x_test, y_data, t_test = train_test_split(
            X_data, y_data, test_size=0.15, random_state=42
        )

        differentiation_method = ps.SmoothedFiniteDifference(order=2, d=1)

        z = 4.5

        library_functions = [
            lambda x, y: ((y / np.sqrt(y**2 + (z * 720 / 16) ** 2)))
            / np.sqrt(y**2 + (z * 720 / 16) ** 2) ** 1,
            lambda x, y: ((y / np.sqrt(y**2 + (z * 720 / 16) ** 2)))
            / np.sqrt(y**2 + (z * 720 / 16) ** 2) ** 2,
            lambda x, y: ((y / np.sqrt(y**2 + (z * 720 / 16) ** 2)))
            / np.sqrt(y**2 + (z * 720 / 16) ** 2) ** 3,
            lambda x, y: ((y / np.sqrt(y**2 + (z * 720 / 16) ** 2)))
            / np.sqrt(y**2 + (z * 720 / 16) ** 2) ** 4,
            lambda x, y: ((y / np.sqrt(y**2 + (z * 720 / 16) ** 2)))
            / np.sqrt(y**2 + (z * 720 / 16) ** 2) ** 5,
        ]

        library_function_names = [
            lambda x, y: "cos(theta)/R^1",
            lambda x, y: "cos(theta)/R^2",
            lambda x, y: "cos(theta)/R^3",
            lambda x, y: "cos(theta)/R^4",
            lambda x, y: "cos(theta)/R^5",
        ]

        custom_library = ps.CustomLibrary(
            library_functions=library_functions, function_names=library_function_names
        )

        inputs_per_library = np.array([[0, 1], [0, 0]])
        tensor_array = [[1, 1]]

        feature_library = ps.GeneralizedLibrary(
            [custom_library, ps.PolynomialLibrary(degree=5, include_bias=False)],
            tensor_array=tensor_array,
            inputs_per_library=inputs_per_library,
        )

        # perf = []
        # thresholds = np.linspace(0, 1, 21)
        #
        # for threshold in thresholds:
        optimizer = ps.STLSQ(threshold=0.1)
        model = ps.SINDy(
            differentiation_method=differentiation_method,
            feature_library=feature_library,
            optimizer=optimizer,
            feature_names=["I", "r"],
            discrete_time=False,
        )

        model.fit(X_data, t=y_data, multiple_trajectories=True, ensemble=False)
        print(model.get_feature_names())
        model.print()
        print("R2 score: ", model.score(x_test, t_test, multiple_trajectories=True))

        #    perf.append(model.score(x_test, t_test, multiple_trajectories=True))
        # plt.plot(thresholds, perf)
        # plt.show()

        # sim_t = np.linspace(0, 10, 51)
        # for trial in range(len(X_data)):
        #    sim = model.simulate(x_test[trial][0], t = t_test[trial])
        #
        #    print(model.score(x_test[trial], t_test[trial]))
        #    plt.plot(t_test[trial], sim[:,1], "--")
        #    plt.plot(t_test[trial], [d[1] for d in x_test[trial]])
        #    plt.show()

        return model

    else:
        print("Invalid model type!")


def plotModel(model, t_range, data_points, number_of_generated_plots, type, plot_type):
    """Plots the predicted paths of the model

    Args:
        model : scikit-learn predictor object
        t_range : The two bounds of time between which to plot the model response
        data_points : Number of datapoints in each path
        number_of_generated_plots : Number of plots to generate between 0.0 and 1.0
        type : Type of model to plot. "ML" and "SINDy" are available.
        plot_type : Type of plot to create. "2D" and "3D" options are available.
    """
    # Code for Machine Learning (DL) model
    if type == "ML":
        x_plot = np.linspace(t_range[0], t_range[1], data_points)  # Data for time
        y_plot = np.linspace(0.0, 1.0, number_of_generated_plots)  # Data for currents
        z_plot = []  # Data for radial distances

        for current in y_plot:
            r = 500  # Starting radial distance

            model_PredictFromSingleFeature = predictorFromSingleFeature(model)
            sim = scipy.integrate.solve_ivp(
                model_PredictFromSingleFeature,
                (x_plot[0], x_plot[-1]),
                [current, r],
                t_eval=x_plot,
            )  # Advances the intial conditions until last time in datapoint

            z_plot.extend([sim.y[1]])

            print(str(current) + ": Solution converges at:", z_plot[-1][-1])

        # Code for 2D plot. Independently plots paths for all current values.
        if plot_type == "2D":
            for i, current in enumerate(y_plot):
                plt.plot(x_plot, z_plot[i], "--", label=str(current))

            plt.xlabel("Time (s)")
            plt.ylabel("Radial Distance (pixels)")
            plt.legend()
            plt.show()

        # Code for 3D plot. Creates a mesh of x and y values so that each z value can be linked to a point in this mesh.
        elif plot_type == "3D":
            x_plot, y_plot = np.meshgrid(x_plot, y_plot)

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(
                x_plot,
                y_plot,
                np.array(z_plot),
                cmap=cm.coolwarm,
                linewidth=0,
                antialiased=False,
            )

            # ax.set_xlim(0, 5)
            ax.set_ylim(0, 1.0)
            # ax.set_zlim(0)

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Current (scaled)")
            ax.set_zlabel("Radial Distance (pixels)")

            fig.colorbar(surf, shrink=0.5, aspect=5)

            plt.show()

        else:
            print('Invalid plot type. Choose either "2D" or "3D"')

    # Code for SINDy models. You can ignore this.
    elif type == "SINDy":
        if plot_type == "3D":
            epsilon = 10e-6

            x_plot = np.linspace(0, 10, data_points)
            y_plot = np.linspace(0.0, 1.0, number_of_generated_plots)
            z_plot = []

            dt = x_plot[1] - x_plot[0]

            for curr in y_plot:
                sim = model.simulate([curr, 500], t=x_plot)

                z_plot.append(sim[:, 1])
                for i in range(1, len(z_plot[-1])):
                    if abs(z_plot[-1][i] - z_plot[-1][i - 1]) < epsilon:
                        for j in range(i, len(z_plot[-1])):
                            z_plot[-1][j] = None

                        break

                print(
                    "For I = "
                    + str(curr)
                    + ", radial distance converges at "
                    + str(z_plot[-1][i - 1])
                )

            if plot_type == "2D":
                for curr in y_plot:
                    plt.plot(
                        x_plot,
                        z_plot[-1],
                        "--",
                        label="SINDy model_" + str(float(curr)),
                    )

                plt.xlabel("Time (s)")
                plt.ylabel("Radial Distance (pixels)")
                plt.legend()
                plt.show()

            elif plot_type == "3D":
                x_plot, y_plot = np.meshgrid(x_plot, y_plot)

                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                surf = ax.plot_surface(
                    x_plot,
                    y_plot,
                    np.array(z_plot),
                    cmap=cm.coolwarm,
                    linewidth=0,
                    antialiased=False,
                )

                # ax.set_xlim(0, 5)
                ax.set_ylim(0, 1.0)
                # ax.set_zlim(0)

                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Current (scaled)")
                ax.set_zlabel("Radial Distance (pixels)")

                fig.colorbar(surf, shrink=0.5, aspect=5)

                plt.show()

            else:
                print('Invalid plot type. Choose either "2D" or "3D"')

    else:
        print("Invalid model type!")


def processDataRemoveFinalDuplicateVals(data):
    """Removes duplicate values at ends of trials

    Args:
        data : List containing data separated by trials

    Returns:
        data: List containing data separated by trials in 
        which duplicate values at the end have been removed
    """
    data_temp = data["r"][-1].copy()

    allEqual = True  # Determines if all values in the trial are equal i.e., the particle is unaffected
    for point in range(1, len(data["r"][-1])):
        if data["r"][-1][point] != data["r"][-1][point - 1]:
            allEqual = False
            break

    if not allEqual:  # If all values in the trial are equal i.e., the particle is unaffected then don't remove anything
        for i in range(len(data_temp) - 1, -1, -1):
            if data_temp[i] == data_temp[i - 1]:
                data["t"][-1].pop(i)
                data["I"][-1].pop(i)
                data["r"][-1].pop(i)

            else:
                break

    return data


def processDataUnaffectedParticle(data):
    """Sets particles that move away from the solenoid as being constant instead

    Args:
        data : List containing data separated by trials

    Returns:
        list: List containing data separated by trials in which 
        unaffected particles have been set as constant instead
    """    
    if data["r"][-1][0] < data["r"][-1][-1]:
        for i in range(1, len(data["r"][-1])):
            data["r"][-1][i] = data["r"][-1][0]

    return data


def processData(data):
    """Runs pre-processing on the last trial recorded in data

    Args:
        data : List containing data separated by trials

    Returns:
        list: List containing data separated by trials 
        which has been pre-processed
    """    
    data = processDataRemoveFinalDuplicateVals(data)
    data = processDataUnaffectedParticle(data)

    return data


def readAndProcessFiles(filenames):
    """Reads the file and creates data from it

    Args:
        filenames : List containing filepaths to the data files

    Returns:
        list: List containing data separated by trials
    """
    data = {"t": [[]], "I": [[]], "r": [[]]}

    for file in filenames:
        with open(os.path.dirname(__file__) + "/Data/" + file, mode="r") as file:
            csvFile = csv.reader(file)

            i = 0
            for line in csvFile:

                if i > 0:  # Ignores the field headers
                    t = float(line[0])
                    I = float(line[2])

                    particle_loc = line[1][1:-1].split(", ")
                    particle_loc = list(map(int, particle_loc))

                    solenoid_loc = line[3][1:-1].split(", ")
                    solenoid_loc = list(map(int, solenoid_loc))

                    r = functions.distance(
                        particle_loc[0],
                        particle_loc[1],
                        solenoid_loc[0],
                        solenoid_loc[1],
                    )

                    # If nothing has been added to the list yet or current datapoint corresponds
                    # to start of new path, then append a new blank list in the data which means
                    # that we are starting a new trial
                    if len(data["t"][-1]) > 0 and data["t"][-1][-1] > t:
                        data = processData(data)

                        data["t"].append([])
                        data["I"].append([])
                        data["r"].append([])

                    data["t"][-1].append(t)
                    data["I"][-1].append(I)
                    data["r"][-1].append(r)

                else:
                    i += 1

            data = processData(data)

    return data


def plotData(data):
    """Plots all trials

    Args:
        data : List containing data separated by trials
    """    
    for trial in range(len(data["t"])):
        plt.plot(data["t"][trial], data["r"][trial])

    plt.xlabel("Time (s)")
    plt.ylabel("Radial Distance (pixels)")
    plt.show()


def plotDataTrendByCurrent(data, number_of_dataset_plots_to_show):
    """Plots all trials with each current value having a unique color

    Args:
        data : List containing data separated by trials
        number_of_generated_plots : Number of plots to generate between 0.0 and 1.0. 0.0 is ignored.
        This number must a factor of actual number of plots available in recorded data.
    """ 
    # Creates keys of all current values and creates empty lists in dictionary for those keys   
    data_dict_by_current = {}
    for val in range(1, number_of_dataset_plots_to_show+1):
        data_dict_by_current[str(val / (number_of_dataset_plots_to_show))] = []

    # Adds data into dictionary corresponding to its key
    for trial in range(len(data["I"])):
        if str(float(data["I"][trial][0])) in data_dict_by_current.keys():
            data_dict_by_current[str(float(data["I"][trial][0]))].append([])

            for point in range(len(data["I"][trial])):
                data_dict_by_current[str(float(data["I"][trial][point]))][-1].append(
                    [
                        data["t"][trial][point],
                        data["r"][trial][point],
                    ]
                )

    for key in range(1, number_of_dataset_plots_to_show+1):
        for trial in data_dict_by_current[
            str(float(key / (number_of_dataset_plots_to_show)))
        ]:
            features = [point[0] for point in trial]
            labels = [point[1] for point in trial]

            plt.plot(features, labels, color="C" + str(int(float(key))))  # Plots data

        plt.scatter(
            [],
            [],
            color="C" + str(int(float(key))),
            label=round(key / (number_of_dataset_plots_to_show), 2),
        )  # Plots blank data for each current value so that legend is only set once

    plt.xlabel("Time (s)")
    plt.ylabel("Radial Distance (pixels)")
    plt.legend()
    # plt.ylim(0, 100)
    plt.show()


if __name__ == "__main__":
    filenames = ["Tenth_One Solenoid Only/1_Solenoid_Calibration_Data.csv"]
    data = readAndProcessFiles(filenames)
    print("After Reading and Processing Data")
    printTrialsAndDataCount(data["I"])
    

    number_of_dataset_plots_to_show = 10  # Sets the number of plots to show for plotDataTrendByCurrent()

    # Uncomment this to plot data
    plotData(data)

    # Uncomment this to plot data which is color separated according to the current value
    plotDataTrendByCurrent(data, number_of_dataset_plots_to_show)


    # Sets the number of current values to train on. This should be a factor of the number of current values in saved dataset.
    number_current_vals_to_train_on = 21

    current_vals_to_train_on = [
        str(i / float(number_current_vals_to_train_on - 1))
        for i in range(4, number_current_vals_to_train_on)
    ]  # The current values on which to train the model. Change the bounds of range() to change which values to start and end at.

    # Uncomment the following line and comment the "current_vals_to_train_on = [...]" line to manually type which current values to train on.
    # current_vals_to_train_on = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

    types = ["ML", "SINDy"]
    type = types[0]  # types[0] for "ML" model and types[1] for "SINDy" model

    model = createPredictor(data, current_vals_to_train_on, type=type)

    # Uncomment the following two lines and comment the "model = createPredictor(...)" line to see the result on an existing saved model
    #with open('Models/' + 'Predict_rdot_from_I_r_R2_0.916_No_0.0_to_0.15_shuffled' + '.pkl', 'rb') as f:
    #   model = pickle.load(f)


    t_range = [0, 4]  # Bounds of the time for generated plots in plotModel()

    data_points_for_generated_plots = int(25 * t_range[1] + 1)  # Number of datapoints in each plot for plotModel()
    number_of_generated_plots = 11  # Sets the number of generated plots for plotModel(). Includes 0.0 current.

    plotModel(
        model,
        t_range=t_range,
        data_points=data_points_for_generated_plots,
        number_of_generated_plots=number_of_generated_plots,
        type=type,
        plot_type="3D",
    )  # plot_type "3D" creates a 3D plot while plot_type "2D" creates a 2D plot for result of the model
