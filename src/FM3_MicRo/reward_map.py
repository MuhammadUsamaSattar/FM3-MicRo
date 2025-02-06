import numpy as np
import argparse
import matplotlib.pyplot as plt
from gymnasium_env.envs.System.Library.functions import distance
import math
from gymnasium_env.envs.Library.llm import LLM
import yaml
import json
from pathlib import Path
import time
from datetime import datetime


def calculate_rewards(data, data_buffer, llm, prompt_file, verbose):
    """Example function that takes two points and returns a computed value."""
    # return np.linalg.norm(np.array(point1) - np.array(point2))  # Example: Euclidean distance

    if len(data_buffer) > 0:
        if verbose:
            print(f"Number of points in current buffer: {len(data_buffer)}")

        context_prompt_file = (
            Path(__file__).parent.parent.parent
            / f"src/FM3_MicRo/prompts/rl_fm_rewards/{prompt_file}"
        )

        with open(context_prompt_file) as stream:
            try:
                content = yaml.safe_load(stream)
                context = content["prompt"]
                r = content["range"]
            except yaml.YAMLError as exc:
                print(exc)

        contexts = []
        prompts = []

        for _ in data_buffer:
            contexts.append(context)

            txt = (
                f"\n"
                f"The particle was located at ({data_buffer[0][0][0]:.2f}, {data_buffer[0][0][1]:.2f}).\n"
                f"The particle is currently located at ({data_buffer[0][1][0]:.2f}, {data_buffer[0][1][1]:.2f}).\n"
                f"The goal is currently located at (0.00, 0.00).\n\n"
                f"What is the reward score?"
            )
            prompts.append(txt)

        t = time.time()
        outputs = llm.get_response(
            len(data_buffer),
            contexts,
            prompts,
            1000,
        )

        if verbose:
            print(f"Total time taken in inference of entire buffer: {(time.time()-t):.3f} seconds")
            print(outputs)
            print(f"{'Point':<25} {'Value':<10}") 

        for i in range(len(data_buffer)):
            try:
                output = float(outputs[i])
            except ValueError:
                output = float(outputs[i][: outputs[i].find("\n")])

            data[data_buffer[i]] = output / r[1]
            
            if verbose:
                print(f"{str(data_buffer[i]):<25} {data[data_buffer[i]]:>10.3f}")

    return data


def generate_workspace_data(
    workspace_radius, resolution, reward_calculation_radius, llm, prompt_file, verbose
):
    """Generates data for all pairs of points within range in a circular workspace."""
    data = {}

    # Generate points within the circular workspace
    x_vals = np.arange(-workspace_radius, workspace_radius + resolution, resolution)
    y_vals = np.arange(-workspace_radius, workspace_radius + resolution, resolution)
    points = [
        (round(x / resolution) * resolution, round(y / resolution) * resolution)
        for x in x_vals
        for y in y_vals
        if x**2 + y**2 <= workspace_radius**2
    ]

    total_points_processed = 0
    # Compute values for each valid pair of points
    for p1 in points:
        x_vals = np.arange(
            p1[0] - reward_calculation_radius,
            p1[0] + reward_calculation_radius + resolution,
            resolution,
        )
        y_vals = np.arange(
            p1[1] - reward_calculation_radius,
            p1[1] + reward_calculation_radius + resolution,
            resolution,
        )

        for p2 in [
            (x, y)
            for x in x_vals
            for y in y_vals
            if ((x**2 + y**2 <= workspace_radius**2))
        ]:
            p2 = (
                round(p2[0] / resolution) * resolution,
                round(p2[1] / resolution) * resolution,
            )
            data_buffer = []

            data_buffer.append((p1, p2))
            total_points_processed += 1
            data = calculate_rewards(data, data_buffer, llm, prompt_file, verbose)

        print(
            f"Total time taken for {total_points_processed} datapoints: {(time.time()-total_run_time):.3f} seconds.\nProcessing rate: {((time.time()-total_run_time)/total_points_processed):.3f} seconds/datapoint.\n"
        )

    return data, points


def visualize_data_from_json(file_path, input_point, show_plot_after_save, verbose):
    """Loads data from a JSON file and visualizes it for a specific input point."""

    t = time.time()
    # Load JSON data
    with open(file_path, "r") as file:
        data = json.load(file)
    print(f"Total number of datapoints: {len(data)}")

    points = []
    nearest_input_point = [(float("inf"), float("inf")), float("inf")]
    for key, value in data.items():
        key_tuple = tuple(map(float, key.split("_")))  # Convert string to tuple of ints
        p1 = (key_tuple[0], key_tuple[1])
        if p1 not in points:
            points.append(p1)

            d = distance(input_point[0], input_point[1], p1[0], p1[1])
            if d < nearest_input_point[1]:
                nearest_input_point = [p1, d]

    input_point = nearest_input_point[0]

    x_coords = []
    y_coords = []
    values = []
    valued_points = []

    # Extract points where the first coordinate matches the input point
    if verbose:
        print(f"{'Point':<50} {'Value':<10}")

    # Convert keys back to tuples and filter based on input_point
    for key, value in data.items():
        # Convert "x1_y1_x2_y2" back to ((x1, y1), (x2, y2))
        key_tuple = tuple(map(float, key.split("_")))  # Convert string to tuple of ints
        if len(key_tuple) == 4:
            p1 = (key_tuple[0], key_tuple[1])
            p2 = (key_tuple[2], key_tuple[3])

            if p2 not in [(0, 0), p1]:
                if p1 == input_point:
                    if verbose:
                        print(f"{str((p1, p2)):<50} {value:>10.3f}")

                    x_coords.append(p2[0])
                    y_coords.append(p2[1])
                    values.append(value)
                    valued_points.append(p2)

                elif p2 not in valued_points:
                    x_coords.append(p2[0])
                    y_coords.append(p2[1])
                    values.append(0)

    print(f"Time taken: {time.time() - t}")
    # Create scatter plot
    plt.figure(figsize=(8, 8))

    # Scatter plot for all points
    scatter = plt.scatter(x_coords, y_coords, c=values, cmap="coolwarm", s=50, marker=".", vmin=-1.0, vmax=1.0)

    # Scatter plot for goal and current position
    plt.scatter([0], [0], c='black', s=50, marker='x')
    plt.scatter([input_point[0]], [input_point[1]], c='black', s=50, marker='o')

    # Add colorbar and labels
    plt.colorbar(scatter, label="Reward Value")
    plt.title(f"Visualization for Input Point {input_point}")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()

    # Extract variables from the JSON filename
    filename = Path(file_path).name
    file_stem, file_ext = filename.rsplit(".", 1)  # Remove extension
    parts = file_stem.split("_map_")
    
    if len(parts) == 2:
        file_suffix = parts[1]  # Extract model_id and parameters

        # Construct new plot filename
        plot_filename = f"{input_point[0]}_{input_point[1]}_map_{file_suffix}.png"
        plot_path = Path(file_path).parent / plot_filename

        # Save the plot
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved to: {plot_path}")

    if show_plot_after_save:
        plt.show()


def pre_calculate_valid_pairs(workspace_radius, reward_calculation_radius, resolution, verbose):
    """Mathematically calculates the total number of valid pairs."""
    ## Number of points in the workspace within the circle of radius R
    #num_points_in_circle = (np.pi * workspace_radius**2) / (
    #    resolution**2
    #)  # Approximate number of points in circle
#
    ## Number of valid pairs for each point: Area of circle with radius r
    #valid_pairs_per_point = (np.pi * reward_calculation_radius**2) / (resolution**2)
#
    ## Total valid pairs: each point can form valid pairs with other points
    #total_valid_pairs = valid_pairs_per_point * num_points_in_circle

    # Generate points within the circular workspace
    t = time.time()
    x_vals = np.arange(-workspace_radius, workspace_radius + resolution, resolution)
    y_vals = np.arange(-workspace_radius, workspace_radius + resolution, resolution)
    points = [
        (round(x / resolution) * resolution, round(y / resolution) * resolution)
        for x in x_vals
        for y in y_vals
        if x**2 + y**2 <= workspace_radius**2
    ]

    total_valid_pairs = 0
    for p1 in points:
        data_buffer = []
        x_vals = np.arange(
            p1[0] - reward_calculation_radius,
            p1[0] + reward_calculation_radius + resolution,
            resolution,
        )
        y_vals = np.arange(
            p1[1] - reward_calculation_radius,
            p1[1] + reward_calculation_radius + resolution,
            resolution,
        )

        for p2 in [
            (x, y)
            for x in x_vals
            for y in y_vals
            if ((x**2 + y**2 <= workspace_radius**2))
        ]:
            p2 = (
                round(p2[0] / resolution) * resolution,
                round(p2[1] / resolution) * resolution,
            )
            total_valid_pairs += 1

    print(f"Total time taken for [pre_calculate_valid_pairs()]: {time.time()-t}")

    return total_valid_pairs


def dump_dict_as_variables(data_dict, file_path, file_name, verbose):
    """Dumps each key-value pair in the dictionary as separate variables in a JSON file."""
    # Create the folder if it doesn't exist
    file_path.mkdir(parents=True, exist_ok=True)

    # Prepare the dictionary to store variables
    variables_dict = {}
    for key, value in data_dict.items():
        # Convert tuple keys to a valid string representation
        var_name = (
            str(key)
            .replace(" ", "")
            .replace(",", "_")
            .replace("(", "")
            .replace(")", "")
        )
        variables_dict[var_name] = value

    # Write the variables to the JSON file
    with open(file_path / file_name, "w") as file:
        json.dump(variables_dict, file, indent=4)

    if verbose:
        print(f"Reward map has been written to the file: {file_path / file_name}")


def parse_tuple(arg):
    """Convert a command-line argument of the form 'x,y' into a tuple (x, y)."""
    try:
        x, y = map(float, arg.strip("()").split(","))
        return (x, y)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Input point must be in the format: x,y (e.g., 0,0)"
        )
    
def str_to_bool(value):
    if isinstance(value, bool):
        return value  # Already a boolean
    if value.lower() in ("yes", "true", "t", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate and visualize workspace data."
    )

    parser.add_argument("--mode", type=str, default="generate", choices=["generate", "plot", "calculate_data_points"], required=True, help="Mode of script.")
    parser.add_argument("--verbose", type=bool, default=False, help="Enable verbose mode.")
    parser.add_argument("--model-id", type=str, help="Model ID to use.")
    parser.add_argument("--model-quant", type=str, default="fp16", help="Model quantization type.")
    parser.add_argument("--workspace-radius", type=int, help="Workspace radius.")
    parser.add_argument("--reward-calculation-radius", type=float, help="Reward calculation radius.")
    parser.add_argument("--resolution", type=float, help="Resolution step size.")
    parser.add_argument("--plot-after-generation", type=bool, default=False, help="Plot the grid after generation of the reward map.")
    parser.add_argument("--input-point", type=parse_tuple, default=(0, 0), help="The point around which to display the rewards.")
    parser.add_argument("--file-name", type=str, help="The name of the .json file from which to extract data for plot generation.")
    parser.add_argument("--show-plot-after-save", type=str_to_bool, default=True, help="Shows the plot after saving, when running in plot mode.")
    parser.add_argument("--prompt-file", type=str, help="Name of the prompt file.")


    args = parser.parse_args()

    # Assign variables from command-line arguments
    mode = args.mode
    verbose = args.verbose
    model_id = args.model_id
    model_quant = args.model_quant
    workspace_radius = args.workspace_radius
    reward_calculation_radius = args.reward_calculation_radius
    resolution = args.resolution
    plot_after_generation = args.plot_after_generation
    input_point = args.input_point
    file_name = args.file_name
    show_plot_after_save = args.show_plot_after_save
    prompt_file = args.prompt_file

    file_path = Path(__file__).parent.parent.parent / "src/FM3_MicRo/reward_maps"

    if verbose and (mode == "calculate_data_points" or mode == "generate"):
        print(
            f"Predicted length of generated dict: {pre_calculate_valid_pairs(workspace_radius, reward_calculation_radius, resolution, verbose)}"
        )

    if mode == "generate":
        total_run_time = time.time()
        file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_map_{model_id.lower()}_{prompt_file[11:-5]}_r_{reward_calculation_radius}_resolution_{resolution}.json"

        llm = LLM(
            model_id=model_id,
            model_quant=model_quant,
            device="cuda",
            verbose=False,
        )

        data_dict, all_points = generate_workspace_data(
            workspace_radius, resolution, reward_calculation_radius, llm, prompt_file, verbose
        )

        if verbose:
            print(f"Length of generated dict: {len(data_dict)}")
            print(f"Number of nodes in the worspace: {len(all_points)}\n")

        dump_dict_as_variables(data_dict, file_path, file_name, verbose)

    elif mode == "plot" or plot_after_generation:
        # Plot the reward map for a particular input point     
        visualize_data_from_json(file_path / file_name, input_point, show_plot_after_save, verbose)
