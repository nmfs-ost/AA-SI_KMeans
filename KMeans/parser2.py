# Imports for command-line parsing, file operations, config loading, and clustering
import argparse
import os
import sys
import yaml  # For loading YAML config files
import json  # For loading JSON config files
from pathlib import Path  # For working with file system paths
from KMeans.KMeans import KMClusterMap  # Importing the clustering function/class (assuming it's defined in echoml)
from loguru import logger # For logging (optional, can be used for debugging)
from echopype import __version__ as echopype_version  # Importing echopype version
from matplotlib import __version__ as matplotlib_version  # Importing matplotlib version
from numpy import __version__ as numpy_version  # Importing numpy version
from pandas import __version__ as pandas_version  # Importing pandas version
from xarray import __version__ as xarray_version  # Importing xarray version
from scipy import __version__ as scipy_version  # Importing scipy version
from sklearn import __version__ as scikit_learn_version  # Importing scikit-learn version
# Dummy implementation of KMClusterMap for debug/testing purposes
# In actual use, remove this and rely on the real import above

# Loads a config file (.yaml or .json) and returns it as a Python dictionary
def load_config(config_path):
    ext = Path(config_path).suffix  # Get the file extension
    with open(config_path, 'r') as f:
        if ext == ".yaml" or ext == ".yml":
            return yaml.safe_load(f)  # Load YAML config
        elif ext == ".json":
            return json.load(f)  # Load JSON config
        else:
            raise ValueError("Unsupported config file type: must be .yaml or .json")

def save_config(config, save_path):
    save_path = Path(save_path).resolve()
    print(save_path)
    # Define the directory where the YAML will be saved
    encoded_dir = save_path / "ENCODED_DIRECTORY"
    encoded_dir.mkdir(parents=True, exist_ok=True)

    # Decide filename (optional: base it on input_path or fixed name)
    yaml_filename = "ENCODED_YAML.yaml"

    yaml_path = encoded_dir / yaml_filename

    logger.debug(f"Saving config to {yaml_path} (YAML) derived from {config.get('input_path', 'unknown')}")
    
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)



def save_yaml(data, output_path):
    """
    Save a dictionary as a YAML file.

    Args:
        data (dict): Data to save.
        output_path (str): Path to save the YAML file.
    """
    # Ensure the kmap_exports directory exists
    default_export_dir = os.path.expanduser("~/home/kmap_exports")
    os.makedirs(default_export_dir, exist_ok=True)

    # Ensure the directory for the output path exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(default_export_dir, exist_ok=True)

    with open(output_path, "w") as  file:
        yaml.dump(data, file, default_flow_style=False)
    print(f"Updated YAML saved to: {output_path}")



import argparse

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="KMeans Inter-Frequency Clustering for Acoustic Data",
        epilog="Example usage:\n"
               "  python script.py /path/to/config.yaml --run_kmeans --n_clusters 4\n"
               "  python script.py /data/input.raw --frequency_list 38 120 --plot_echograms\n",
        formatter_class=argparse.RawTextHelpFormatter  # Allow multi-line help/epilog
    )

    # Positional
    parser.add_argument("input_path", help="(str) Path to .raw/.nc file or config.yaml/config.json\n"
                                           "  e.g., /data/input.raw")

    # Optional paths
    parser.add_argument("--raw_path", help="(str) Override .raw path\n  e.g., --raw_path /data/file.raw")
    parser.add_argument("--nc_path", help="(str) Override .nc path\n  e.g., --nc_path /data/file.nc")
    parser.add_argument("--yaml_path", help="(str) Load config from YAML\n  e.g., --yaml_path config.yaml")
    parser.add_argument("--json_path", help="(str) Load config from JSON\n  e.g., --json_path config.json")
    parser.add_argument("--region_files", nargs='+',
                        help="(list of str) Paths to region files\n  e.g., --region_files reg1.reg reg2.reg")
    parser.add_argument("--line_files", nargs='+',
                        help="(list of str) Paths to line files\n  e.g., --line_files line1.json line2.json")

    # KMeans options
    parser.add_argument("--run_kmeans", action="store_false", default=True,
                        help="(bool) Run KMeans (default: True; set this flag to disable)")
    parser.add_argument("--n_clusters", type=int,
                        help="(int) Number of clusters\n  e.g., --n_clusters 4")
    parser.add_argument("--init", help="(str) Initialization method\n  e.g., --init k-means++")
    parser.add_argument("--max_iter", type=int,
                        help="(int) Max iterations\n  e.g., --max_iter 300")
    parser.add_argument("--n_init", type=int,
                        help="(int) Number of initializations\n  e.g., --n_init 10")
    parser.add_argument("--random_state", type=int,
                        help="(int) Random seed\n  e.g., --random_state 42")
    parser.add_argument("--frequency_list", nargs='+',
                        help="(list of str) Frequencies to include\n  e.g., --frequency_list 38 120")

    # Visualization and preprocessing
    parser.add_argument("--pre_clustering_model", help="(str) Pre-model to apply\n  e.g., --pre_clustering_model PCA")
    parser.add_argument("--color_map", help="(str) Matplotlib colormap\n  e.g., --color_map viridis")
    parser.add_argument("--plot_clustermaps", action="store_false", default=True,
                        help="(bool) Plot cluster maps (default: True; set this flag to disable)")
    parser.add_argument("--plot_echograms", action="store_false", default=True,
                        help="(bool) Plot echograms (default: True; set this flag to disable)")
    parser.add_argument("--remove_noise", action="store_false", default=True,
                        help="(bool) Remove noise (default: True; set this flag to disable)")

    # Filtering
    parser.add_argument("--ping_time_begin", help="(str) Start time (ISO)\n  e.g., --ping_time_begin 2020-01-01T00:00:00")
    parser.add_argument("--ping_time_end", help="(str) End time (ISO)\n  e.g., --ping_time_end 2020-01-01T01:00:00")
    parser.add_argument("--range_sample_begin", type=int,
                        help="(int) Start sample index\n  e.g., --range_sample_begin 100")
    parser.add_argument("--range_sample_end", type=int,
                        help="(int) End sample index\n  e.g., --range_sample_end 500")

    # Data reduction
    parser.add_argument("--mvbs_data_reduction_type",
                        help="(str) Reduction type\n  e.g., --mvbs_data_reduction_type ping")
    parser.add_argument("--mvbs_ping_num", type=int,
                        help="(int) Number of pings to average\n  e.g., --mvbs_ping_num 5")
    parser.add_argument("--mvbs_ping_time_bin",
                        help="(str) Time bin size\n  e.g., --mvbs_ping_time_bin 5s")
    parser.add_argument("--mvbs_range_meter_bin", type=float,
                        help="(float) Range bin size in meters\n  e.g., --mvbs_range_meter_bin 1.0")
    parser.add_argument("--mvbs_range_sample_num", type=int,
                        help="(int) Range samples per bin\n  e.g., --mvbs_range_sample_num 20")

    # Save path
    parser.add_argument("--save_path", help="(str) Output path\n  e.g., --save_path results/output.yaml")

    return parser.parse_args()


def override_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    args_dict = vars(args)  # Converts Namespace to dict
    print(f"Overriding config with args: {args_dict}")
    for key, val in args_dict.items():
        if val is not None:
            # For lists, only override if it's not empty
            if isinstance(val, list) and not val:
                continue
            config[key] = val

    return config

# Main entry point of the script
def main():
    args = parse_args()  # Parse command-line arguments
    input_path = Path(args.input_path)

    # Load or create the initial config based on input_path
    if input_path.suffix in ['.yaml', '.yml']:
        config = load_config(input_path)  # Load existing config
        logger.debug(f"Loaded config from {input_path}")
        
        yaml_path = str(input_path) if input_path.suffix in ['.yaml', '.yml'] else None
        
    elif input_path.suffix in ['.raw', '.nc']:
        yaml_path = "PREPARING_PATH"
    # If the input is a .raw or .nc file, set the appropriate path variable
        if input_path.suffix == '.nc':
            nc_path = str(input_path)
            raw_path = None
        else:
            raw_path = str(input_path)
            nc_path = None
        logger.debug(f"Creating minimal config for input file: {input_path}")
        # Create a minimal config if a data file is provided directly
        config = {
            "color_map": "jet",
            "echopype_version": echopype_version,
            "filename": "<name><runkmeans><n_clusters><init><max_iter><n_init><random_state><frequency_list><pre_clustering_model><plot_clustermaps><plot_echograms><remove_noise><ping_time_begin><ping_time_end><range_sample_begin><range_sample_end><data_reduction_type><ping_num><ping_time_bin><range_meter_bin><range_sample_num>",
            "frequency_list": [
                "38kHz",
                "70kHz",
                "120kHz",
                "18kHz",
                "200kHz"
            ],
            "init": "k-means++",
            "input_path": str(input_path),
            "line_files": [
            ],
            "matplotlib_version": matplotlib_version,
            "max_iter": 300,
            "mvbs_data_reduction_type": "sample_number",
            "mvbs_ping_num": 58,
            "mvbs_ping_time_bin": "2S",
            "mvbs_range_meter_bin": 2,
            "mvbs_range_sample_num": 1,
            "n_clusters": 8,
            "n_init": 10,
            "nc_path": nc_path,
            "numpy_version": numpy_version,
            "pandas_version": pandas_version,
            "ping_time_begin": None,
            "ping_time_end": None,
            "plot_clustermaps": True,
            "plot_echograms": True,
            "pre_clustering_model": "DIRECT",
            "python_version": "3.8.10",
            "random_state": 42,
            "range_sample_begin": None,
            "range_sample_end": None,
            "raw_path": raw_path,
            "region_files": [
            ],
            "remove_noise": False,
            "run_kmeans": True,
            "save_path": "/home/mryan/Documents/GitHub/AA-SI_KMeans/examples/aa-kmap_exports",
            "scikit_learn_version": scikit_learn_version,
            "scipy_version": scipy_version,
            "xarray_version": xarray_version,
            "yaml_path": yaml_path,
        }
        
        
    else:
        print("Unsupported file type for input_path.")
        sys.exit(1)

    # Ensure config reflects the input path for config files
    if input_path.suffix in [".yaml", ".yml"]:
        config["yaml_path"] = str(input_path)
        args.yaml_path = str(input_path)  # Set args.yaml_path for consistency
        logger.debug(f"Config file path set to: {config['yaml_path']}")
    if input_path.suffix == ".json":
        config["json_path"] = str(input_path)
        args.json_path = str(input_path)  # Set args.json_path for consistency
        logger.debug(f"Config file path set to: {config['json_path']}")
        
    


    # Save updated config if output path is specified
    #if args.yaml_path or args.json_path:
        
    #    save_path = args.yaml_path or args.json_path
    
    config = override_config_with_args(config, args)
    save_config(config, config["save_path"])

        # Step 3: Override config
    
    logger.debug(f"Saved updated config to {config['save_path']}")

    print(json.dumps(config, indent=2))
    
    # Run the clustering if specified by flag or config
    if args.run_kmeans or config.get("run_kmeans", True):
        print("Running KMeans clustering with the following configuration:")
        KMClusterMap(
            # Path Configuration
            file_path = config.get('input_path'),
            save_path = config.get('save_path'),

            # KMeans Configuration
            frequency_list = config.get('frequency_list'),
            cluster_count = config.get('n_clusters'),
            random_state = config.get('random_state'),

            # Pre-Clustering Model
            model = config.get('model', config.get('pre_clustering_model', 'DIRECT')),

            # Plotting
            color_map = config.get('color_map', 'viridis'),
            plot_echograms = config.get('plot_echograms', True),

            # MVBS & Data Reduction
            data_reduction_type = config.get('data_reduction_type', config.get('mvbs_data_reduction_type')),
            range_meter_bin = config.get('range_meter_bin', config.get('mvbs_range_meter_bin')),
            ping_time_bin = config.get('ping_time_bin', config.get('mvbs_ping_time_bin')),
            range_sample_num = config.get('range_sample_num', config.get('mvbs_range_sample_num')),
            ping_num = config.get('ping_num', config.get('mvbs_ping_num')),

            # Noise Removal
            remove_noise = config.get('remove_noise', False),

            # Subset Selection
            ping_time_begin = config.get('ping_time_begin'),
            ping_time_end = config.get('ping_time_end'),
            range_sample_begin = config.get('range_sample_begin'),
            range_sample_end = config.get('range_sample_end'),
        )

# Standard Python entry point
if __name__ == "__main__":
    main()
 