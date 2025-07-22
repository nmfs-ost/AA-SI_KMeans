# Imports for command-line parsing, file operations, config loading, and clustering
import argparse
from calendar import c
import os
import sys
import yaml  # For loading YAML config files
import json  # For loading JSON config files
from pathlib import Path  # For working with file system paths
from KMeans import KMClusterMap  # Importing the clustering function/class from KMeans/__init__.py or KMeans/KMeans.py
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
        else:
            raise ValueError("Unsupported config file type: must be .yaml or .yml")

def save_config(config, save_path, name):
    save_path = Path(save_path).resolve()
    # Define the directory where the YAML will be saved
    asset_path = save_path / name
    asset_path.mkdir(parents=True, exist_ok=True)

    # Decide filename (optional: base it on input_path or fixed name)
    yaml_filename = name + ".yaml"

    yaml_path = asset_path / yaml_filename
    config["asset_path"] = str(asset_path)  # Store the directory in config
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
    parser.add_argument("--sonar_model", help="(str) Echosounder type (e.g., EK60, EK80, EM2040)\n"
                                           "  e.g., EK60")
    parser.add_argument("--raw_path", help="(str) Override .raw path\n  e.g., --raw_path /data/file.raw")
    parser.add_argument("--nc_path", help="(str) Override .nc path\n  e.g., --nc_path /data/file.nc")
    parser.add_argument("--yaml_path", help="(str) Load config from YAML\n  e.g., --yaml_path config.yaml")
    parser.add_argument("--name", help="(str) Name for saving outputs\n  e.g., --name my_output")
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
    parser.add_argument("--precluster_model", help="(str) Pre-model to apply\n  e.g., --precluster_model PCA")
    parser.add_argument("--color_map", help="(str) Matplotlib colormap\n  e.g., --color_map viridis")
    parser.add_argument("--plot_clustergrams", action="store_false", default=True,
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

    # MVBS options
    parser.add_argument("--ping_num", type=int,
                        help="(int) Number of pings to average\n  e.g., --mvbs_ping_num 5")
    parser.add_argument("--ping_time_bin",
                        help="(str) Time bin size\n  e.g., --mvbs_ping_time_bin 5s")
    parser.add_argument("--range_meter_bin", type=float,
                        help="(float) Range bin size in meters\n  e.g., --mvbs_range_meter_bin 1.0")
    parser.add_argument("--range_sample_num", type=int,
                        help="(int) Range samples per bin\n  e.g., --range_sample_num 20")
    parser.add_argument("--range_bin", type=str, help="(str) Range binning method\n  e.g., --range_bin 20m")
    parser.add_argument("--range_var", type=str, help="(str) Range variable name\n  e.g., --range_var echo_range")
    # Save path
    parser.add_argument("--save_path", help="(str) Output path\n  e.g., --save_path results/output.yaml")
    
    parser.add_argument("--save_nc", action="store_true", default=False, help="(bool) Save results as .nc file (default: False; set this flag to enable)")
    parser.add_argument("--save_echograms", action="store_true", default=True, help="(bool) Save echograms (default: True; set this flag to disable)")
    parser.add_argument("--save_clustergrams", action="store_true", default=True, help="(bool) Save cluster maps (default: True; set this flag to disable)")
    parser.add_argument("--echogram_color_map", help="(str) Colormap for echograms\n  e.g., --echogram_color_map viridis")
    parser.add_argument("--clustergram_color_map", help="(str) Colormap for cluster maps\n  e.g., --clustergram_color_map plasma")
    
    
    return parser.parse_args()


def override_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    args_dict = vars(args)  # Converts Namespace to dict
    
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
        config["yaml_path"] = yaml_path  # Set yaml_path in config
    
    elif input_path.suffix in ['.raw', '.nc']:

        # If the input is a .raw or .nc file, set the appropriate path variable

        logger.debug(f"Creating minimal yaml config for input file: {input_path}")
        # Create a minimal config if a data file is provided directly
        yaml_file = Path(__file__).parent / "default_config.yaml" #TODO derive names from raw minus extension.
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)
        
        if input_path.suffix == '.nc':
            nc_path = str(input_path)
            raw_path = None
            config["input_path"] = nc_path
            config["nc_path"] = nc_path  # Set nc_path in config
            
        else:
            raw_path = str(input_path)
            nc_path = None
            config["input_path"] = raw_path
            config["raw_path"] = raw_path  # Set raw_path in config
        
        config["name"] = input_path.stem
        config["yaml_path"] = str((Path(config['save_path']) / f"{config['name']}" / f"{config['name']}.yaml").resolve())


    else:
        logger("Unsupported file type for input_path. Must be .yaml, .yml, .raw, or .nc")
        sys.exit(1)
        
    
    
    config = override_config_with_args(config, args)

    save_config(config, config["save_path"], config["name"])

        # Step 3: Override config
    
    logger.debug(f"Saved updated config to {config['save_path']}")
    # Log the config for debugging
    logger.info("Configuration post-overrides:")
    logger.info(json.dumps(config, indent=4))

    # Run the clustering if specified by flag or config
    if args.run_kmeans or config.get("run_kmeans", True):

        KMClusterMap(config["yaml_path"]) # So this is the new configuration that was just saved. This is one reason why its own path is self-referential


# Standard Python entry point
if __name__ == "__main__":
    main()
 