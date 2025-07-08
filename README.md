
# aa-kmap

## 🔧 Program Behavior

This program accepts either a **raw data file** or a **YAML/JSON configuration file** as its first (positional) argument. Based on the type of input provided, the program branches into one of two execution modes. In both cases, optional CLI arguments can be supplied to **override settings**, and a new configuration file is saved with those overrides encoded in the filename.

---

### 🗂️ 1. Using a YAML or JSON Configuration

```bash
$ aa-kmap <config_file.yaml|.json> [--n_clusters <int>] [--frequency_list <list>] [--save_path <output_path>]
```

- Loads the configuration from the specified file.
- Runs the **KMEANSInterFrequency Algorithm**, which includes:
  - **MFRC**: Mean Frequency Response Clustering
  - **MADFRC**: Mean Absolute Difference Frequency Response Clustering
- Any additional CLI arguments act as **overrides** to the loaded configuration.
- A new configuration file is saved, with parameter changes embedded in the filename (e.g., cluster count or frequency list).
- This allows for incremental refinement through a traceable chain of configuration versions.

---

### 📄 2. Using a Raw File / NC File

```bash
$ aa-kmap <input_file.raw|.nc> [--n_clusters <int>] [--frequency_list <list>] [--save_path <output_path>]
```

- Automatically generates a base configuration from the raw data.
- Optional arguments customize the generated configuration.
- A new YAML file is created and saved, usually with a timestamp.
- This mode is ideal for initial exploration and parameter tuning from scratch.

> 📝 In both modes, the program handles all config and metadata file creation internally — users only need to manage file paths.

---


usage: aa-kmap <raw_file|config.yaml|config.json> [options]

KMeans Inter-Frequency Mapper for Acoustic Data Clustering

positional arguments:
  input_path            Path to either a raw file or a YAML/JSON configuration file.

optional arguments:
  -h, --help            Show this help message and exit.
  --raw_path RAW_PATH   Path to the input raw data file.
  --nc_path NC_PATH     Path to NetCDF or processed data file.
  --yaml_path YAML_PATH Path to save or load the YAML configuration.
  --region_files REGION_FILES [REGION_FILES ...]
                        List of EVR region files to include.
  --line_files LINE_FILES [LINE_FILES ...]
                        List of EVL line files to include.

KMeans Options:
  --run_kmeans          Flag to execute the KMeans clustering algorithm.
  --n_clusters N_CLUSTERS
                        Number of clusters to generate (default: 8).
  --init INIT_METHOD    Initialization method for centroids (default: k-means++).
  --max_iter MAX_ITER   Maximum number of iterations for a single run (default: 300).
  --n_init N_INIT       Number of time the k-means algorithm will be run (default: 10).
  --random_state RANDOM_STATE
                        Seed for reproducibility (default: 42).
  --frequency_list FREQUENCY_LIST [FREQUENCY_LIST ...]
                        List of frequencies to use, e.g., 38kHz 70kHz 120kHz.

Pre-clustering Model:
  --pre_clustering_model MODEL_NAME
                        Pre-clustering model type (default: DIRECT).

Plotting Options:
  --color_map COLOR_MAP
                        Matplotlib colormap to use (default: jet).
  --plot_clustermaps    Plot and save cluster maps.
  --plot_echograms      Plot and save echograms.

Noise Removal:
  --remove_noise        Enable noise removal.

Ping & Range Selection:
  --ping_time_begin PING_TIME_BEGIN
  --ping_time_end PING_TIME_END
  --range_sample_begin RANGE_SAMPLE_BEGIN
  --range_sample_end RANGE_SAMPLE_END

Data Reduction Options:
  --data_reduction_type TYPE
                        Type of data reduction (e.g., sample_number).
  --ping_num PING_NUM   Number of pings to include (default: 1).
  --ping_time_bin TIME_BIN
                        Binning interval for ping time (default: 2S).
  --range_meter_bin METER_BIN
                        Binning resolution for range in meters (default: 2).
  --range_sample_num SAMPLE_NUM
                        Number of range samples (default: 1).

Output:
  --save_path SAVE_PATH
                        Path to save the resulting configuration or processed file.



## 🧰 Available Configuration Options/Overrides

```yaml
raw_path: str
nc_path: str
yaml_path: str

region_files:
  - "resources/zooplankton_patch.EVR"
  - "resources/zooplankton_patch1.EVR"
  - "resources/zooplankton_patch2.EVR"

line_files:
  - "resources/zooplankton_patch0.EVL"
  - "resources/zooplankton_patch2.EVL"

run_kmeans: true
n_clusters: 8
init: "k-means++"
max_iter: 300
n_init: 10
random_state: 42

frequency_list:
  - "38kHz"
  - "70kHz"
  - "120kHz"
  - "18kHz"
  - "200kHz"

pre_clustering_model:
  model: "DIRECT"

color_map: "jet"
plot_clustermaps: true
plot_echograms: true

noise_removal:
  remove_noise: false

ping_time_begin: null
ping_time_end: null
range_sample_begin: null
range_sample_end: null

data_reduction_type: "sample_number"
ping_num: 1
ping_time_bin: "2S"
range_meter_bin: 2
range_sample_num: 1
```