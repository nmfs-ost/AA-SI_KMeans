# `aa-kmap`

## 🔧 Program Behavior

The `aa-kmap` program performs **K-Means inter-frequency clustering** on acoustic data. It accepts either a **raw data file** or a **YAML/JSON configuration file** as its first (positional) argument. Based on the input type, the program runs in one of two modes:

- **Existing Configuration Mode**: Load and modify an existing YAML/JSON config.
- **New Configuration Mode**: Generate a new config from raw input data.

In both modes, **optional arguments** may override configuration settings, and the modified config is saved with the overrides encoded in the filename.

---

## 🗂️ 1. Load from YAML/JSON Config (Existing Configuration)

```bash
aa-kmap <config.yaml|config.json> 
  [--raw_path <str>] 
  [--nc_path <str>] 
  [--yaml_path <str>] 
  [--json_path <str>] 
  [--region_files <list of str>] 
  [--line_files <list of str>] 
  [--run_kmeans <bool>] 
  [--n_clusters <int>] 
  [--init <str>] 
  [--max_iter <int>] 
  [--n_init <int>] 
  [--random_state <int>] 
  [--frequency_list <list of str>] 
  [--pre_clustering_model <str>] 
  [--color_map <str>] 
  [--plot_clustermaps] 
  [--plot_echograms] 
  [--remove_noise] 
  [--ping_time_begin <datetime str>] 
  [--ping_time_end <datetime str>] 
  [--range_sample_begin <int>] 
  [--range_sample_end <int>] 
  [--data_reduction_type <str>] 
  [--ping_num <int>] 
  [--ping_time_bin <str>] 
  [--range_meter_bin <float>] 
  [--range_sample_num <int>] 
  [--save_path <str>]

```

- Loads settings from the specified configuration file.
- Executes the **KMeans Inter-Frequency Algorithm**, which includes:
  - `MIFRC`: Mean Inter-Frequency Response Clustering
  - `MADIFRC`: Mean Absolute Differences Inter-Frequency Response Clustering
- CLI arguments override config values as needed.
- A new configuration file is saved with modified parameters reflected in the filename.

---

## 📄 2. Create from `.raw` or `.nc` File (New Configuration)

```bash
aa-kmap <input.raw|input.nc> 
  [--raw_path <str>] 
  [--nc_path <str>] 
  [--yaml_path <str>] 
  [--json_path <str>] 
  [--region_files <list of str>] 
  [--line_files <list of str>] 
  [--run_kmeans <bool>] 
  [--n_clusters <int>] 
  [--init <str>] 
  [--max_iter <int>] 
  [--n_init <int>] 
  [--random_state <int>] 
  [--frequency_list <list of str>] 
  [--pre_clustering_model <str>] 
  [--color_map <str>] 
  [--plot_clustermaps] 
  [--plot_echograms] 
  [--remove_noise] 
  [--ping_time_begin <datetime str>] 
  [--ping_time_end <datetime str>] 
  [--range_sample_begin <int>] 
  [--range_sample_end <int>] 
  [--data_reduction_type <str>] 
  [--ping_num <int>] 
  [--ping_time_bin <str>] 
  [--range_meter_bin <float>] 
  [--range_sample_num <int>] 
  [--save_path <str>]

```

- Generates a base configuration from the raw/NetCDF file.
- CLI options customize the new configuration.
- A YAML file is saved with a timestamp or encoded parameter name.
- Use this mode for parameter exploration or building a config from scratch.

---

## 📌 Usage Overview

```bash
usage: aa-kmap <input_file.raw|.nc|config.yaml|.json> [options]
```

KMeans Inter-Frequency Clustering for Acoustic Data.

### 🔣 Positional Arguments

| Argument     | Description                                                   |
|--------------|---------------------------------------------------------------|
| `input_path` | Path to either a raw data file (`.raw`, `.nc`) or a config file (`.yaml`, `.json`). |

---

### ⚙️ Optional Arguments

| Argument                     | Description |
|-----------------------------|-------------|
| `-h`, `--help`              | Show help message and exit. |
| `--raw_path`                | Path to the input raw data file. |
| `--nc_path`                 | Path to NetCDF or processed data file. |
| `--yaml_path`               | Path to save or load the YAML configuration. |
| `--json_path`               | Path to save or load the JSON configuration. |
| `--region_files`            | List of `.EVR` region files to include. |
| `--line_files`              | List of `.EVL` line files to include. |

---

### 📊 KMeans Clustering Options

| Argument              | Description |
|-----------------------|-------------|
| `--run_kmeans`        | Flag to run the KMeans algorithm (`true`, `false`, etc.). |
| `--n_clusters`        | Number of clusters to create (default: `8`). |
| `--init`              | Initialization method (default: `k-means++`). |
| `--max_iter`          | Max iterations for KMeans (default: `300`). |
| `--n_init`            | Number of times KMeans is run (default: `10`). |
| `--random_state`      | Random seed (default: `42`). |
| `--frequency_list`    | List of frequencies to use, e.g., `38kHz 70kHz 120kHz`. |

---

### 🧠 Pre-Clustering Model

| Argument                  | Description |
|---------------------------|-------------|
| `--pre_clustering_model` | Pre-clustering model type (`MADIFRC` by default). |

---

### 🎨 Plotting Options

| Argument              | Description |
|-----------------------|-------------|
| `--color_map`         | Matplotlib colormap (default: `jet`). |
| `--plot_clustermaps`  | Plot the cluster maps. |
| `--plot_echograms`    | Plot the echograms. |

---

### 🔇 Noise Removal

| Argument           | Description |
|--------------------|-------------|
| `--remove_noise`   | Enable noise removal logic. |

---

### 🧭 Ping & Range Selection

| Argument              | Description |
|-----------------------|-------------|
| `--ping_time_begin`   | Start time for ping sub-selection. |
| `--ping_time_end`     | End time for ping sub-selection. |
| `--range_sample_begin`| Starting range sample index. |
| `--range_sample_end`  | Ending range sample index. |

---

### 📉 Data Reduction Options

| Argument                 | Description |
|--------------------------|-------------|
| `--data_reduction_type`  | Type of reduction, e.g., `sample_number`. |
| `--ping_num`             | Number of pings to include (default: `1`). |
| `--ping_time_bin`        | Time binning interval (default: `2S`). |
| `--range_meter_bin`      | Range bin size in meters (default: `2`). |
| `--range_sample_num`     | Number of samples per range bin (default: `1`). |

---

### 💾 Output

| Argument        | Description |
|-----------------|-------------|
| `--save_path`   | Directory to save output, config, and plots. If omitted, current working directory is used. |
