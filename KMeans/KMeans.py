#!/usr/bin/python

# System tools.

import os, sys, json

# Data science tools.

import numpy as np
import pandas as pd
import xarray as xr

import hvplot.xarray  # ensure hvplot is enabled
import holoviews as hv

# Machine Learning tools.

from sklearn.cluster import KMeans
from sklearn import preprocessing
from itertools import chain, combinations

# Echosounder tools.

import echopype as ep
import echoregions as er

# Plotting tools.

import matplotlib.pyplot as plt
import itertools

# Logging tools.

from loguru import logger
import yaml

pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)

class FrequencyData():
    """Given some dataset 'Sv', list all frequencies available. This class offers methods which help map out frequencies and channels plus additional utilities. 
    """   
    
    def __init__(self, Sv):
        """Initializes class object and parses the frequencies available within the echdata object (xarray.Dataset) 'Sv'.

        Args:
            Sv (xarray.Dataset): The 'Sv' echodata object.
        """
        
        self.Sv = Sv # Crreate a self object.
        self.frequency_list = [] # Declares a frequency list to be modified.
        
        self.construct_frequency_list() # Construct the frequency list.
        #TODO : This string needs cleaning up ; remove unneeded commas and empty tuples.
        self.frequency_set_combination_list = self.construct_frequency_set_combination_list() # Constructs a list of available frequency set permutations. Example : [('18 kHz',), ('38 kHz',), ('120 kHz',), ('200 kHz',), ('18 kHz', '38 kHz'), ('18 kHz', '120 kHz'), ('18 kHz', '200 kHz'), ('38 kHz', '120 kHz'), ('38 kHz', '200 kHz'), ('120 kHz', '200 kHz'), ('18 kHz', '38 kHz', '120 kHz'), ('18 kHz', '38 kHz', '200 kHz'), ('18 kHz', '120 kHz', '200 kHz'), ('38 kHz', '120 kHz', '200 kHz'), ('18 kHz', '38 kHz', '120 kHz', '200 kHz')]
        # print(self.frequency_set_combination_list)
        self.frequency_pair_combination_list = self.construct_frequency_pair_combination_list() # Constructs a list of all possible unequal permutation pairs of frequencies. Example : [('18 kHz', '38 kHz'), ('18 kHz', '120 kHz'), ('18 kHz', '200 kHz'), ('38 kHz', '120 kHz'), ('38 kHz', '200 kHz'), ('120 kHz', '200 kHz')] 
        # print(self.frequency_pair_combination_list)
        
    def construct_frequency_list(self):
        """Parses the frequencies available in the xarray 'Sv'
        """
        for i in range(len(self.Sv.Sv)): # Iterate through the natural index associated with Sv.Sv .
            
            self.frequency_list.append(str(self.Sv.Sv[i].coords.get("channel")).split(" kHz")[0].split("GPT")[1].strip()+" kHz") # Extract frequency.
            
        return self.frequency_list # Return string array frequency list of the form [18kHz, 70kHz, 200 kHz]
        
        
    def powerset(self, iterable):
        """Generates combinations of elements of iterables ; powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

        Args:
            iterable (_type_): A list.

        Returns:
            _type_: Returns combinations of elements of iterables.
        """        
        
        s = list(iterable) # Make a list from the iterable.
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1)) # Returns a list of tuple elements containing combinations of elements which derived from the iterable object.


    def construct_frequency_set_combination_list(self):
        """Constructs a list of available frequency set permutations. 
        Example : [
            ('18 kHz',), ('38 kHz',), ('120 kHz',), ('200 kHz',), ('18 kHz', '38 kHz'),
            ('18 kHz', '120 kHz'), ('18 kHz', '200 kHz'), ('38 kHz', '120 kHz'), ('38 kHz', '200 kHz'),
            ('120 kHz', '200 kHz'), ('18 kHz', '38 kHz', '120 kHz'), ('18 kHz', '38 kHz', '200 kHz'),
            ('18 kHz', '120 kHz', '200 kHz'), ('38 kHz', '120 kHz', '200 kHz'),
            ('18 kHz', '38 kHz', '120 kHz', '200 kHz')
            ]


        Returns:
            list<tuple>: A list of tuple elements containing frequency combinations which is useful for the KMeansOperator class.
        """  
              
        return list(self.powerset(self.frequency_list)) # Returns a list of tuple elements containing frequency combinations which is useful for the KMeansOperator class.
    
    
    def print_frequency_set_combination_list(self):
        """Prints frequency combination list one element at a time.
        """
                
        for i in self.frequency_set_combination_list:  # For each frequency combination associated with Sv.
            print(i) # Print out frequency combination tuple.
            
            
    def construct_frequency_pair_combination_list(self):
        """Returns a list of tuple elements containing frequency combinations which is useful for the KMeansOperator class.

        Returns:
            list<tuple>: A list of tuple elements containing frequency combinations which is useful for the KMeansOperator class.
        """  
              
        return list(itertools.combinations(self.frequency_list, 2)) # Returns a list of tuple elements containing frequency combinations which is useful for the KMeansOperator class.
    
    def print_frequency_pair_combination_list(self):
        """Prints frequency combination list one element at a time.
        """
                
        for i in self.frequency_pair_combination_list:  # For each frequency combination associated with Sv.
            print(i) # Print out frequency combination tuple.
            

    def print_frequency_list(self):
        """Prints each frequency element available in Sv.
        """        
        
        for i in self.frequency_list:# For each frequency in the frequency_list associated with Sv.
            print(i) # Print out the associated frequency.
            



class KMeansOperator: # Reference: https://medium.datadriveninvestor.com/unsupervised-learning-with-python-k-means-and-hierarchical-clustering-f36ceeec919c
    
    def __init__(self, Sv, channel_list = None, k = None, random_state = None, n_init = 10, max_iter = 300, frequency_list = None, precluster_model = "DIRECT"): # TODO Need to take in channel list instead of query matrix.
        """_summary_

        Args:
            Sv (_type_): _description_
            channel_list (_type_, optional): _description_. Defaults to None.
            k (_type_, optional): _description_. Defaults to None.
            random_state (_type_, optional): _description_. Defaults to None.
            frequency_list (_type_, optional): _description_. Defaults to None.
        """
        self.channel_list = channel_list # If the user chooses, they may provide a channel list insead of a specify frequency set.
        self.frequency_set_string = "" # Declare a frequency set string for simple labeling purpose with small dewscriptions of frequencies applied to kmeans.
        
        self.Sv = Sv # Echodata xarray object.
        self.frequency_list = frequency_list # Make a class object from frequency_list that was passed.
        self.simple_frequency_list = frequency_list
        self.k = k # KMeans configuration variable. The cluster count.
        self.random_state = random_state # Class variable 'random_state' is a general kmeans parameter.
        self.precluster_model = precluster_model # KMeans configuration variable. Pre-clustering DF model. Constructed from Sv.Sv and dictates which dataframe is fed into the KMeans clustering operation.
        self.n_init = n_init # KMeans configuration variable. 
        self.max_iter = max_iter # KMeans configuration variable. Max iterations.
        
        # If no frequency_list is provided, check if a channel_list is provided
        if self.frequency_list is None:
            if self.channel_list is not None:
                # If channel_list is provided, construct frequency_list from it
                self.frequency_list = []
                self.construct_frequency_list(frequencies_provided=False)
                self.construct_frequency_set_string()
                self.assign_sv_clusters()
            else:
                # If neither frequency_list nor channel_list is provided, raise an error
                raise ValueError("Provide a frequency_list or channel_list input parameter.")
        else:
            # If frequency_list is provided, use it to construct the internal frequency_list
            self.construct_frequency_list(frequencies_provided=True)
            self.construct_frequency_set_string()
            self.assign_sv_clusters()

        
    def construct_frequency_list(self, frequencies_provided):
        """Either using a channel_list or a frequency_list this function provides one which satisfies all requirements of this class structure. In particular the channels and frequencies involved have to be known and mapped to oneanother.

        Args:
            frequencies_provided (boolean): was a frequency_list provided at object creation? If so then 'True' if a channel_list instead was used then 'False'.
        """        
        if frequencies_provided == True:
            self.simple_frequency_list = self.frequency_list
            self.frequency_list = [] # Declare a frequency list to be populated with string frequencies of the form [[1,'38kHz'],[2,'120kHz'],[4,'200kHz']] where the first element is meant to be the channel representing the frequency. This is an internal object. Do not interfere.
            for j in self.simple_frequency_list: # For each frequency 'j'.
                for i in range(len(self.Sv.Sv)): # Check each channel 'i'.
                    if str(self.Sv.Sv[i].coords.get("channel")).split(" kHz")[0].split("GPT")[1].strip() == j.split("kHz")[0].strip(): # To see if the channel associates with the frequency 'j' .
                        self.frequency_list.append([i,str(self.Sv.Sv[i].coords.get("channel")).split(" kHz")[0].split("GPT")[1].strip()+" kHz"]) # If so append it and the channel to the 'frequency_list'.
        else:
            for i in self.channel_list:
                self.frequency_list.append([i,str(self.Sv.Sv[i].coords.get("channel")).split(" kHz")[0].split("GPT")[1].strip()+" kHz"])
                
            
    def construct_frequency_set_string(self):
        """So the idea behind a frequency_set_string is to serve as a quick and simple representative string which describes the kmeans clustering map and the frequency data that was employed to create it.
        """        
        frequency_set_string = "" # Declare frequency_set_string. We call it a set even though it is a list is becasue it is not meant to change but the list aspect was useful because it allowed mutability.
        for i in self.frequency_list: # For each frequency in the frequency_list.
            frequency_set_string += i[1].split(" ")[0]+"kHz,"

        frequency_set_string = "<("+frequency_set_string+"," # Start defining the frequency_set_string.
        frequency_set_string = frequency_set_string.split(",,")[0]+")>" # Finishing defining the frequency_set_string.
        self.frequency_set_string = frequency_set_string # Make 'frequency_set_string' a class object 'self.frequency_set_string' .
        
        
    def construct_pre_clustering_df(self):
        """This is the dataframe which is passed to KMeans algorithm and is operated on. This df is synthesized by taking the Sv(s) associated with various frequencies.

        Returns:
            pd.DataFrame: The dataset which is directly fed into KMeans.
        """        
        pre_clustering_df = pd.DataFrame() # Declare empty df which will eventually conatin columns of 'Sv' value columns ripped from DataFrames which were converted from DataArrays. This is like a flattening of dimensionalities and allows 'Sv' to be represented as a single column per frequency.
        sv_frequency_map_list = [] # Declare empty list to conatin equivilent copies of the clustering data through iteration. This redundancy is tolerated becasue it allows a clean mapping give then
        sv_frequency_absolute_difference_map_list = []
        
        self.frequency_pair_combination_list = list(itertools.combinations(self.frequency_list, 2))
        
        if self.precluster_model == "DIRECT": # The DIRECT clustering model clusters direct Sv.Sv values. 
 
 
            for i in self.frequency_list: # Need a channel mapping function.
                channel_df = self.Sv.Sv[i[0]].to_dataframe(name=None, dim_order=None) # Convert Sv.Sv[channel] into a pandas dataframe.
                channel_df.rename(columns = {'Sv':str(self.Sv.Sv[i[0]].coords.get("channel")).split("kHz")[0].split("GPT")[1].strip()+"kHz"}, inplace = True) # Rename the column to the frequency associated with said channel. This value is pulled from the xarray.
                sv_frequency_map_list.append(channel_df[str(self.Sv.Sv[i[0]].coords.get("channel")).split("kHz")[0].split("GPT")[1].strip()+"kHz"]) # Append columns with each frequency or channel. (It is always best to retain a map between channels and frequencies.)
                
            pre_clustering_df = pd.concat(sv_frequency_map_list, axis = 1) # Creats a new dataframe from the values previously constructed. This is done to keep it steril.
        
        if self.precluster_model == "ABSOLUTE_DIFFERENCES": # The ABSOLUTE_DIFFERENCES clustering model clusters the absolute value of the differences between a pair permutation of frequency based Sv.Sv values. 
            # Each pair permutation is given it's own column within the dataframe that is fed into KMeans in the same way that with DIRECT each frequency is given it's own column within the dateframe that is fed into KMeans.
            # In other words this model was built to solve the problems of DIRECT by not allowing identical frequencies to be clustered together meaningfully becasue there should no no new information produced by that. 
            # If you attempt to feed the same frequencies in you will get a blank screen. This mean that 100% of the visual information is meaningful.
        
            for i in self.frequency_pair_combination_list:
                # For each pair of frequencies, compute the absolute difference of their Sv values.
                # 1. Get the Sv values for each channel in the pair, convert to DataFrame, and extract the "Sv" column.
                # 2. Subtract the two Sv arrays, take the absolute value.
                # 3. Build a column name describing the operation, e.g., "abs(Sv(38kHz)-Sv(120kHz))".
                # 4. Create a DataFrame with this column, and append it to the list for later concatenation.
                sv_frequency_absolute_difference_df = (self.Sv.Sv[i[0][0]].to_dataframe(name=None, dim_order=None)["Sv"] - self.Sv.Sv[i[1][0]].to_dataframe(name=None, dim_order=None)["Sv"]).abs().values
                #index_name is a string like "abs(Sv(38kHz)-Sv(120kHz))" It is a label for the column in the pre-clustering dataframe.
                index_name = "abs(Sv("+str(self.Sv.Sv[i[0][0]].coords.get("channel")).split("kHz")[0].split("GPT")[1].strip()+"kHz" +")-Sv("+ str(self.Sv.Sv[i[1][0]].coords.get("channel")).split("kHz")[0].split("GPT")[1].strip()+"kHz))"
                x = pd.DataFrame(data = sv_frequency_absolute_difference_df, columns = [index_name])
                sv_frequency_absolute_difference_map_list.append(x[index_name])
        
            print(sv_frequency_absolute_difference_map_list)
            pre_clustering_df = pd.concat(sv_frequency_absolute_difference_map_list, axis = 1) # Version 1 
            
        return pre_clustering_df.reset_index().select_dtypes(include=['float64']) # Returns pre-clustering dataframe.


    def assign_sv_clusters(self):
        """
        Performs KMeans clustering on the pre-processed Sv data and assigns cluster labels to the Sv xarray object.

        This method:
        - Constructs the pre-clustering DataFrame (features for clustering).
        - Normalizes the features (zero mean, unit variance).
        - Runs KMeans clustering.
        - Maps the resulting cluster labels back to the original Sv data structure.
        - Stores the cluster map as a new DataArray in self.Sv.

        Example:
            >>> kmeans_op = KMeansOperator(Sv, frequency_list=["38 kHz", "120 kHz"], k=3)
            >>> kmeans_op.assign_sv_clusters()
            # Now kmeans_op.Sv contains a new DataArray with cluster assignments.

        Returns:
            None. Modifies self.Sv in-place by adding a new DataArray with cluster assignments.
        """
        # Step 1: Construct the dataframe to be fed into KMeans clustering.
        self.pre_clustering_df = self.construct_pre_clustering_df()  # Each column is a frequency or feature.
        logger.info(f"{self.precluster_model} Preclustering DataFrame:")
        print(self.pre_clustering_df)

        # Step 2: Normalize the dataframe so each feature has mean 0 and variance 1.
        logger.info("Normalizing the preclustering dataframe.")
        df_normalized = preprocessing.scale(self.pre_clustering_df)
        logger.info("Normalized dataframe:")
        print(df_normalized)

        # Step 3: Convert normalized data to DataFrame for compatibility.
        self.df_clustered = pd.DataFrame(df_normalized)

        # Step 4: Run KMeans clustering.
        logger.info("Calculating KMeans clustering.")
        kmeans = KMeans(
            n_clusters=self.k,
            random_state=self.random_state,
            init='k-means++',
            n_init=self.n_init,
            max_iter=self.max_iter
        )
        X = self.df_clustered.values  # Feature matrix for sklearn
        clustered_records = kmeans.fit_predict(X)  # Cluster labels for each record

        # Step 5: Map cluster labels back to the original Sv DataFrame.
        # Use the first channel's DataFrame to get the correct index structure.
        self.Sv_df = self.Sv.Sv[0].to_dataframe(name=None, dim_order=None)
        # Add cluster labels as a new column (add 1 to avoid zero-based cluster numbers).
        self.Sv_df[self.frequency_set_string] = clustered_records + 1

        # Step 6: Convert the DataFrame with cluster labels back to xarray.
        self.clustering_data = self.Sv_df.to_xarray()

        # Step 7: Prepare the cluster map for all channels (to match Sv dimensionality).
        km_cluster_maps = []
        for i in range(len(self.Sv.Sv)):
            # Repeat the cluster assignments for each channel to match dimensions.
            km_cluster_maps.append(self.clustering_data[self.frequency_set_string].values)

        # Step 8: Add the cluster map as a new DataArray to self.Sv.
        self.Sv["km_cluster_map" + self.frequency_set_string] = xr.DataArray(
            data=km_cluster_maps,
            dims=['channel', 'ping_time', 'range_sample'],
            attrs=dict(
                description="The kmeans cluster group number.",
                units="Unitless",
                clusters=self.k,
                km_frequencies=self.frequency_set_string,
                random_state=self.random_state,
            )
        )

    

class KMClusterMap:
    """_summary_
    """
    def __init__(self, config_path):
        
        self.config_path = config_path # Path to the configuration file.
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        
        # Assign all config values to attributes
        for key, value in config.items():
            setattr(self, key, value)
        
            
        self.frequency_list_string = self.construct_frequency_list_string()
        self.construct_kmeans_clustergram() # This is the function which constructs the kmeans clustergram. It is called at the end of the run() function.
        
        
            
    def construct_kmeans_clustergram(self):
        """Once this class object is instantiated, a run function is employed to perform some required initialization sequences so that all class object variables can be constructed. 
        Another way to think about it is that run() handles the top level abstraction of KMeanClusterMap() managment and Sv prep for KMeans clustering analysis. 
        This involves converting, cropping, dropping Nans, computing MVBSusing physical units or sample number, 
        removing noise and applying .EVL or .EVR files.
        """        
        
        if (self.input_path is None or self.input_path.split(".")[-1].lower() not in {"nc", "raw", "yaml", "yml"}):
            
            logger.error("Provide valid .raw, .nc, or .yaml/.yml file")
                                                      
            
        else:
            
            
            if not os.path.exists(self.save_path): # If directory does not exist.
                
                os.makedirs(self.save_path)
            

            self.Sv = self.compute_Sv()    

            self.Sv = self.RROI(self.Sv, self.ping_time_begin, self.ping_time_end, self.range_sample_begin, self.range_sample_end )

            self.Sv = self.drop_nans(self.Sv)

            if self.remove_noise == True:

                self.Sv = self.noise_reduction(self.Sv)

            self.Sv = self.configure_MVBS(self.Sv)                   # Process routine. This deals with MVBS parametrization.

            #if self.line_files != None:
                
                #self.Sv = self.__process_line_files(self.line_files)
                
            if self.region_files != None:
                
                self.Sv = self.__process_region_files(self.region_files)

            logger.info('Preparing KMeans clustering operation') # Logging message.
            self.kmeans_operation = KMeansOperator( Sv = self.Sv,  frequency_list = self.frequency_list, k = self.n_clusters, random_state = self.random_state, n_init = self.n_init, max_iter = self.max_iter, precluster_model = self.precluster_model) # Create a KMeansOperator object which will handle the clustering operation.)  
            self.frequency_map = self.kmeans_operation.frequency_list # Makes a copy of the constructed frequency list data for this class since we need it for plotting.
            logger.info('Frequency map :') # Logging message.
            print(self.frequency_map) # Print out the frequency map.
            
            if self.save_path != None: # If a save path was provided.     
                
                logger.info('Saving cluster map and corresponding echograms...')    # Logging message.
                self.full_save(self.kmeans_operation.Sv) # This saves the kmeans cluster map and a corresponding echogram for each involved frequency.
            
            if self.plot_echograms == True:
                
                logger.info('Plotting cluster map and corresponding echograms...')    # Logging message.    
                plt.show()
                
            if self.plot_clustergrams == True:
                
                logger.info('Plotting cluster map and corresponding clustergrams...')    # Logging message.    
                plt.show()
            
            
    def compute_Sv(self):
        
        # Check if the input file is a YAML file (either .yaml or .yml extension)
        if self.input_path.split(".")[-1] == "yaml" or self.input_path.split(".")[-1] == "yml":
            # Log that we are opening the YAML file for debugging purposes
            logger.debug('Opening ' + self.input_path) # Logging message.

            # Open the YAML file for reading
            with open(self.config_path, 'r') as yaml_file:
                # Parse the YAML file into a Python dictionary
                yaml_data = yaml.safe_load(yaml_file)

                # Try to extract 'raw_path' and 'nc_path' from the YAML dictionary
                raw_path = yaml_data.get('raw_path')
                nc_path = yaml_data.get('nc_path')

                # Now, depending on which path is provided, process accordingly:
                # If a raw file path is provided in the YAML
                if raw_path:
                    # Open the raw file using echopype, specifying the sonar model
                    ed = ep.open_raw(raw_file=raw_path, sonar_model=self.sonar_model)
                    # Convert the raw file to NetCDF and save it to the specified directory
                    ed.to_netcdf(save_path=self.save_path)
                    # Compute Sv (volume backscattering strength), drop NaNs, and crop to the specified region
                    self.Sv = ep.calibrate.compute_Sv(ed).dropna(dim="range_sample", how="any").isel(
                    range_sample=slice(self.range_sample_begin, self.range_sample_end),
                    ping_time=slice(self.ping_time_begin, self.ping_time_end)
                    )
                    logger.info('Sv computed from raw file.') # Logging message.
                # If a NetCDF file path is provided in the YAML
                elif nc_path:
                    # Open the already converted NetCDF file using echopype
                    ed = ep.open_converted(nc_path)
                    # Compute Sv, drop NaNs, and crop to the specified region
                    self.Sv = ep.calibrate.compute_Sv(ed).dropna(dim="range_sample", how="any").isel(
                    range_sample=slice(self.range_sample_begin, self.range_sample_end),
                    ping_time=slice(self.ping_time_begin, self.ping_time_end)
                    )
                else:
                    # If neither path is provided, print an error message
                    print("YAML file must contain either 'raw_file' or 'nc_file' key.")
                    
        # Check if the input file is a .raw file
        if self.input_path.split(".")[-1] == "raw":
            
            raw_file = self.input_path  # Assign the input path to raw_file variable

            # Open the raw file using echopype, specifying the sonar model as 'EK60'
            ed = ep.open_raw(raw_file=raw_file, sonar_model='EK60')

            # Convert the raw file to NetCDF format and save it to the specified directory
            ed.to_netcdf(save_path=self.save_path)

            # Compute Sv (volume backscattering strength) from the echodata object,
            # drop any NaN values along the 'range_sample' dimension,
            # and crop the data to the specified range_sample and ping_time slices
            self.Sv = ep.calibrate.compute_Sv(ed).dropna(
            dim="range_sample", how="any"
            ).isel(
            range_sample=slice(self.range_sample_begin, self.range_sample_end),
            ping_time=slice(self.ping_time_begin, self.ping_time_end)
            )

        # Check if the input file is a .nc (NetCDF) file
        if self.input_path.split(".")[-1] == "nc":
            # Assign the input path to nc_file variable
            nc_file = self.input_path

            # Open the NetCDF file using echopype's open_converted function to get the echodata object
            ed = ep.open_converted(nc_file)

            # Compute Sv (volume backscattering strength) from the echodata object,
            # drop any NaN values along the 'range_sample' dimension,
            # and crop the data to the specified range_sample and ping_time slices
            self.Sv = ep.calibrate.compute_Sv(ed).dropna(
            dim="range_sample", how="any"
            ).isel(
            range_sample=slice(self.range_sample_begin, self.range_sample_end),
            ping_time=slice(self.ping_time_begin, self.ping_time_end)
            )
        
        if self.Sv is None:
            logger.error("Sv is None. Provide a .yaml configuration, .raw, or .nc file.")
            
        else:
            logger.info('Sv computed successfully.')
            return self.Sv # Return Sv xarray object.
            

    def process_region_files(self):
        """
        Processes region files (EVR) to mask Sv data according to specified regions.

        This method:
        - Reads region files (EVR) using echoregions.
        - Creates a depth coordinate for the Sv dataset.
        - Swaps the 'range_sample' dimension with 'depth' for easier interpretation.
        - Applies the region mask to the Sv data.
        - Plots the mask and masked Sv for visual inspection.

        Returns:
            None. Modifies self.Sv in-place by applying the region mask.
        """
        if self.region_files is not None:
            logger.info("Processing region files for masking Sv data.")

            Sv = self.Sv

            # --- Create a depth coordinate for the Sv dataset ---
            # The echo_range variable gives the distance from the transducer to each sample.
            # We assume water_level is constant across frequencies and times for simplicity.
            # Select the first channel and first ping_time for echo_range and water_level.
            echo_range = Sv.echo_range.isel(channel=0, ping_time=0)
            water_level = Sv.water_level.isel(channel=0, time3=0)
            # Compute the absolute depth by adding water_level to echo_range.
            depth = water_level + echo_range
            # Remove the 'channel' coordinate from depth to avoid conflicts.
            depth = depth.drop_vars('channel')
            # Add the computed depth as a new coordinate to the Sv dataset.
            Sv['depth'] = depth
            # Swap the 'range_sample' dimension for 'depth' for easier interpretation and plotting.
            Sv = Sv.swap_dims({'range_sample': 'depth'})

            logger.info("Reading EVR region file(s) and applying mask.")

            # --- Read and apply the region mask from EVR file ---
            # For now, only the first region file in the list is processed.
            EVR_FILE = self.region_files[0]  # TODO: Extend to process all region files in the list
            # Read the EVR file using echoregions, which parses region definitions.
            r2d = er.read_evr(EVR_FILE)
            logger.info(f"Region IDs found in EVR file: {r2d.data.region_id.values}")

            # --- Create a mask for a specific region ID (e.g., region ID 1) ---
            # The mask will be True (or 1) inside the region and NaN outside.
            # Select the first channel and drop the 'channel' coordinate for compatibility.
            mask = r2d.mask(Sv.isel(channel=0).drop('channel'), [1], mask_var="ROI")
            logger.info(f"Mask max value: {mask.max().values}")

            # --- Plot the mask for visual inspection ---
            plt.figure()
            mask.plot()
            plt.title("Region Mask")

            # --- Apply the mask to Sv ---
            # Use xarray's where: values outside the mask become NaN.
            Sv_masked = Sv.where(mask.isnull())

            # --- Plot the masked Sv for the first channel ---
            # Transpose for correct orientation and set yincrease=False to match echogram convention.
            plt.figure()
            Sv_masked.isel(channel=0).T.plot(yincrease=False)
            plt.title("Masked Sv (Channel 0)")

            logger.info("Region masking and plotting complete.")
            
    def RROI(self, Sv, ping_time_begin, ping_time_end, range_sample_begin, range_sample_end ):
        """
        Crops a rectangular region of interest (RROI) from the Sv xarray Dataset.

        This function selects a subset of the Sv data based on the specified ping_time and range_sample indices.
        It is useful for focusing analysis on a specific region of the echogram, reducing computational load,
        or excluding irrelevant data.

        Args:
            Sv (xarray.Dataset): The input Sv dataset (typically with dimensions: channel, ping_time, range_sample).
            ping_time_begin (int): The starting index for the ping_time dimension (inclusive).
            ping_time_end (int): The ending index for the ping_time dimension (exclusive).
            range_sample_begin (int): The starting index for the range_sample dimension (inclusive).
            range_sample_end (int): The ending index for the range_sample dimension (exclusive).

        Returns:
            xarray.Dataset: The cropped Sv dataset containing only the specified region.

        Example:
            >>> # Suppose Sv has shape (channel=3, ping_time=1000, range_sample=500)
            >>> km = KMClusterMap(...)
            >>> Sv_cropped = km.RROI(Sv, ping_time_begin=100, ping_time_end=200, range_sample_begin=50, range_sample_end=150)
            >>> print(Sv_cropped.dims)
            # Output: {'channel': 3, 'ping_time': 100, 'range_sample': 100}

        Explanation:
            - Uses xarray's isel method to slice the ping_time and range_sample dimensions.
            - The indices are zero-based and follow Python's slice semantics (start inclusive, end exclusive).
            - This operation does not modify the original Sv; it returns a new cropped Dataset.

        Notes:
            - If ping_time_begin or range_sample_begin is None, slicing will start from the beginning.
            - If ping_time_end or range_sample_end is None, slicing will go to the end.
            - Make sure the indices are within the bounds of the Sv dimensions to avoid IndexError.
        """
        logger.info("Rectangular region of interest is cropped out of analysis region.")
        Sv_RROI = Sv.isel(
            range_sample=slice(range_sample_begin, range_sample_end),
            ping_time=slice(ping_time_begin, ping_time_end)
        )
        logger.info("Region of Interests Cropped!")
        return Sv_RROI
    
    def drop_nans(self, Sv):
        """
        Drops NaN values along the 'range_sample' dimension of the Sv xarray Dataset.

        This function removes any samples (along the 'range_sample' dimension) that contain NaN values,
        resulting in a cleaner dataset for further analysis or clustering.

        Args:
            Sv (xarray.Dataset): The input Sv dataset.

        Returns:
            xarray.Dataset: The Sv dataset with NaNs dropped along 'range_sample'.

        Example:
            >>> Sv_clean = self.drop_nans(Sv)
        """
        logger.info("Dropping NaN values from Sv along 'range_sample' dimension.")
        Sv_naNs_dropped = Sv.dropna(dim="range_sample", how="any")
        logger.info(f"NaN values dropped. Resulting shape: {Sv_naNs_dropped.sizes}")
        return Sv_naNs_dropped
    
    def noise_reduction(self, Sv):
                    
        logger.info('Removing noise from Sv...')                            # Logging message.
        logger.info('   range_sample = ' + str(self.range_sample_num))      # Logging message.
        logger.info('   ping_num = ' + str(self.ping_num))                  # Logging message.
        logger.info("Noise Removed!")
        # Use echopype's clean module to remove noise from Sv.

        return ep.clean.remove_background_noise(Sv, range_sample_num=self.range_sample_num, ping_num=self.ping_num)  
      
    def configure_MVBS(self, Sv):
        """Configure MVBS using provided class variables. This internal method should not be directly employed by user.
        """
        if self.range_meter_bin != None and self.ping_time_bin != None:
            logger.info('Calculating MVBS using reduction by physical units.')  # Logging message.
            logger.info('   range_meter_bin = ' + str(self.range_meter_bin))    # Logging message.
            logger.info('   ping_time_bin = ' + str(self.ping_time_bin))        # Logging message.
            self.Sv = ep.commongrid.compute_MVBS(Sv, range_bin = self.range_bin, ping_time_bin = self.ping_time_bin )
                
        if self.ping_num != None and self.range_sample_num != None:
            logger.info('Calculating MVBS using reduction by sample number.')
            logger.info('   range_sample_num = ' + str(self.range_sample_num))
            logger.info('   ping_num = ' + str(self.ping_num))
            self.Sv = ep.commongrid.compute_MVBS_index_binning( Sv, range_sample_num=self.range_sample_num, ping_num=self.ping_num )
                
        return Sv      
        
    def preprocess(self):
        """
        Preprocesses Sv data: applies noise removal and computes MVBS.
        Uses updated echopype.clean API.
        """
        Sv_input = self.Sv

        # Step 1: Noise and artifact removal
        if self.remove_noise:
            # Estimate and remove background noise
            noise_est = ep.clean.estimate_background_noise(
                Sv_input,
                range_sample_num=self.range_sample_num,
                ping_num=self.ping_num
            )
            Sv_denoised = ep.clean.remove_background_noise(Sv_input, self.ping_num, self.range_sample_num)

            # Apply additional masking (optional, can be customized)
            #Sv_masked = ep.clean.mask_impulse_noise(Sv_denoised)
            #Sv_masked = ep.clean.mask_transient_noise(Sv_masked)
            #Sv_input = Sv_masked
            self.Sv_clean = Sv_denoised

        # Step 2: MVBS computation (physical units)
        if self.range_meter_bin is not None and self.ping_time_bin is not None:
            self.MVBS_physical_unit_type_reduction = ep.commongrid.compute_MVBS(
                Sv_input
            )

        # Step 3: MVBS computation (index binning)
        if self.range_sample_num is not None and self.ping_num is not None:
            self.MVBS_sample_number_type_reduction = ep.commongrid.compute_MVBS_index_binning(
                Sv_input,
                range_sample_num=self.range_sample_num,
                ping_num=self.ping_num
            )

                    
            
    
    def plot_clustergram(self, Sv):
        """
        Plots the KMeans clustergram for the clustered Sv data.

        Args:
            Sv (xarray.Dataset): The dataset containing the KMeans cluster map.

        This function uses matplotlib and xarray's plotting to visualize the cluster assignments
        produced by KMeans clustering. The clustergram is shown using the specified colormap and
        number of clusters. The y-axis is inverted to match echogram conventions.
        """
        # Get the colormap with the number of clusters
        cmap = plt.get_cmap(self.color_map, self.n_clusters)
        # Plot the first channel's cluster map, transposed for correct orientation
        Sv["km_cluster_map"+self.kmeans_operation.frequency_set_string][0].transpose("range_sample","ping_time").plot(cmap=cmap)
        # Optionally, you could plot the original Sv for comparison (commented out)
        # self.Sv["Sv"][0].transpose("range_sample","ping_time").plot()

        # Set the plot title with relevant clustering and file information
        plt.title(
            self.kmeans_operation.frequency_set_string +
            ",    n_clusters = " + str(self.n_clusters) +
            ",    random_state = " + str(self.random_state) +
            ",    file = " + self.input_path +
            ",    colormap = " + self.color_map
        )
        # Invert the y-axis for echogram-style display
        plt.gca().invert_yaxis()
        # Show the plot
        plt.show()
        plt.clf()  # clear current figure
        plt.cla()  # clear current axes

        
    def save_echogram(self, data_array, channel):
        
        frequency =self.get_frequency(channel)
         # Get the channel index from the frequency string.
        #cmap = plt.get_cmap(self.color_map, self.n_clusters)
        logger.info("Saving echogram for frequency: " + frequency + ", channel: " + str(channel))      
        # Transpose and plot using hvplot
        plot = data_array[int(channel)].transpose("range_sample", "ping_time").hvplot(
            x="ping_time",
            y="range_sample",
            cmap=self.color_map,
            title=f"frequency = {frequency},    file = {self.input_path},    colormap = {self.color_map}",
            invert_yaxis=True,
            aspect='auto'
        )

        # Save as static image or HTML
        #title = f"{self.asset_path}/eg:{self.name}<{frequency}>.html"
        
        
        #hv.save(plot, title, backend='bokeh')




    def save_clustergram(self, Sv):

        #cmap = plt.get_cmap(self.color_map, self.n_clusters) 
        Sv["km_cluster_map"+self.kmeans_operation.frequency_set_string][0].transpose("range_sample","ping_time").plot()
        plt.title(self.kmeans_operation.frequency_set_string+",    n_clusters = "+str(self.n_clusters)+",    random_state = "+str(self.random_state)+",    file = "+self.input_path+",    colormap = "+self.color_map)
        plt.gca().invert_yaxis()
        plt.savefig(self.asset_path+"/km:"+self.name+"<"+ self.frequency_list_string+"k="+str(self.n_clusters)+"_rs="+str(self.random_state)+"_cm="+self.color_map+"_md="+str(self.precluster_model)+"_rmb="+str(self.range_meter_bin)+">", dpi=2048)
        plt.clf()  # clear current figure
        plt.cla()  # clear current axes


    def full_save(self, Sv):
        self.save_clustergram(Sv)
        for frequency in self.frequency_map:
            self.save_echogram(self.Sv["Sv"],frequency[0])

    def construct_frequency_list_string(self):
        """
        Constructs a string representation of the frequency_list for labeling or file naming.

        Returns:
            str: A string with all frequency values concatenated and separated by underscores.

        Example:
            >>> self.frequency_list = [[0, "38 kHz"], [1, "120 kHz"]]
            >>> self.construct_frequency_list_string()
            '38 kHz_120 kHz_'
        """
        # Initialize an empty string to accumulate frequency names
        frequency_list_string = ""
        # Iterate over the frequency_list, which is expected to be a list of [channel_index, frequency_string]
        for frequency in self.frequency_list:
            # Append the frequency string and an underscore to the result
            frequency_list_string = frequency_list_string + frequency[1] + "_"
        # Return the concatenated string of frequencies, separated by underscores
        return frequency_list_string

    def get_frequency(self, channel):
        """
        Given a channel index, return the corresponding frequency string from the frequency_list.

        Args:
            channel (int): The channel index to look up.

        Returns:
            str or None: The frequency string (e.g., "38 kHz") if found, otherwise None.

        Example:
            >>> km = KMClusterMap(...)
            >>> km.frequency_list = [[0, "38 kHz"], [1, "120 kHz"]]
            >>> km.get_frequency(0)
            '38 kHz'
        """
        for frequency in self.frequency_map:
            # frequency is expected to be [channel_index, frequency_string]
            if str(frequency[0]) == str(channel):
                return frequency[1]
        
            
    def get_channel(self, frequency):
        """
        Given a frequency string, return the corresponding channel index from the frequency_list.

        Args:
            frequency (str): The frequency string to look up (e.g., "38 kHz").

        Returns:
            int or None: The channel index if found, otherwise None.

        Example:
            >>> km = KMClusterMap(...)
            >>> km.frequency_list = [[0, "38 kHz"], [1, "120 kHz"]]
            >>> km.get_channel("38 kHz")
            0
        """
        for frequency in self.frequency_list:
            # frequency is expected to be [channel_index, frequency_string]
            if frequency[1] == frequency:
                return frequency[0]
        return None  # Return None if frequency not found

