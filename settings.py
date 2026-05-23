# Settings for running this Notebook

INTERNAL_SETTINGS = {}

INTERNAL_SETTINGS["files"] = {}
INTERNAL_SETTINGS["files"]["dem_folder"] = "input_geotiffs" #Here the to-be-used-geotiffs are located

INTERNAL_SETTINGS["fft"] = {}
INTERNAL_SETTINGS["fft"]["tile_size_km"] = 12 # 12 average length and width of a tile that is processed individually
INTERNAL_SETTINGS["fft"]["tile_size_px"] = 23 # 23

INTERNAL_SETTINGS["fft"]["sealevel_threshold"] = 0.75 # 20

INTERNAL_SETTINGS["fft"]["tile_overlap_percent"] = 85 # 85 How much (by a tile length in percent) do the tiles overlap
INTERNAL_SETTINGS["fft"]["fft_levels"] = 17  # 17

INTERNAL_SETTINGS["output"] = {}
INTERNAL_SETTINGS["output"]["folder_name"] = "output/label_images" 
INTERNAL_SETTINGS["output"]["label_mode_filter_radius_km"] =  15

INTERNAL_SETTINGS["output"]["label_count"] = 10 # 10 as for now


# This one calculates the actual filter size in pixels regarding the final image
INTERNAL_SETTINGS["fft"]["tile_overlap_multi"] = 1 / (1 - (INTERNAL_SETTINGS["fft"]["tile_overlap_percent"]/100))
