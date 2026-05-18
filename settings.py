# Settings for running this Notebook

INTERNAL_SETTINGS = {}

INTERNAL_SETTINGS["files"] = {}
INTERNAL_SETTINGS["files"]["dem_folder"] = "input_geotiffs" #Here the to-be-used-geotiffs are located

INTERNAL_SETTINGS["fft"] = {}
INTERNAL_SETTINGS["fft"]["tile_size_km"] = 9 # 9 average length and width of a tile that is processed individually
INTERNAL_SETTINGS["fft"]["tile_size_px"] = 21 # 20

INTERNAL_SETTINGS["fft"]["sealevel_threshold"] = 0.75 # 20

INTERNAL_SETTINGS["fft"]["tile_overlap_percent"] = 95 # 95 How much (by a tile length in percent) do the tiles overlap
INTERNAL_SETTINGS["fft"]["fft_levels"] = 7  # 14

INTERNAL_SETTINGS["output"] = {}
INTERNAL_SETTINGS["output"]["folder_name"] = "output/label_images" 
INTERNAL_SETTINGS["output"]["q_factor"] = 5 # 5 Bigger Q-Factor, smaller image result, faster computation
INTERNAL_SETTINGS["output"]["label_mode_filter"] = 4  # 20 this is in arbitrary units. It will adapt to the final size, so bigger images are filtered in the same way
INTERNAL_SETTINGS["output"]["label_count"] = 10 # 10 as for now

# Speed settings to quickly test things. Just set to True
if True:
    INTERNAL_SETTINGS["fft"]["tile_size_km"] = 12
    INTERNAL_SETTINGS["fft"]["tile_overlap_percent"] = 90 # 85 How much (by a tile length in percent) do the tiles overlap

# This one calculates the actual filter size in pixels regarding the final image
INTERNAL_SETTINGS["output"]["label_mode_filter_radius"] = int(INTERNAL_SETTINGS["output"]["label_mode_filter"] / (INTERNAL_SETTINGS["output"]["q_factor"] / 5))
INTERNAL_SETTINGS["fft"]["tile_overlap_multi"] = 1 / (1 - (INTERNAL_SETTINGS["fft"]["tile_overlap_percent"]/100))
