# Einstellungen für dieses Notebook

INTERNAL_SETTINGS = {}

INTERNAL_SETTINGS["files"] = {}
INTERNAL_SETTINGS["files"]["dem_folder"] = "input_geotiffs" # Hier befinden sich die zu verwendenden GeoTIFFs

INTERNAL_SETTINGS["fft"] = {}
INTERNAL_SETTINGS["fft"]["tile_size_km"] = 12 # 12 durchschnittliche Länge und Breite einer einzeln verarbeiteten Kachel
INTERNAL_SETTINGS["fft"]["tile_size_px"] = 33 # 23

INTERNAL_SETTINGS["fft"]["sealevel_threshold"] = 0.75 # 20

INTERNAL_SETTINGS["fft"]["tile_overlap_percent"] = 85 # 85 Überlappung der Kacheln in Prozent der Kachellänge
INTERNAL_SETTINGS["fft"]["fft_levels"] = 17  # 17

INTERNAL_SETTINGS["output"] = {}
INTERNAL_SETTINGS["output"]["folder_name"] = "output/label_images" 
INTERNAL_SETTINGS["output"]["label_mode_filter_radius_km"] =  15 # 15

INTERNAL_SETTINGS["output"]["label_count"] = 10 # 10 aktueller Wert


# Berechnet die tatsächliche Filtergröße in Pixeln bezogen auf das finale Bild
INTERNAL_SETTINGS["fft"]["tile_overlap_multi"] = 1 / (1 - (INTERNAL_SETTINGS["fft"]["tile_overlap_percent"]/100))
