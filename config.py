# config.py

#I dati si riferiscono all'anno
cities_data = {
    "terlizzi_italy": {
        "num_accidents": 200, #100
        "center_lat": 41.12, "center_lon": 16.53, "lat_std": 0.005, "lon_std": 0.003, #####"lon_std": 0.007,
        "num_traffic_sensors": 100,            # Numero di sensori di traffico fittizi 100
        "num_readings_per_sensor": 75,         # Numero di letture per ciascun sensore (nell'arco di 1 anno) 75
        "buffer_distance_meters": 100
        #num_traffic_sensors era 10
        #num_readings_per_sensor": 50

        #"avg_population_density": 500 # Valore di esempio per città piccola
    },
    "molfetta_italy": {
        "num_accidents": 300,
        "center_lat": 41.20, "center_lon": 16.60, "lat_std": 0.006, "lon_std": 0.004, #"lat_std": 0.004, "lon_std": 0.002,
        "num_traffic_sensors": 150,
        "num_readings_per_sensor": 100,
        "buffer_distance_meters": 150
    },
    "bari_italy": {
        "num_accidents": 700,
        "center_lat": 41.12, "center_lon": 16.87, "lat_std": 0.02, "lon_std": 0.025,
        "num_traffic_sensors": 400, "num_readings_per_sensor": 150,
        "buffer_distance_meters": 200
        #"avg_population_density": 2500 # Valore di esempio per città media/grande
    }
}