# main.py
import os
import osmnx as ox
import folium
import folium.plugins
import pandas as pd
import geopandas as gpd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import (
    load_road_network,
    load_pois,
    load_accidents,
    load_traffic_data
)
from src.generate_fake_data import generate_fake_accidents_data, generate_fake_traffic_data
from src.preprocessing import (
    preprocess_accidents,
    get_node_edge_features_from_osm,
    preprocess_traffic_data
)
from src.data_integration import (
    aggregate_traffic_to_osm_elements,
    aggregate_features_by_node,
    calculate_incident_severity_features,
)
from src.optimization import score_candidate_locations, recommend_interventions
from src.visualization import plot_osm_data_on_map, plot_recommendations_on_map
from config import cities_data

# Importa metriche per i report
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score, accuracy_score, confusion_matrix
)

# Funzione ausiliaria per la gestione del pathfile
def get_city_file_path(city_name, file_type):
    """Genera un percorso file standardizzato per una data città e tipo di file."""
    safe_city_name = city_name.replace(', ', '_').replace(' ', '_').replace('/', '_').lower()
    if file_type == 'graphml':
        return os.path.join("data", f"{safe_city_name}_road_network.graphml")
    elif file_type == 'geojson_pois':
        return os.path.join("data", f"{safe_city_name}_pois.geojson")
    return None

def main():
    print("--- Inizio del Progetto di Analisi del Traffico Urbano ---")

    # --- Configurazione della Città ---
    city_name = "Terlizzi, Italy"
    #city_name = "Molfetta, Italy"
    #city_name = "Bari, Italy"

    #BUFFER_DISTANCE_METERS = 100
    MODEL_FILENAME = "random_forest_model.joblib"
    TARGET_COLUMN_BINARY = 'ha_incidente_vicino'

    for folder in ['data', 'reports']:
        if not os.path.exists(folder):
            os.makedirs(folder)

    ## --- Acquisizione e Caricamento Dati OSM ---
    print(f"\nFase 0: Acquisizione/Caricamento Dati OpenStreetMap per '{city_name}'...")

    graphml_filepath = get_city_file_path(city_name, 'graphml')
    geojson_filepath = get_city_file_path(city_name, 'geojson_pois')

    G = None
    pois_gdf = None

    # Logica per scaricare o caricare la rete stradale
    if not os.path.exists(graphml_filepath):
        print(f"File della rete stradale per '{city_name}' non trovato. Scaricamento in corso...")
        try:
            G = ox.graph_from_place(city_name, network_type="drive", simplify=True, retain_all=False) #Più pesante se mettiamo "all"
            nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
            ox.save_graphml(G, filepath=graphml_filepath)
            print(f"Rete stradale per '{city_name}' salvata in {graphml_filepath}")
        except Exception as e:
            print(f"Errore durante lo scaricamento della rete stradale per '{city_name}': {e}")
            print("Assicurati che il nome della città sia valido e che ci sia una connessione internet.")
            return
    else:
        print(f"Caricamento rete stradale esistente da {graphml_filepath}")
        G = load_road_network(filepath=graphml_filepath)

    if G is not None:
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    else:
        print("Errore: Impossibile ottenere il grafo G.")
        return

    # Logica per scaricare o caricare i Punti di Interesse
    if not os.path.exists(geojson_filepath):
        print(f"File POI per '{city_name}' non trovato. Scaricamento in corso...")
        try:
            tags = {
                'highway': ['traffic_signals', 'crossing'],
                'amenity': ['school', 'hospital', 'kindergarten'],
                'leisure': ['park', 'playground'],
                'shop': True, # Cattura tutti i tipi di shop
                'building': 'public' # Cattura edifici pubblici
            }
            pois_gdf = ox.features_from_place(city_name, tags=tags)

            if not pois_gdf.empty:
                # Converti geometrie multiple a punti se necessario, filtra per Point
                pois_gdf['geometry'] = pois_gdf['geometry'].apply(lambda geom: geom.geoms[0] if geom.geom_type == 'MultiPoint' else geom)
                pois_gdf = pois_gdf[pois_gdf.geometry.geom_type == 'Point']
                pois_gdf.to_file(geojson_filepath, driver="GeoJSON")
                print(f"POI per '{city_name}' salvati in {geojson_filepath}")
            else:
                print(f"Nessun POI rilevante trovato per '{city_name}'. Il file {geojson_filepath} non sarà creato.")
                pois_gdf = gpd.GeoDataFrame() # Imposta a GeoDataFrame vuoto
        except Exception as e:
            print(f"Errore durante lo scaricamento dei POI per '{city_name}': {e}")
            pois_gdf = gpd.GeoDataFrame() # Imposta a GeoDataFrame vuoto in caso di errore
    else:
        print(f"Caricamento POI esistenti da {geojson_filepath}")
        pois_gdf = load_pois(filepath=geojson_filepath)
        if pois_gdf is None or pois_gdf.empty:
            print("Il GeoDataFrame dei POI è vuoto o non è stato caricato correttamente. Verrà considerato vuoto.")
            pois_gdf = gpd.GeoDataFrame()

    # --- Visualizzazione Dati OSM Scaricati ---
    print("\nVisualizzazione dei dati OSM scaricati...")
    map_osm_filepath = os.path.join("reports", f"{city_name.replace(', ', '_').replace(' ', '_').lower()}_osm_data_map.html")
    plot_osm_data_on_map(G, pois_gdf, filepath=map_osm_filepath)

    city_name_lower = city_name.replace(', ', '_').replace(' ', '_').replace('/', '_').lower()

    #######
    # Calcola il centro dinamico (centroid) dei POI per allineare traffico e incidenti
    if not pois_gdf.empty:
        # È buona pratica assicurarsi che il GeoDataFrame sia in un CRS geografico (lat/lon)
        # prima di calcolare i centroidi per le coordinate. OSMnx di solito lo fa già (EPSG:4326).
        if pois_gdf.crs and pois_gdf.crs.is_projected:
            pois_gdf_geographic = pois_gdf.to_crs(epsg=4326)
        else:
            pois_gdf_geographic = pois_gdf

        # Calcola il centroid di tutte le geometrie dei POI combinate
        # .unary_union crea una singola geometria da tutte le geometrie nel GeoDataFrame.
        city_center_from_pois_lat = pois_gdf_geographic.geometry.union_all().centroid.y
        city_center_from_pois_lon = pois_gdf_geographic.geometry.union_all().centroid.x
        print(f"Calcolato centro della città dai POI: Lat={city_center_from_pois_lat:.4f}, Lon={city_center_from_pois_lon:.4f}")
    else:
        # Fallback: se per qualche motivo non ci sono POI, usa il centro predefinito dalla configurazione
        city_center_from_pois_lat = cities_data[city_name_lower]["center_lat"]
        city_center_from_pois_lon = cities_data[city_name_lower]["center_lon"]
        print("Nessun POI trovato per calcolare il centro, usando il centro predefinito da CITIES_DATA.")
    #######

    # --- Dati Incidenti fittizi o reali ---
    #df_accidents = load_accidents("data/incidenti_stradali.xlsx")
    accidents_filepath = f"data/incidenti_stradali_{city_name_lower}.xlsx"
    df_accidents = load_accidents(accidents_filepath)
    if df_accidents is None or df_accidents.empty:
        print(f"ATTENZIONE: Dati incidenti stradali da {accidents_filepath} non disponibili o vuoti. Verrà creato un dataset fittizio IN TEMPO REALE.")
        # Genera dati fittizi sul momento, usando la stessa logica di generate_fake_data.py ma solo per una città
        # passandogli direttamente le coordinate del centro città correnti.

        # Esempio:
        center_lat_city, center_lon_city = ox.geocode(city_name)
        #buffer_distance_for_risk = 100 # Distanza per i POI

        #df_accidents = generate_fake_accidents_data(

        nodes_gdf_full_area = ox.graph_to_gdfs(G, nodes=True, edges=False)
        # Assicura che sia nel CRS geografico corretto (WGS84)
        nodes_gdf_full_area = nodes_gdf_full_area.to_crs("EPSG:4326")

        nodes_gdf_full_area['Latitudine'] = nodes_gdf_full_area['y']
        nodes_gdf_full_area['Longitudine'] = nodes_gdf_full_area['x']

        fake_incidents_df_raw = generate_fake_accidents_data(
            nodes_gdf_full_area=nodes_gdf_full_area,
            pois_gdf=pois_gdf,
            buffer_distance=cities_data[city_name_lower]["buffer_distance_meters"],
            num_accidents=cities_data[city_name_lower]["num_accidents"],
            # Puoi specificare incident_generation_center_lat/lon e spread qui,
            # oppure lasciarli a None per usare il centro medio e lo spread di default.
            # Ad esempio, per forzare un centro specifico:
            # incident_generation_center_lat=41.1321,
            # incident_generation_center_lon=16.5461,
            # incident_generation_spread_lat=0.005, # Esempio: un po' meno disperso
            # incident_generation_spread_lon=0.005
        )
        ###fake_incidents_df_raw  = generate_fake_accidents_data(
        ###    nodes_gdf=nodes_gdf,
        ###    pois_gdf=pois_gdf,
        ###    center_lat=city_center_from_pois_lat,
        ###    center_lon=city_center_from_pois_lon,
        ###    lat_std=cities_data[city_name_lower]["lat_std"], #0.015, # std per lat
        ###    lon_std=cities_data[city_name_lower]["lon_std"],  #0.02 # std per lon
        ###    buffer_distance=cities_data[city_name_lower]["buffer_distance_meters"],
        ###    num_accidents=cities_data[city_name_lower]["num_accidents"]
        ###)

        #####nodes_gdf = nodes_gdf.to_crs("EPSG:4326") # Assicura CRS geografico
        #####nodes_gdf['Latitudine'] = nodes_gdf['y']
        #####nodes_gdf['Longitudine'] = nodes_gdf['x']
        #####city_center_lat_inc = nodes_gdf['Latitudine'].mean()
        #####city_center_lon_inc = nodes_gdf['Longitudine'].mean()
#####
        #####fake_incidents_df_raw = generate_fake_accidents_data(
        #####    nodes_gdf=nodes_gdf, # Ora questo è il GeoDataFrame completo dei nodi OSM
        #####    pois_gdf=pois_gdf,
        #####    buffer_distance=cities_data[city_name_lower]["buffer_distance_meters"],
        #####    num_accidents=cities_data[city_name_lower]["num_accidents"],
        #####    incident_generation_center_lat=city_center_lat_inc,
        #####    incident_generation_center_lon=city_center_lon_inc,
        #####    # Gli spread per la generazione degli incidenti (se vuoi raggrupparli)
        #####    # Questi sono DEVIANO dalla media dei nodi OSM, NON sono i tuoi vecchi lat_std/lon_std
        #####    incident_generation_spread_lat=0.01, # Esempio: ~1km di dispersione
        #####    incident_generation_spread_lon=0.01 # Esempio: ~1km di dispersione
        #####)

        # Adesso fake_incidents_df_raw è un pandas DataFrame con gli incidenti generati
        # Dovrai convertirlo in un GeoDataFrame prima di passarlo
        df_accidents = gpd.GeoDataFrame(
            fake_incidents_df_raw,
            geometry=gpd.points_from_xy(fake_incidents_df_raw.Longitudine, fake_incidents_df_raw.Latitudine),
            crs="EPSG:4326" # Il CRS finale degli incidenti è sempre WGS84
        )

        output_dir = "data"
        filepath_accidents = os.path.join(output_dir, f"incidenti_stradali_{city_name_lower}.xlsx")
        df_accidents.to_excel(filepath_accidents, index=False)
        print(f"Dataset incidenti fittizio generato in tempo reale e salvato: {filepath_accidents}.")
    else:
        print(f"Dati incidenti caricati da: {accidents_filepath}")

    # ... (le fasi 1, 2, 3, 4, 5, 6, 7 come definito nell'ultima risposta)
    # ...
    # --- Dati Traffico (Nuova Sezione) ---
    #print("\n--- Acquisizione e Pre-elaborazione Dati Traffico/Incidenti ---")
    # --- 2. Pre-processing e Feature Engineering ---

    #nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)

    traffic_filepath = f"data/traffico_{city_name_lower}.xlsx"
    df_traffic = load_traffic_data(traffic_filepath)

    # Tags POI per il traffico (potrebbe essere più ampio di quello degli incidenti)
    traffic_pois_tags = {
        "amenity": ["school", "kindergarten", "university", "college", "restaurant", "cafe", "bar", "fast_food", "hospital", "clinic"],
        "shop": True, # tutti i tipi di shop
        "office": True,
        "industrial": True,
        "public_transport": ["station", "bus_station", "tram_stop", "subway_entrance"],
        "leisure": ["stadium", "park"], # aree di attrazione
        "landuse": ["retail", "commercial"]
    }
    traff_pois_gdf = ox.features_from_place(city_name, traffic_pois_tags)

    if traff_pois_gdf is not None and not traff_pois_gdf.empty:
        traff_pois_gdf = traff_pois_gdf.dropna(subset=['geometry'])
        if 'original_poi_id' not in traff_pois_gdf.columns:
            if isinstance(traff_pois_gdf.index, pd.MultiIndex):
                traff_pois_gdf['original_poi_id'] = traff_pois_gdf.index.map(str) + '_poi'
            else:
                traff_pois_gdf['original_poi_id'] = traff_pois_gdf.index.astype(str) + '_poi'
    else:
        traff_pois_gdf = gpd.GeoDataFrame(columns=['geometry', 'original_poi_id'], crs="EPSG:4326")
        print("AVVISO: Nessun POI significativo scaricato per il traffico o traff_pois_gdf vuoto.")

    # Pesi dei tipi di strada per il traffico
    #road_weights = {
    #    'motorway': 5.0, 'trunk': 4.5, 'primary': 4.0, 'secondary': 3.0,
    #    'tertiary': 2.0, 'residential': 1.0, 'unclassified': 1.0, 'service': 0.5
    #}

    # Aggiungi un punto di controllo QUI per vedere cosa load_traffic_data ha prodotto
    #print("\n--- DEBUG STEP 1: df_traffic DOPO load_traffic_data ---")
    #if df_traffic is not None and not df_traffic.empty:
    #    print("Dtype di df_traffic (dopo load_traffic_data):\n", df_traffic[['ConteggioVeicoli', 'VelocitaMedia', 'IndiceCongestione']].dtypes)
    #    print("Statistiche descrittive di df_traffic (dopo load_traffic_data):\n", df_traffic[['ConteggioVeicoli', 'VelocitaMedia', 'IndiceCongestione']].describe())
    #else:
    #    print("df_traffic è None o vuoto dopo load_traffic_data (passerà alla generazione fittizia).")
    #print("---------------------------------------------------\n")


    traffic_aggregated_df = None # Inizializza a None

    if df_traffic is None or df_traffic.empty:
        print(f"ATTENZIONE: Dati traffico da {traffic_filepath} non disponibili o vuoti. Verrà creato un dataset fittizio IN TEMPO REALE.")

        ###df_traffic = generate_fake_traffic_data(
        ###    G=G,
        ###    nodes_gdf=nodes_gdf,
        ###    pois_gdf=traff_pois_gdf,
        ###    city_name=city_name,
        ###    num_sensors=cities_data[city_name_lower]["num_traffic_sensors"],
        ###    num_readings_per_sensor=cities_data[city_name_lower]["num_readings_per_sensor"],
        ###    center_lat=city_center_from_pois_lat,
        ###    center_lon=city_center_from_pois_lon,
        ###    lat_std=cities_data[city_name_lower]["lat_std"],
        ###    lon_std=cities_data[city_name_lower]["lon_std"],
        ###    buffer_distance_poi=cities_data[city_name_lower]["buffer_distance_meters"]
        ###)

        df_traffic = generate_fake_traffic_data(
            G=G, # Passa l'intero grafo
            nodes_gdf_full_area=nodes_gdf_full_area, # Passa tutti i nodi della città
            pois_gdf=pois_gdf, # Passa tutti i POI della città
            city_name=city_name,
            num_sensors=cities_data[city_name_lower]["num_traffic_sensors"], # Numero di sensori desiderato
            num_readings_per_sensor=cities_data[city_name_lower]["num_readings_per_sensor"], # Numero di letture per sensore
            buffer_distance_poi=cities_data[city_name_lower]["buffer_distance_meters"], # Distanza buffer per POI
        )

        # Aggiungi un punto di controllo QUI per vedere cosa generate_fake_traffic_data ha prodotto
        #print("\n--- DEBUG STEP 2: df_traffic DOPO generate_fake_traffic_data ---")
        #print("Dtype di df_traffic (dopo generate_fake_data):\n", df_traffic[['ConteggioVeicoli', 'VelocitaMedia', 'IndiceCongestione']].dtypes)
        #print("Statistiche descrittive di df_traffic (dopo generate_fake_data):\n", df_traffic[['ConteggioVeicoli', 'VelocitaMedia', 'IndiceCongestione']].describe())
        #print("---------------------------------------------------\n")


        df_traffic.to_excel(traffic_filepath, index=False)
        #print(f"Dataset traffico fittizio generato in tempo reale e salvato: {traffic_filepath}.")
        print("\nDataFrame di traffico fittizio generato:")
        print(df_traffic.head())
        print(f"Shape del DataFrame di traffico: {df_traffic.shape}")

    else:
        print(f"Dati traffico caricati da: {traffic_filepath}")


    print("\nFase 2: Pre-elaborazione e Ingegneria delle Feature...")
    gdf_traffic_processed = preprocess_traffic_data(df_traffic)
    traffic_aggregated_df = aggregate_traffic_to_osm_elements(gdf_traffic_processed, G)

    gdf_incidents_processed = preprocess_accidents(df_accidents)
    nodes_features, edges_features = get_node_edge_features_from_osm(G, pois_gdf)

    ###################################
    # Assicurati che questi center_lat/lon siano gli stessi che usi per generate_fake_data e per scaricare il grafo
    print("Generazione della mappa di distribuzione dei dati di incidenti/traffico...")
    map_center_lat = cities_data[city_name_lower]["center_lat"]
    map_center_lon = cities_data[city_name_lower]["center_lon"]

    debug_map = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=13) # Aumenta lo zoom se i punti sono troppo piccoli
    nodes_for_map = ox.graph_to_gdfs(G, edges=False)
    if nodes_for_map.crs != "EPSG:4326":
        nodes_for_map = nodes_for_map.to_crs(epsg=4326)

    # Aggiungi i punti di traffico
    if gdf_traffic_processed is not None and not gdf_traffic_processed.empty:
        # Assicurati che il CRS sia EPSG:4326 per la visualizzazione con Folium
        traffic_for_map = gdf_traffic_processed.to_crs(epsg=4326) if gdf_traffic_processed.crs != "EPSG:4326" else gdf_traffic_processed

        # Creiamo una copia del GeoDataFrame per evitare di modificare l'originale
        # e per essere sicuri che tutte le operazioni avvengano sulla versione che andrà sulla mappa.
        traffic_for_map_clean = traffic_for_map.copy()

    if 'Timestamp' in traffic_for_map_clean.columns:
        # Converte la colonna 'Timestamp' in una stringa formattata
        traffic_for_map_clean['Timestamp_str'] = traffic_for_map_clean['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        # Rimuovi la colonna originale 'Timestamp' che non è serializzabile in JSON
        traffic_for_map_clean = traffic_for_map_clean.drop(columns=['Timestamp'])

        # Aggiungi un LayerGroup per il traffico
        #traffic_points_layer = folium.FeatureGroup(name='Traffic Points').add_to(debug_map)
        traffic_points_layer = folium.FeatureGroup(name='Traffico').add_to(debug_map)
        folium.features.GeoJson(
            traffic_for_map_clean,
            marker=folium.CircleMarker(radius=2, fill=True, fill_opacity=0.7, color='blue', fill_color='blue'),
            #tooltip=folium.features.GeoJsonTooltip(fields=['ConteggioVeicoli', 'VelocitaMedia', 'IndiceCongestione'])
            tooltip=folium.features.GeoJsonTooltip(fields=['ConteggioVeicoli', 'VelocitaMedia', 'IndiceCongestione', 'Timestamp_str'])
        ).add_to(traffic_points_layer)
        print(f"Aggiunti {len(traffic_for_map_clean)} punti di traffico alla mappa.")
    else:
        print("Nessun punto di traffico da aggiungere alla mappa (gdf_traffic_processed è vuoto o None).")

    # Aggiungi i punti degli incidenti
    if gdf_incidents_processed is not None and not gdf_incidents_processed.empty:
        # Assicurati che il CRS sia EPSG:4326 per la visualizzazione con Folium
        accidents_for_map = gdf_incidents_processed.to_crs(epsg=4326) if gdf_incidents_processed.crs != "EPSG:4326" else gdf_incidents_processed

        # Aggiungi un LayerGroup per gli incidenti
        #accident_points_layer = folium.FeatureGroup(name='Accident Points').add_to(debug_map)
        accident_points_layer = folium.FeatureGroup(name='Incidenti').add_to(debug_map)
        folium.features.GeoJson(
            accidents_for_map,
            marker=folium.CircleMarker(radius=3, fill=True, fill_opacity=0.8, color='red', fill_color='red'),
            tooltip=folium.features.GeoJsonTooltip(fields=['Gravita'])
        ).add_to(accident_points_layer)
        print(f"Aggiunti {len(accidents_for_map)} punti di incidenti alla mappa.")
    else:
        print("Nessun punto di incidente da aggiungere alla mappa (gdf_incidents_processed è vuoto o None).")

    # Aggiungi i punti POI
    if pois_gdf is not None and not pois_gdf.empty: # Assumendo che pois_gdf sia il GeoDataFrame dei POI
        # Assicurati che il CRS sia EPSG:4326 per la visualizzazione con Folium
        pois_for_map = pois_gdf.to_crs(epsg=4326) if pois_gdf.crs != "EPSG:4326" else pois_gdf

        # Creiamo una copia del GeoDataFrame per evitare di modificare l'originale
        # e per essere sicuri che tutte le operazioni avvengano sulla versione che andrà sulla mappa.
        pois_for_map_clean = pois_for_map.copy()

        #print("\n--- DEBUG: Informazioni sulle colonne del GeoDataFrame POI ---")
        #print(pois_for_map_clean.info()) --> Per trovare colonna DateTime
        #print("--- FINE DEBUG ---")

        for col in pois_for_map_clean.columns:
            # Controlla se la colonna è di tipo datetime64 (in qualsiasi sua variante, es. con o senza timezone)
            if pd.api.types.is_datetime64_any_dtype(pois_for_map_clean[col]):
                print(f"Trovata colonna '{col}' di tipo datetime. Conversione in stringa...")
                # Converti la colonna in stringa. Il formato '%Y-%m-%d %H:%M:%S' è un buon default.
                # Se la colonna potrebbe contenere NaN, è bene gestirli per evitare errori,
                # anche se .dt.strftime di solito li converte in 'NaT' che poi diventa 'nan' come stringa.
                # Possiamo usare .fillna('') per rimuovere i 'nan' se non vogliamo vederli.
                pois_for_map_clean[col] = pois_for_map_clean[col].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
                print(f"Colonna '{col}' convertita in stringa.")
            # Puoi aggiungere un 'elif' per altri tipi problematici se ne emergono in futuro
            # elif pd.api.types.is_timedelta64_dtype(pois_for_map_clean[col]):
            #     print(f"Trovata colonna '{col}' di tipo timedelta. Conversione in stringa...")
            #     pois_for_map_clean[col] = pois_for_map_clean[col].astype(str)
            #     print(f"Colonna '{col}' convertita in stringa.")

        print("Fine controllo e conversione delle colonne di tipo datetime.")
        ###if 'check_date' in pois_for_map_clean.columns:
        ###    # Aggiungiamo un controllo extra per assicurarci che sia effettivamente un tipo datetime
        ###    if pd.api.types.is_datetime64_any_dtype(pois_for_map_clean['check_date']):
        ###        #Converti la colonna 'check_date' in una stringa formattata
        ###        pois_for_map_clean['check_date_str'] = pois_for_map_clean['check_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        ###        # Rimuovi la colonna originale 'check_date' che non è serializzabile in JSON
        ###        pois_for_map_clean = pois_for_map_clean.drop(columns=['check_date'])
        ###        print("Colonna 'check_date' convertita in stringa e rimossa l'originale.")
        ###    else:
        ###        print("AVVISO: La colonna 'check_date' esiste ma non è di tipo datetime. Nessuna conversione necessaria.")


        # Aggiungi un LayerGroup per i POI
        #poi_points_layer = folium.FeatureGroup(name='POI Points').add_to(debug_map)
        poi_points_layer = folium.FeatureGroup(name='Punti di Interesse').add_to(debug_map)
        folium.features.GeoJson(
            pois_for_map_clean,
            marker=folium.CircleMarker(radius=2, fill=True, fill_opacity=0.7, color='green', fill_color='green'),
            #tooltip=folium.features.GeoJsonTooltip(fields=['amenity', 'name']) # Adatta i campi in base alle tue colonne POI
            tooltip=folium.features.GeoJsonTooltip(
                fields=['amenity', 'name'],
                aliases=['Categoria:', 'Nome:']
            )
        ).add_to(poi_points_layer)
        print(f"Aggiunti {len(pois_for_map_clean)} punti POI alla mappa.")
    else:
        print("Nessun punto POI da aggiungere alla mappa (gdf_pois è vuoto o None).")

    # Aggiungi il controllo dei layer (per attivare/disattivare i tipi di punti sulla mappa)
    folium.LayerControl().add_to(debug_map)

    # Salva la mappa
    incident_traffic_filepath = f"data/distribuzione_incidenti_traffico_{city_name_lower}.html"
    debug_map.save(incident_traffic_filepath)
    print(f"Mappa della distribuzione dei dati di incidenti/traffico salvata: {incident_traffic_filepath}")
    ###################################

    #print(f"CRS incidents_gdf: {gdf_accidents_processed.crs}")
    #print(f"CRS pois_gdf: {pois_gdf.crs}")
    #print(f"CRS gdf_traffic_processed: {gdf_traffic_processed.crs}")
    #print(f"CRS del grafo G: {ox.graph_to_gdfs(G, edges=False).crs}")

    ## Dati demografici:
    #df_demographics = load_demographics("data/dati_demografici.xlsx")
    #if df_demographics is None or df_demographics.empty:
    #    print("ATTENZIONE: Dati demografici non disponibili o vuoti. L'analisi potrebbe essere meno completa.")

    #df_traffic_processed = preprocess_traffic(df_traffic)
    #df_demographics_processed = preprocess_demographics(df_demographics, city_name=city_name)

    # --- 3. Integrazione Dati (Geospatial Joining) ---
    print("\n--- Fase 3: Integrazione Feature e Preparazione per il Modellamento ---")

    # 1. Aggregazione di tutte le feature per nodo OSM
    # Questa è la chiamata alla funzione che unisce tutto --> L'output sarà il dataset pronto per l'addestramento
    final_features_df = aggregate_features_by_node(
        nodes_features_gdf=nodes_features,      # Output di get_node_edge_features_from_osm
        incidents_gdf=gdf_incidents_processed,           # Output di preprocess_incident_data
        traffic_aggregated_df=traffic_aggregated_df, # Output di aggregate_traffic_to_osm_elements
        pois_gdf=pois_gdf,                     # Output di preprocess_poi_data
        buffer_distance=cities_data[city_name_lower]["buffer_distance_meters"]
    )

    print(f"Dataset finale con feature integrate creato. Dimensioni: {final_features_df.shape}")
    print("Prime 5 righe del dataset finale:")
    print(final_features_df.head())

    #DA METTERE ALLA FINE DELLA FASE 3
    # --- Salva il dataset finale per un uso futuro ---
    print("\n--- Salvataggio del Dataset Finale ---")

    output_dir = 'data/processed' # Cartella dove salvare il dataset pulito/integrato
    os.makedirs(output_dir, exist_ok=True) # Crea la cartella se non esiste

    output_filepath = os.path.join(output_dir, f'final_features_df_{city_name_lower}.xlsx')
    #final_features_df.to_csv(output_filepath_csv, index=False) # index=False per non salvare l'indice predefinito
    #final_features_df.to_csv(output_filepath_csv, index=False, float_format='%.2f', decimal=',')

    # Crea la nuova colonna target binaria ---
    # Questa colonna indicherà se c'è stato almeno un incidente (1) o nessuno (0)
    # Assicurati che 'num_incidenti_vicini' esista e sia numerica

    #TARGET_COLUMN_BINARY = 'ha_incidente_vicino'
    if 'num_incidenti_vicini' in final_features_df.columns:
        final_features_df[TARGET_COLUMN_BINARY] = (final_features_df['num_incidenti_vicini'] > 0).astype(int)
        print(f"Colonna {TARGET_COLUMN_BINARY} creata per la classificazione binaria.")
        print(f"Distribuzione {TARGET_COLUMN_BINARY}:\n{final_features_df[TARGET_COLUMN_BINARY].value_counts()}")
    else:
        print("ATTENZIONE: La colonna 'num_incidenti_vicini' non è presente, impossibile creare il target binario.")

    final_features_df.to_excel(output_filepath, index=False, float_format='%.2f')
    print(f"Dataset finale salvato in: {output_filepath}")

    print("\n--- Fase 4: Previsione ---")
    # 2. Definizione della Variabile Target e delle Feature
    #TARGET_COLUMN = 'num_incidenti_vicini' # Questa è la colonna che vogliamo prevedere

    # Crea la lista delle feature (tutte le colonne tranne 'osmid', la colonna target e num_incidenti_vicini)
    # Rimosso num_incidenti_vicini e ha_incidente_vicino per evitare Data Leakage
    #features = [col for col in final_features_df.columns if col not in ['osmid', 'num_scuole_vicine', 'num_incidenti_vicini', TARGET_COLUMN_BINARY]]
    features = [col for col in final_features_df.columns if col not in ['osmid', 'num_incidenti_vicini', TARGET_COLUMN_BINARY]]

    # 1. Separare Features (X) e Target (y)
    X = final_features_df[features] # DataFrame delle feature
    y = final_features_df[TARGET_COLUMN_BINARY] # Series della variabile target binaria

    print(f"\nVariabile target per il modello '{TARGET_COLUMN_BINARY}' definita.")
    print(f"Numero di feature per il modello: {len(features)}")
    # Aggiungi un print per vedere le features effettivamente usate
    print("Features usate nel modello:", features) # DEBUG: Controlla che num_incidenti_vicini non sia qui
    print(f"Prime 5 righe delle feature (X):\n{X.head()}")
    print(f"Prime 5 righe della target (y):\n{y.head()}")
    #print(f"Numero di feature per il modello: {len(features)}")
    #print(f"Prime 5 righe delle feature (X):\n{X.head()}")
    #print(f"Prime 5 righe della target (y):\n{y.head()}")

    # 3. Gestione di valori mancanti finali (se presenti)
    if X.isnull().sum().sum() > 0:
        print("\nATTENZIONE: Valori mancanti rilevati nelle feature dopo l'integrazione. Riempimento con 0.")
        X = X.fillna(0)
    else:
        print("\nNessun valore mancante rilevato nelle feature.")

    # 4. Suddivisione del Dataset in Training e Test Set
    # test_size=0.2 significa 20% per il test, 80% per il training
    # random_state per riproducibilità dei risultati
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Aggiunto stratify=y per mantenere la proporzione delle classi (0 e 1) sia nel training che nel test set.

    print("\nDataset suddiviso in Training e Test Set:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("\nAddestramento del modello Random Forest Classifier...")
    #model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                                   class_weight='balanced', max_depth=7,
                                   #min_samples_leaf=10)
                                   min_samples_leaf=5)
    # class_weight='balanced' può aiutare a gestire lo sbilanciamento delle classi.
    model.fit(X_train, y_train)

    print("Modello addestrato con successo.")

    #print("\nAddestramento del modello Random Forest Regressor...")
    #model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 per usare tutti i core
    #model.fit(X_train, y_train)
    #print("Modello addestrato con successo.")

    # Valutare il modello
    #print("\nValutazione del modello...")
    #y_pred = model.predict(X_test)

    print("\nValutazione del modello...")
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilità della classe 1 (incidente)
    y_pred_class = model.predict(X_test) # Classe prevista (0 o 1)

    ###y_pred_proba_lgbm_model = lgbm_model.predict_proba(X_test)[:, 1] # Probabilità della classe 1 (incidente)
    ###y_pred_class_lgbm_model = lgbm_model.predict(X_test) # Classe prevista (0 o 1)

    # Arrotonda le previsioni a numeri interi se num_incidenti_vicini è un conteggio
    #y_pred_rounded = np.round(y_pred)

    print(f"Accuracy: {accuracy_score(y_test, y_pred_class):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred_class, zero_division=0):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred_class, zero_division=0):.2f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_class, zero_division=0):.2f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.2f}")
    print("Matrice di Confusione:\n", confusion_matrix(y_test, y_pred_class))

    ###print(f"lgbm_model --> Accuracy: {accuracy_score(y_test, y_pred_class_lgbm_model):.2f}")
    ###print(f"lgbm_model --> Precision: {precision_score(y_test, y_pred_class_lgbm_model, zero_division=0):.2f}")
    ###print(f"lgbm_model --> Recall: {recall_score(y_test, y_pred_class_lgbm_model, zero_division=0):.2f}")
    ###print(f"lgbm_model --> F1-Score: {f1_score(y_test, y_pred_class_lgbm_model, zero_division=0):.2f}")
    ###print(f"lgbm_model --> ROC AUC: {roc_auc_score(y_test, y_pred_proba_lgbm_model):.2f}")
    ###print("lgbm_model --> Matrice di Confusione:\n", confusion_matrix(y_test, y_pred_class_lgbm_model))

    #mae = mean_absolute_error(y_test, y_pred_rounded)
    #rmse = np.sqrt(mean_squared_error(y_test, y_pred_rounded))
    #r2 = r2_score(y_test, y_pred_rounded)

    #print(f"MAE (Mean Absolute Error): {mae:.2f}")
    #print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    #print(f"R² (Coefficiente di Determinazione): {r2:.2f}")

    ## Puoi anche visualizzare le previsioni vs i valori reali per un piccolo campione
    #print("\nConfronto tra valori reali e previsti (primi 10 del test set):")
    #comparison_df = pd.DataFrame({'Reale': y_test, 'Previsto': y_pred_rounded})
    #print(comparison_df.head(10))

    #traffic_aggregated_df = None
    #if traffic_aggregated_df is not None and not traffic_aggregated_df.empty and 'Latitudine' in traffic_aggregated_df.columns and 'Longitudine' in df_traffic_processed.columns:
    #    print("Integrazione dati traffico reali (richiede map-matching). Questa logica è ancora un placeholder complesso.")
    #    pass
    #else:
    #    print("Dati traffico non disponibili, non georeferenziati, o vuoti. Creazione di un dataset di traffico fittizio per nodo.")
    #    # nodes_gdf già ottenuto sopra
    #    traffic_data_placeholder = {
    #        'osmid': nodes_gdf.index.tolist(),
    #        'avg_conteggio_veicoli': np.random.randint(50, 500, size=len(nodes_gdf)),
    #        'avg_velocita': np.random.uniform(10, 60, size=len(nodes_gdf)),
    #        'indice_congestione': np.random.uniform(1, 10, size=len(nodes_gdf)),
    #        'ora_del_giorno': 8,
    #        'giorno_della_settimana': 2
    #    }
    #    traffic_aggregated_df = pd.DataFrame(traffic_data_placeholder)

    print("\nFase 6: Scoring dei Nodi e Generazione Raccomandazioni...")
    # DataFrame dei nodi OSM arricchito con tutte le feature spaziali e contestuali
    ################nodes_features_integrated = assign_spatial_features_to_nodes(nodes_features, pois_gdf, G)

    ###nodes_features_integrated_filepath = f"data/nodes_features_integrated_{city_name_lower}.xlsx"

    #calibrated_model = CalibratedClassifierCV(estimator=model, method='isotonic', cv=5)
    #calibrated_model.fit(X_train, y_train)

    severity_features_df = calculate_incident_severity_features(
        nodes_features_gdf=nodes_gdf, # O nodes_features_gdf se sai che ha ancora la geometry corretta e CRS
        incidents_gdf=gdf_incidents_processed,
        buffer_distance=cities_data[city_name_lower]["buffer_distance_meters"]
    )
    print(f"Gravità media incidenti calcolata per {severity_features_df.shape[0]} nodi.")
    #print(f"Severity features head:\n{severity_features_df.head()}")

    # Unisci la gravità media incidenti al final_features_df per lo scoring e le raccomandazioni
    # Assicurati che final_features_df sia indicizzato per 'osmid' o abbia 'osmid' come colonna
    # (è già una colonna dopo final_features_df.reset_index() alla fine di aggregate_features_by_node)
    final_features_for_scoring = final_features_df.merge(
        severity_features_df,
        on='osmid',
        how='left'
    )
    # Riempi i NaN per i nodi senza incidenti vicini per la gravità media
    final_features_for_scoring['gravita_media_incidente'] = final_features_for_scoring['gravita_media_incidente'].fillna(0.0)
    print(f"Dataset finale per lo scoring arricchito con gravita_media_incidente. Dimensioni: {final_features_for_scoring.shape}")
    print(f"Prime 5 righe del dataset per lo scoring (con gravita_media_incidente):\n{final_features_for_scoring.head()}")

    #### SALVARE DOPO? MAGARI DANDO UNO SCOERE DI RISCHIO, NON SOLO GRAVITA'
    #output_final_features_filepath = os.path.join(output_dir, 'final_features_for_scoring.xlsx')
    #final_features_for_scoring.to_excel(output_final_features_filepath, index=False, float_format='%.2f')
    #print(f"Dataset finale per scoring e raccomandazioni salvato in: {output_final_features_filepath}")


    # Prevedi le probabilità di rischio per TUTTI i nodi nel dataset finale
    # USA LE STESSE FEATURE (X) USATE PER L'ADDESTRAMENTO DEL MODELLO!

    # Dopo aver addestrato il modello, vuoi ottenere le probabilità di rischio
    # non solo per il set di test, ma per tutti i nodi presenti nel dataset completo dei nodi
    X_all_nodes_for_prediction = final_features_for_scoring[features]

    # Gestisci eventuali NaN residui solo per la predizione (dovrebbero essere 0 se X_train non aveva NaN)
    if X_all_nodes_for_prediction.isnull().sum().sum() > 0:
        print("ATTENZIONE: Trovati NaN nelle feature per la predizione. Riempimento con 0.")
        X_all_nodes_for_prediction = X_all_nodes_for_prediction.fillna(0)

    # Probabilità stimata dal modello che si verifichi almeno un incidente in prossimità di un dato nodo (in futuro)
    # "Quanto è probabile che accada un incidente qui?"
    final_features_for_scoring['probabilità_rischio_incidente_predetta'] = model.predict_proba(X_all_nodes_for_prediction)[:, 1]
    print(f"Probabilità di rischio predette per {final_features_for_scoring['probabilità_rischio_incidente_predetta'].count()} nodi.")

    #output_final_features_filepath = os.path.join(output_dir, 'final_features_for_scoring.xlsx')
    #final_features_for_scoring.to_excel(output_final_features_filepath, index=False, float_format='%.2f')
    #print(f"Dataset finale per scoring e raccomandazioni salvato in: {output_final_features_filepath}")

    # Calcolo dello score combinato
    candidate_locations_scored = score_candidate_locations(
        final_features_for_scoring.copy() # Passa una copia per sicurezza
        #risk_proba_col='probabilità_rischio_incidente_predetta',
        #traffic_count_col='avg_conteggio_veicoli',
        #severity_col='gravita_media_incidente'
    )
    print(f"Nodi candidati. Prime 5 righe:\n{candidate_locations_scored.head()}")

    output_final_features_filepath = os.path.join(output_dir, f'final_features_for_scoring_{city_name_lower}.xlsx')
    candidate_locations_scored.to_excel(output_final_features_filepath, index=False, float_format='%.3f')
    #candidate_locations_scored.to_excel(output_final_features_filepath, index=False, engine='openpyxl')
    print(f"Dataset finale per scoring e raccomandazioni salvato in: {output_final_features_filepath}")

    ###########
    print("--- Statistiche Descrittive per Probabilità di Rischio Predetta ---")
    print(candidate_locations_scored['probabilità_rischio_incidente_predetta'].describe())

    print("\n--- Statistiche Descrittive per Indice di Rischio ---")
    print(candidate_locations_scored['indice_rischio'].describe())

    # Potrebbe essere utile anche la mediana per capire la distribuzione
    print(f"\nMediana Probabilità Rischio: {candidate_locations_scored['probabilità_rischio_incidente_predetta'].median():.4f}")
    print(f"Mediana Indice Rischio: {candidate_locations_scored['indice_rischio'].median():.4f}")

    # Imposta lo stile per una migliore leggibilità
    sns.set_style("whitegrid")

    # Grafico a dispersione (Scatter Plot): probabilità vs indice di rischio
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='probabilità_rischio_incidente_predetta', y='indice_rischio', data=candidate_locations_scored, alpha=0.5, hue='indice_rischio', palette='viridis', size='indice_rischio', sizes=(20, 400))
    plt.title('Probabilità di Rischio Predetta vs Indice di Rischio')
    plt.xlabel('Probabilità di Rischio Predetta (Modello ML)')
    plt.ylabel('Indice di Rischio (Composito)')
    plt.grid(True)
    plt.show()

    correlation_coefficient = candidate_locations_scored['probabilità_rischio_incidente_predetta'].corr(candidate_locations_scored['indice_rischio'])
    print(f"Coefficiente di Correlazione di Pearson tra 'Probabilità di Rischio Predetta' e 'Indice di Rischio': {correlation_coefficient:.4f}")
    # Vicino a +1: Indica una forte correlazione lineare positiva.
    # Ciò significa che quando la probabilità_rischio_incidente_predetta aumenta,
    # anche l'indice_rischio tende ad aumentare in modo proporzionale.
    # Più il valore è vicino a 1, più forte è la relazione.


    # Istogrammi delle distribuzioni
    ###plt.figure(figsize=(12, 5))
###
    ###plt.subplot(1, 2, 1) # 1 riga, 2 colonne, primo grafico
    ###sns.histplot(candidate_locations_scored['probabilità_rischio_incidente_predetta'], bins=30, kde=True, color='skyblue')
    ###plt.title('Distribuzione Probabilità di Rischio Predetta')
    ###plt.xlabel('Probabilità')
    ###plt.ylabel('Frequenza')
###
    ###plt.subplot(1, 2, 2) # 1 riga, 2 colonne, secondo grafico
    ###sns.histplot(candidate_locations_scored['indice_rischio'], bins=30, kde=True, color='lightcoral')
    ###plt.title('Distribuzione Indice di Rischio')
    ###plt.xlabel('Indice')
    ###plt.ylabel('Frequenza')
###
    ###plt.tight_layout() # Adatta i subplot per evitare sovrapposizioni
    ###plt.show()
    ###########

    # Generazione raccomandazioni
    #TOP_N_RECOMMENDATIONS = 10 # Puoi modificare questo valore
    recommendations = recommend_interventions(candidate_locations_scored)
    #print(f"\nLe prime {TOP_N_RECOMMENDATIONS} raccomandazioni:")
    #for i, rec in enumerate(top_recommendations):
        #print(f"{i+1}. Nodo {rec['osmid']} (Score: {rec['score']:.2f}, Rischio pred. {rec['probabilità_rischio_incidente_predetta']:.2f}, Gravità media storica: {rec['gravita_media_incidente']:.2f}): {rec['recommendation']}")

    # --- 7. Visualizzazione Raccomandazioni su Mappa ---
    print("\nFase 7: Visualizzazione Raccomandazioni...")

    # CONVERTI LA LISTA IN UN DATAFRAME
    if recommendations: # Controlla se la lista non è vuota prima di convertirla
        recommendations_df = pd.DataFrame(recommendations)
        # Ora puoi tranquillamente fare il controllo .empty su recommendations_df
    else:
        print("Nessuna raccomandazione generata.")
        recommendations_df = pd.DataFrame() # Inizializza come DataFrame vuoto se non ci sono raccomandazioni

    # Assicurati di avere la variabile 'city_name' definita, ad esempio:
    # city_name = "Terlizzi, Italy" # O come la stai ottenendo dal tuo flusso

    output_recommendations_df_filepath = os.path.join(output_dir, f'recommendations_df_{city_name_lower}.xlsx')
    recommendations_df.to_excel(output_recommendations_df_filepath, index=False, float_format='%.2f')
    print(f"Dataset prova raccomandazioni salvato in: {output_recommendations_df_filepath}")

    # Prepara il percorso del file per la mappa
    output_map_filename = f"{city_name_lower}_recommendations_map.html"
    output_map_filepath = os.path.join("reports", output_map_filename)

    # Assicurati che la cartella 'reports' esista
    os.makedirs("reports", exist_ok=True)

    # La funzione plot_recommendations_on_map necessita del grafo G e delle raccomandazioni.
    plot_recommendations_on_map(G, recommendations_df, filepath=output_map_filepath)

    print(f"Mappa delle raccomandazioni generata e salvata in: {output_map_filepath}")

    print("\n--- Progetto Completato con Successo! ---")
    print(f"Controlla la cartella 'reports' per il report e le mappe interattive per '{city_name_lower}'.")




    ################nodes_features_integrated.to_excel(nodes_features_integrated_filepath, index=True, float_format='%.2f')
    # index=True per includere l'indice (osmid) come prima colonna nel file Excel
    # float_format='%.2f' per formattare i numeri float con due cifre decimali
    ################print(f"Dataset 'nodes_features_integrated_{city_name_lower}' salvato in: {nodes_features_integrated_filepath}")
    #unified_features_df = create_unified_features_df(nodes_features_integrated, edges_features, accidents_per_node_df, traffic_aggregated_df, G)

    ############################################## DECOMMENTARE IN CASO PER DOPO
    #accidents_per_node_df = assign_accidents_to_osm_elements(gdf_incidents_processed, G)

    ###unified_features_df = create_unified_features_df(nodes_features_integrated, edges_features, accidents_per_node_df, traffic_aggregated_df, G)
###
    ###if unified_features_df.empty:
    ###    print("Errore critico: DataFrame unificato delle features vuoto. Impossibile continuare.")
    ###    return
###
    #### --- 4. Modellazione Predittiva ---
    ####print("\nFase 4: Modellazione Predittiva...")
    ####traffic_model, traffic_imputer, traffic_features_cols = train_traffic_predictor(traffic_aggregated_df)
    ####accident_model, accident_imputer, accident_features_cols = train_accident_predictor(unified_features_df)
###
    #### --- 5. Ottimizzazione e Generazione Raccomandazioni ---
    ###print("\nFase 5: Ottimizzazione e Generazione Raccomandazioni...")
    ###candidate_locations_scored = score_candidate_locations(
    ###    unified_features_df,
    ###    traffic_model, accident_model,
    ###    traffic_imputer, accident_imputer,
    ###    traffic_features_cols, accident_features_cols
    ###)
###
    ###top_recommendations = recommend_interventions(candidate_locations_scored, top_n=10)
###
    #### --- 6. Valutazione e Reportistica ---
    ###print("\nFase 6: Valutazione e Reportistica...")
    ###traffic_metrics = None
    ###if traffic_model:
    ###    if traffic_aggregated_df is not None and not traffic_aggregated_df.empty:
    ###        features = traffic_aggregated_df[traffic_features_cols]
    ###        target_flow = traffic_aggregated_df['avg_conteggio_veicoli']
    ###        features_processed = traffic_imputer.transform(features)
    ###        y_pred = traffic_model.predict(features_processed)
    ###        traffic_metrics = {
    ###            'mae': mean_absolute_error(target_flow, y_pred),
    ###            'rmse': np.sqrt(mean_squared_error(target_flow, y_pred))
    ###        }
    ###    else:
    ###        traffic_metrics = {'mae': 0.0, 'rmse': 0.0}
###
    ###accident_metrics = None
    ###if accident_model:
    ###    features_for_pred = unified_features_df[accident_features_cols]
    ###    features_processed = accident_imputer.transform(features_for_pred)
    ###    if 'ha_rischio_incidente_alto' in unified_features_df.columns:
    ###        y_true = unified_features_df['ha_rischio_incidente_alto']
    ###        y_pred_proba = accident_model.predict_proba(features_processed)[:, 1]
    ###        y_pred = accident_model.predict(features_processed)
###
    ###        accident_metrics = {
    ###            'roc_auc': roc_auc_score(y_true, y_pred_proba),
    ###            'f1_score': f1_score(y_true, y_pred),
    ###            'precision': precision_score(y_true, y_pred),
    ###            'recall': recall_score(y_true, y_pred)
    ###        }
    ###    else:
    ###        accident_metrics = {'roc_auc': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}
###
    ###generate_report(top_recommendations, traffic_metrics, accident_metrics)

    # --- 7. Visualizzazione ---
    ###print("\nFase 7: Visualizzazione Raccomandazioni...")
    ###output_map_filename = f"{city_name.replace(', ', '_').replace(' ', '_').lower()}_recommendations_map.html"
    ###output_map_filepath = os.path.join("reports", output_map_filename)
###
    #### Assicurati che la cartella 'reports' esista
    ###os.makedirs("reports", exist_ok=True)
    ###plot_recommendations_on_map(G, top_recommendations, filepath=output_map_filepath)
###
    ###print("\n--- Progetto Completato con Successo! ---")
    ###print(f"Controlla la cartella 'reports' per il report e le mappe interattive per '{city_name}'.")

#CON DEMOGRAFIA
    #city_name_lower = city_name.replace(', ', '_').replace(' ', '_').replace('/', '_').lower()
#
    #city_config = cities_data.get(city_name_lower)
    #if not city_config:
    #    raise ValueError(f"Configurazione non trovata per la città: '{city_name}'. Assicurati che sia definita nel dizionario 'cities_data'.")
#
    #city_avg_pop_density = city_config.get("avg_population_density", 0.0) # Ottiene la densità media di popolazione

    ## Creazione directory di output se non esistono
    #os.makedirs('data', exist_ok=True)
    #os.makedirs('data/processed_data', exist_ok=True)
    #os.makedirs('reports', exist_ok=True)
    #os.makedirs('models', exist_ok=True)

    ## --- Acquisizione Dati OSM (Rete Stradale) ---
    ##print("\n--- Acquisizione Dati OSM (Rete Stradale) ---")
    ### Questa riga ora chiama la funzione get_osm_data da src/data_acquisition.py
    ### che gestisce il caching e lo scaricamento
    ##G = get_osm_data(city_name)
    ##nodes_features, edges_features = get_node_edge_features_from_osm(G)
    ##print("Dati OSM e features base dei nodi/archi acquisiti.")
#
    ###
    ## Fase 1: Acquisizione Dati OSM
    #print("\n--- Fase 1: Acquisizione Dati OSM ---")
    #G, nodes_gdf, edges_gdf = get_osm_data(city_name) # Nota: qui ottieni nodes_gdf ed edges_gdf iniziali
    #print(f"Acquisiti {len(nodes_gdf)} nodi e {len(edges_gdf)} archi per {city_name}.")
    ###
#
    ## --- Acquisizione e Pre-elaborazione POI ---
    #print("\n--- Acquisizione e Pre-elaborazione POI ---")
    ## Definisci i tag per i POI che ti interessano
    #tags_pois = {'amenity': ['school', 'hospital', 'bus_station'], 'public_transport': 'bus_stop'}
#
    ## Esempio: se vuoi salvare i POI in un file specifico per città
    #pois_filepath = os.path.join('data', f"pois_data_{city_name_lower}.csv")
    #df_pois = get_pois_from_osm(city_name, tags_pois, filepath=pois_filepath)
#
    #gdf_pois_processed = preprocess_pois(df_pois)
    #if gdf_pois_processed is None or gdf_pois_processed.empty:
    #    print("Nessun POI pre-elaborato disponibile o GeoDataFrame vuoto.")
    #    gdf_pois_processed = gpd.GeoDataFrame(columns=['amenity', 'public_transport', 'geometry'], crs="EPSG:4326") # Crea un GeoDataFrame vuoto
#
    #print("Dati POI pre-elaborati.")
#
    ## --- Gestione Dati Demografici ---
    ## Per ora, lasciamo df_demographics_processed a None per attivare il fallback
    ## Se avessi un file CSV o Excel per i dati demografici reali, li caricheresti e pre-elaboreresti qui:
    ## try:
    ##     df_demographics_raw = load_demographics("path/to/your/real_demographics.csv")
    ##     df_demographics_processed = preprocess_demographics(df_demographics_raw)
    ## except FileNotFoundError:
    ##     print("File dati demografici reali non trovato. Verrà usato il fallback della densità media della città.")
    ##     df_demographics_processed = None
    ## except Exception as e:
    ##     print(f"Errore nel caricamento/pre-elaborazione dati demografici: {e}. Verrà usato il fallback.")
    ##     df_demographics_processed = None
#
    #df_demographics_processed = None # Imposta a None per usare il fallback della densità media della città
#
    ## --- Pre-elaborazione e Integrazione Features Spaziali (POI e Demografia) ---
    #print("\n--- Pre-elaborazione e Integrazione Features Spaziali ---")
    #nodes_features_integrated = assign_spatial_features_to_nodes(
    #    nodes_features,
    #    gdf_pois_processed,       # GeoDataFrame dei POI (potrebbe essere vuoto)
    #    df_demographics_processed,# Sarà None per usare il fallback demografico
    #    G,
    #    city_avg_pop_density      # Il valore di fallback per la demografia
    #)
    #print("Features spaziali (POI e Demografia) integrate ai nodi.")
#
    ## --- Caricamento e Pre-elaborazione Dati Incidenti ---
    #print("\n--- Caricamento e Pre-elaborazione Dati Incidenti ---")
    #accidents_filepath = os.path.join('data', f"incidenti_stradali_{city_name_lower}.xlsx")
    #df_accidents = load_accidents(accidents_filepath)
    #gdf_accidents_processed = preprocess_accidents(df_accidents)
    #accidents_per_node_df = assign_accidents_to_osm_elements(gdf_accidents_processed, G)
    #print("Dati incidenti pre-elaborati e assegnati ai nodi.")
#
    ## --- Caricamento e Pre-elaborazione Dati Traffico ---
    #print("\n--- Caricamento e Pre-elaborazione Dati Traffico ---")
    #traffic_filepath = os.path.join('data', f"traffic_data_{city_name_lower}.xlsx")
    #df_traffic = load_traffic_data(traffic_filepath)
    #gdf_traffic_processed = preprocess_traffic_data(df_traffic)
    #traffic_aggregated_df = aggregate_traffic_to_osm_elements(gdf_traffic_processed, G)
    #print("Dati traffico pre-elaborati e aggregati ai nodi.")
#
    ## --- Creazione del DataFrame Unificato ---
    #print("\n--- Creazione del DataFrame Unificato ---")
    #unified_features_df = create_unified_features_df(nodes_features_integrated, edges_features, accidents_per_node_df, traffic_aggregated_df, G)
    #print(f"DataFrame unificato creato con {len(unified_features_df)} nodi.")
    ## Salva il dataset unificato per un'analisi futura o debug
    #unified_features_df.to_csv(os.path.join('data', 'processed_data', 'unified_features.csv'), index=True)
    #print("DataFrame unificato salvato in 'data/processed_data/unified_features.csv'.")
#
    ## --- Modellazione e Previsione ---
    #print("\n--- Modellazione e Previsione ---")
    ## La logica del modello è stata resa più robusta per gestire casi di dati mancanti
    #if 'num_incidenti' not in unified_features_df.columns:
    #    print("La colonna 'num_incidenti' non è presente nel DataFrame unificato. Saltando la modellazione.")
    #    recommendations_df = pd.DataFrame(columns=['osmid', 'danger_score', 'recommendation'])
    #else:
    #    unified_features_df['num_incidenti'] = pd.to_numeric(unified_features_df['num_incidenti'], errors='coerce').fillna(0)
#
    #    X = unified_features_df.drop(columns=['num_incidenti', 'geometry'], errors='ignore')
    #    y = unified_features_df['num_incidenti']
#
    #    X = X.select_dtypes(include=np.number)
    #    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
#
    #    if X.empty or len(X.columns) < 1:
    #        print("Non ci sono features valide per la modellazione. Saltando la modellazione.")
    #        recommendations_df = pd.DataFrame(columns=['osmid', 'danger_score', 'recommendation'])
    #    else:
    #        model = train_model(X, y)
    #        predictions = make_predictions(model, X)
    #        recommendations_df = get_recommendations(unified_features_df.index, predictions)
    #        print("Modello addestrato e raccomandazioni generate.")
#
    ## --- Visualizzazione ---
    #print("\n--- Visualizzazione ---")
#
    ## Assicurati che nodes_features_integrated sia un GeoDataFrame con la geometria corretta
    #if 'geometry' not in nodes_features_integrated.columns or not isinstance(nodes_features_integrated, gpd.GeoDataFrame):
    #    # Ricrea la geometria dai nodi del grafo G
    #    nodes_geom_df = ox.graph_to_gdfs(G, edges=False)[['geometry']]
    #    nodes_features_integrated = nodes_features_integrated.merge(nodes_geom_df, left_index=True, right_index=True, how='left')
    #    nodes_features_integrated = gpd.GeoDataFrame(nodes_features_integrated, geometry='geometry', crs=G.graph['crs'])
#
#
    #map_osm_filepath = os.path.join('reports', 'osm_data_map.html')
    #plot_osm_data_on_map(nodes_features_integrated, G, map_osm_filepath)
    #print(f"Mappa dei dati OSM salvata in '{map_osm_filepath}'.")
#
    #map_recs_filepath = os.path.join('reports', 'recommendations_map.html')
    #if not recommendations_df.empty and 'osmid' in recommendations_df.columns:
    #    recommendations_df = recommendations_df.set_index('osmid')
    #    map_data_df = nodes_features_integrated.merge(recommendations_df, left_index=True, right_index=True, how='left')
    #    map_data_df['danger_score'].fillna(0, inplace=True)
    #    plot_recommendations_on_map(map_data_df, G, map_recs_filepath)
    #    print(f"Mappa delle raccomandazioni salvata in '{map_recs_filepath}'.")
    #else:
    #    print("Nessuna raccomandazione valida da mappare o formato errato.")
    #    plot_recommendations_on_map(nodes_features_integrated, G, map_recs_filepath, default_color='gray')
#
    ## --- Generazione Report HTML ---
    #print("\n--- Generazione Report HTML ---")
    #report_filepath = os.path.join('reports', 'report.html')
    #generate_html_report(unified_features_df, report_filepath, map_osm_filepath, map_recs_filepath)
    #print(f"Report HTML salvato in '{report_filepath}'.")
#
    ## Apertura automatica del report e delle mappe nel browser
    #print("\nApertura report e mappe nel browser...")
    #webbrowser.open_new_tab(f"file:///{os.path.abspath(report_filepath)}")
    #webbrowser.open_new_tab(f"file:///{os.path.abspath(map_osm_filepath)}")
    #webbrowser.open_new_tab(f"file:///{os.path.abspath(map_recs_filepath)}")
#
    #print("\n--- Esecuzione completata! ---")

if __name__ == "__main__":
    main()