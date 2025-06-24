import os
import sys

import osmnx as ox
import folium
import folium.plugins
import pandas as pd
import geopandas as gpd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
    recall_score, accuracy_score, confusion_matrix, roc_curve, average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, precision_recall_curve
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
    #city_name = "Terlizzi, Italy"
    #city_name = "Molfetta, Italy"
    city_name = "Bari, Italy"

    MODEL_FILENAME = "random_forest_model.joblib"
    TARGET_COLUMN_BINARY = 'ha_incidente_vicino'

    for folder in ['data', 'reports']:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # --- Acquisizione e Caricamento Dati OSM ---
    print(f"\nFase 0: Acquisizione/Caricamento Dati OpenStreetMap per '{city_name}'...")

    graphml_filepath = get_city_file_path(city_name, 'graphml')
    geojson_filepath = get_city_file_path(city_name, 'geojson_pois')

    G = None
    pois_gdf = None

    # Logica per scaricare o caricare la rete stradale
    if not os.path.exists(graphml_filepath):
        print(f"File della rete stradale per '{city_name}' non trovato. Scaricamento in corso...")
        try:
            G = ox.graph_from_place(city_name, network_type="drive", simplify=True, retain_all=False)
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

    # Calcola il centro dinamico (centroid) dei POI per allineare traffico e incidenti
    if not pois_gdf.empty:
        if pois_gdf.crs and pois_gdf.crs.is_projected:
            pois_gdf_geographic = pois_gdf.to_crs(epsg=4326)
        else:
            pois_gdf_geographic = pois_gdf

        # Calcola il centroid di tutte le geometrie dei POI combinate
        city_center_from_pois_lat = pois_gdf_geographic.geometry.union_all().centroid.y
        city_center_from_pois_lon = pois_gdf_geographic.geometry.union_all().centroid.x
        print(f"Calcolato centro della città dai POI: Lat={city_center_from_pois_lat:.4f}, Lon={city_center_from_pois_lon:.4f}")
    else:
        # Fallback: se per qualche motivo non ci sono POI, usa il centro predefinito dalla configurazione
        city_center_from_pois_lat = cities_data[city_name_lower]["center_lat"]
        city_center_from_pois_lon = cities_data[city_name_lower]["center_lon"]
        print("Nessun POI trovato per calcolare il centro, usando il centro predefinito da CITIES_DATA.")

    # --- Dati Incidenti fittizi o reali ---
    accidents_filepath = f"data/incidenti_stradali_{city_name_lower}.xlsx"
    df_accidents = load_accidents(accidents_filepath)
    if df_accidents is None or df_accidents.empty:
        print(f"ATTENZIONE: Dati incidenti stradali da {accidents_filepath} non disponibili o vuoti. Verrà creato un dataset fittizio IN TEMPO REALE.")

        center_lat_city, center_lon_city = ox.geocode(city_name)

        nodes_gdf_full_area = ox.graph_to_gdfs(G, nodes=True, edges=False)
        nodes_gdf_full_area = nodes_gdf_full_area.to_crs("EPSG:4326")

        nodes_gdf_full_area['Latitudine'] = nodes_gdf_full_area['y']
        nodes_gdf_full_area['Longitudine'] = nodes_gdf_full_area['x']

        fake_incidents_df_raw = generate_fake_accidents_data(
            nodes_gdf_full_area=nodes_gdf_full_area,
            pois_gdf=pois_gdf,
            buffer_distance=cities_data[city_name_lower]["buffer_distance_meters"],
            num_accidents=cities_data[city_name_lower]["num_accidents"],
        )

        df_accidents = gpd.GeoDataFrame(
            fake_incidents_df_raw,
            geometry=gpd.points_from_xy(fake_incidents_df_raw.Longitudine, fake_incidents_df_raw.Latitudine),
            crs="EPSG:4326"
        )

        output_dir = "data"
        filepath_accidents = os.path.join(output_dir, f"incidenti_stradali_{city_name_lower}.xlsx")
        df_accidents.to_excel(filepath_accidents, index=False)
        print(f"Dataset incidenti fittizio generato in tempo reale e salvato: {filepath_accidents}.")
    else:
        print(f"Dati incidenti caricati da: {accidents_filepath}")

    # --- Dati Traffico ---

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

    traffic_aggregated_df = None # Inizializza a None

    if df_traffic is None or df_traffic.empty:
        print(f"ATTENZIONE: Dati traffico da {traffic_filepath} non disponibili o vuoti. Verrà creato un dataset fittizio IN TEMPO REALE.")

        df_traffic = generate_fake_traffic_data(
            G=G, # Passa l'intero grafo
            nodes_gdf_full_area=nodes_gdf_full_area, # Passa tutti i nodi della città
            pois_gdf=pois_gdf, # Passa tutti i POI della città
            city_name=city_name,
            num_sensors=cities_data[city_name_lower]["num_traffic_sensors"], # Numero di sensori
            num_readings_per_sensor=cities_data[city_name_lower]["num_readings_per_sensor"], # Numero di letture per sensore
            buffer_distance_poi=cities_data[city_name_lower]["buffer_distance_meters"], # Distanza buffer per POI
        )

        df_traffic.to_excel(traffic_filepath, index=False)
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

    print("Generazione della mappa di distribuzione dei dati di incidenti/traffico...")
    map_center_lat = cities_data[city_name_lower]["center_lat"]
    map_center_lon = cities_data[city_name_lower]["center_lon"]

    debug_map = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=13)
    nodes_for_map = ox.graph_to_gdfs(G, edges=False)
    if nodes_for_map.crs != "EPSG:4326":
        nodes_for_map = nodes_for_map.to_crs(epsg=4326)

    # Aggiungo i punti di traffico
    if gdf_traffic_processed is not None and not gdf_traffic_processed.empty:
        traffic_for_map = gdf_traffic_processed.to_crs(epsg=4326) if gdf_traffic_processed.crs != "EPSG:4326" else gdf_traffic_processed

        traffic_for_map_clean = traffic_for_map.copy()

    if 'Timestamp' in traffic_for_map_clean.columns:
        # Converte la colonna 'Timestamp' in una stringa formattata
        traffic_for_map_clean['Timestamp_str'] = traffic_for_map_clean['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        # Rimuove la colonna originale 'Timestamp' che non è serializzabile in JSON
        traffic_for_map_clean = traffic_for_map_clean.drop(columns=['Timestamp'])

        # Aggiunge un LayerGroup per il traffico
        traffic_points_layer = folium.FeatureGroup(name='Traffico').add_to(debug_map)
        folium.features.GeoJson(
            traffic_for_map_clean,
            marker=folium.CircleMarker(radius=2, fill=True, fill_opacity=0.7, color='blue', fill_color='blue'),
            tooltip=folium.features.GeoJsonTooltip(fields=['ConteggioVeicoli', 'VelocitaMedia', 'IndiceCongestione', 'Timestamp_str'])
        ).add_to(traffic_points_layer)
        print(f"Aggiunti {len(traffic_for_map_clean)} punti di traffico alla mappa.")
    else:
        print("Nessun punto di traffico da aggiungere alla mappa (gdf_traffic_processed è vuoto o None).")

    # Aggiungo i punti degli incidenti
    if gdf_incidents_processed is not None and not gdf_incidents_processed.empty:
        accidents_for_map = gdf_incidents_processed.to_crs(epsg=4326) if gdf_incidents_processed.crs != "EPSG:4326" else gdf_incidents_processed

        # Aggiunge un LayerGroup per gli incidenti
        accident_points_layer = folium.FeatureGroup(name='Incidenti').add_to(debug_map)
        folium.features.GeoJson(
            accidents_for_map,
            marker=folium.CircleMarker(radius=3, fill=True, fill_opacity=0.8, color='red', fill_color='red'),
            tooltip=folium.features.GeoJsonTooltip(fields=['Gravita'])
        ).add_to(accident_points_layer)
        print(f"Aggiunti {len(accidents_for_map)} punti di incidenti alla mappa.")
    else:
        print("Nessun punto di incidente da aggiungere alla mappa (gdf_incidents_processed è vuoto o None).")

    # Aggiungo i punti POI
    if pois_gdf is not None and not pois_gdf.empty:
        pois_for_map = pois_gdf.to_crs(epsg=4326) if pois_gdf.crs != "EPSG:4326" else pois_gdf

        pois_for_map_clean = pois_for_map.copy()

        for col in pois_for_map_clean.columns:
            # Controlla se la colonna è di tipo datetime64 (in qualsiasi sua variante, es. con o senza timezone)
            if pd.api.types.is_datetime64_any_dtype(pois_for_map_clean[col]):
                print(f"Trovata colonna '{col}' di tipo datetime. Conversione in stringa...")
                pois_for_map_clean[col] = pois_for_map_clean[col].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
                print(f"Colonna '{col}' convertita in stringa.")

        print("Fine controllo e conversione delle colonne di tipo datetime.")

        # Aggiunge un LayerGroup per i POI
        poi_points_layer = folium.FeatureGroup(name='Punti di Interesse').add_to(debug_map)
        folium.features.GeoJson(
            pois_for_map_clean,
            marker=folium.CircleMarker(radius=2, fill=True, fill_opacity=0.7, color='green', fill_color='green'),
            tooltip=folium.features.GeoJsonTooltip(
                fields=['amenity', 'name'],
                aliases=['Categoria:', 'Nome:']
            )
        ).add_to(poi_points_layer)
        print(f"Aggiunti {len(pois_for_map_clean)} punti POI alla mappa.")
    else:
        print("Nessun punto POI da aggiungere alla mappa (gdf_pois è vuoto o None).")

    # Aggiunge il controllo dei layer (per attivare/disattivare i tipi di punti sulla mappa)
    folium.LayerControl().add_to(debug_map)

    # Salva la mappa
    incident_traffic_filepath = f"data/distribuzione_incidenti_traffico_{city_name_lower}.html"
    debug_map.save(incident_traffic_filepath)
    print(f"Mappa della distribuzione dei dati di incidenti/traffico salvata: {incident_traffic_filepath}")

    # --- 3. Integrazione Dati ---
    print("\n--- Fase 3: Integrazione Feature e Preparazione per il Modellamento ---")

    # Aggregazione di tutte le feature per nodo OSM
    # Questa è la chiamata alla funzione che unisce tutto --> L'output sarà il dataset pronto per l'addestramento
    final_features_df = aggregate_features_by_node(
        nodes_features_gdf=nodes_features,
        incidents_gdf=gdf_incidents_processed,
        traffic_aggregated_df=traffic_aggregated_df,
        pois_gdf=pois_gdf,
        buffer_distance=cities_data[city_name_lower]["buffer_distance_meters"]
    )

    print(f"Dataset finale con feature integrate creato. Dimensioni: {final_features_df.shape}")
    print("Prime 5 righe del dataset finale:")
    print(final_features_df.head())

    # --- Salva il dataset finale ---
    print("\n--- Salvataggio del Dataset Finale ---")

    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True) # Crea la cartella se non esiste

    output_filepath = os.path.join(output_dir, f'final_features_df_{city_name_lower}.xlsx')

    if 'num_incidenti_vicini' in final_features_df.columns:
        final_features_df[TARGET_COLUMN_BINARY] = (final_features_df['num_incidenti_vicini'] > 0).astype(int)
        print(f"Colonna {TARGET_COLUMN_BINARY} creata per la classificazione binaria.")
        print(f"Distribuzione {TARGET_COLUMN_BINARY}:\n{final_features_df[TARGET_COLUMN_BINARY].value_counts()}")
    else:
        print("ATTENZIONE: La colonna 'num_incidenti_vicini' non è presente, impossibile creare il target binario.")

    final_features_df.to_excel(output_filepath, index=False, float_format='%.2f')
    print(f"Dataset finale salvato in: {output_filepath}")

    print("\n--- Fase 4: Previsione ---")
    # Definizione della Variabile Target e delle Feature
    # Rimosso num_incidenti_vicini e ha_incidente_vicino (oltre la Y) per evitare Data Leakage
    features = [col for col in final_features_df.columns if col not in ['osmid', 'num_incidenti_vicini', TARGET_COLUMN_BINARY]]

    # Separa Features (X) e Target (y)
    X = final_features_df[features] # DataFrame delle feature
    y = final_features_df[TARGET_COLUMN_BINARY] # Series della variabile target binaria

    print(f"\nVariabile target per il modello '{TARGET_COLUMN_BINARY}' definita.")
    print(f"Numero di feature per il modello: {len(features)}")
    print("Features usate nel modello:", features)
    print(f"Prime 5 righe delle feature (X):\n{X.head()}")
    print(f"Prime 5 righe della target (y):\n{y.head()}")

    # Gestione di valori mancanti finali (se presenti)
    if X.isnull().sum().sum() > 0:
        print("\nATTENZIONE: Valori mancanti rilevati nelle feature dopo l'integrazione. Riempimento con 0.")
        X = X.fillna(0)
    else:
        print("\nNessun valore mancante rilevato nelle feature.")

    # Suddivisione del Dataset in Training e Test Set
    # test_size=0.2 significa 20% per il test, 80% per il training
    # random_state per riproducibilità dei risultati
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Aggiunto stratify=y per mantenere la proporzione delle classi (0 e 1) sia nel training che nel test set

    print("\nDataset suddiviso in Training e Test Set:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("\nAddestramento del modello Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                                   class_weight='balanced', max_depth=7,
                                   min_samples_leaf=5)
    # class_weight='balanced' può aiutare a gestire lo sbilanciamento delle classi.
    model.fit(X_train, y_train)
    print("Modello addestrato con successo.")

    print("\nValutazione del modello...")
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilità della classe 1 (incidente)
    y_pred_class = model.predict(X_test) # Classe prevista (0 o 1)

    print(f"Accuracy: {accuracy_score(y_test, y_pred_class):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred_class, zero_division=0):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred_class, zero_division=0):.2f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_class, zero_division=0):.2f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.2f}")
    print("Matrice di Confusione:\n", confusion_matrix(y_test, y_pred_class))

    print("\nFase 6: Scoring dei Nodi e Generazione Raccomandazioni...")

    # Calcola la gravità media degli incidenti
    severity_features_df = calculate_incident_severity_features(
        nodes_features_gdf=nodes_gdf,
        incidents_gdf=gdf_incidents_processed,
        buffer_distance=cities_data[city_name_lower]["buffer_distance_meters"]
    )
    print(f"Gravità media incidenti calcolata per {severity_features_df.shape[0]} nodi.")

    # Unisce la gravità media incidenti al final_features_df per lo scoring e le raccomandazioni
    final_features_for_scoring = final_features_df.merge(
        severity_features_df,
        on='osmid',
        how='left'
    )
    # Riempie i NaN per i nodi senza incidenti vicini per la gravità media
    final_features_for_scoring['gravita_media_incidente'] = final_features_for_scoring['gravita_media_incidente'].fillna(0.0)
    print(f"Dataset finale per lo scoring arricchito con gravita_media_incidente. Dimensioni: {final_features_for_scoring.shape}")
    print(f"Prime 5 righe del dataset per lo scoring (con gravita_media_incidente):\n{final_features_for_scoring.head()}")

    # Dopo aver addestrato il modello, vogliamo ottenere le probabilità di rischio
    X_all_nodes_for_prediction = final_features_for_scoring[features]

    # Gestisce eventuali NaN residui solo per la predizione (dovrebbero essere 0 se X_train non aveva NaN)
    if X_all_nodes_for_prediction.isnull().sum().sum() > 0:
        print("ATTENZIONE: Trovati NaN nelle feature per la predizione. Riempimento con 0.")
        X_all_nodes_for_prediction = X_all_nodes_for_prediction.fillna(0)

    # Probabilità stimata dal modello che si verifichi almeno un incidente in prossimità di un dato nodo (in futuro)
    final_features_for_scoring['probabilità_rischio_incidente_predetta'] = model.predict_proba(X_all_nodes_for_prediction)[:, 1]
    print(f"Probabilità di rischio predette per {final_features_for_scoring['probabilità_rischio_incidente_predetta'].count()} nodi.")

    # Calcolo dello score combinato
    candidate_locations_scored = score_candidate_locations(
        final_features_for_scoring.copy() # Passa una copia per sicurezza
    )
    print(f"Nodi candidati. Prime 5 righe:\n{candidate_locations_scored.head()}")

    output_final_features_filepath = os.path.join(output_dir, f'final_features_for_scoring_{city_name_lower}.xlsx')
    candidate_locations_scored.to_excel(output_final_features_filepath, index=False, float_format='%.3f')
    print(f"Dataset finale per scoring e raccomandazioni salvato in: {output_final_features_filepath}")

    print("--- Statistiche Descrittive per Probabilità di Rischio Predetta ---")
    print(candidate_locations_scored['probabilità_rischio_incidente_predetta'].describe())

    print("\n--- Statistiche Descrittive per Indice di Rischio ---")
    print(candidate_locations_scored['indice_rischio'].describe())

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
    recommendations = recommend_interventions(candidate_locations_scored)

    # --- 7. Visualizzazione Raccomandazioni su Mappa ---
    print("\nFase 7: Visualizzazione Raccomandazioni...")

    if recommendations:
        recommendations_df = pd.DataFrame(recommendations)
    else:
        print("Nessuna raccomandazione generata.")
        recommendations_df = pd.DataFrame() # Inizializza come DataFrame vuoto se non ci sono raccomandazioni

    output_recommendations_df_filepath = os.path.join(output_dir, f'recommendations_df_{city_name_lower}.xlsx')
    recommendations_df.to_excel(output_recommendations_df_filepath, index=False, float_format='%.2f')
    print(f"Dataset prova raccomandazioni salvato in: {output_recommendations_df_filepath}")

    output_map_filename = f"{city_name_lower}_recommendations_map.html"
    output_map_filepath = os.path.join("reports", output_map_filename)

    # Assicurati che la cartella 'reports' esista
    os.makedirs("reports", exist_ok=True)

    # La funzione plot_recommendations_on_map necessita del grafo G e delle raccomandazioni.
    plot_recommendations_on_map(G, recommendations_df, filepath=output_map_filepath)

    print(f"Mappa delle raccomandazioni generata e salvata in: {output_map_filepath}")

    print("\n--- Progetto Completato con Successo! ---")
    print(f"Controlla la cartella 'reports' per il report e le mappe interattive per '{city_name_lower}'.")


if __name__ == "__main__":
    main()