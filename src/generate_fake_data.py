import random

import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta


def generate_fake_accidents_data(nodes_gdf_full_area, pois_gdf, buffer_distance, num_accidents):
    """
    Genera dati di incidenti fittizi con una maggiore probabilità di accadere in aree con
    caratteristiche di rischio (es. scuole, negozi, attraversamenti pedonali) presenti nei nodi OSM.

    Args:
        nodes_gdf_full_area (gpd.GeoDataFrame): GeoDataFrame dei nodi OSM del grafo per l'INTERA area di studio,
                                                con le loro geometrie. Deve essere già in CRS EPSG:4326.
        pois_gdf (gpd.GeoDataFrame): GeoDataFrame dei POI, con tag rilevanti (amenity, shop, crossing, ecc.).
                                     Deve avere una colonna 'geometry' e 'original_poi_id'
                                     e deve poter essere proiettata.
        buffer_distance (int): Distanza in metri per il buffer attorno ai nodi per valutare il rischio.
        num_accidents (int): Numero totale di incidenti fittizi da generare.

    Returns:
        pd.DataFrame: DataFrame degli incidenti fittizi con Latitudine, Longitudine, Data, Ora, Gravita, Causa, Tipo Veicolo, CondizioniMeteo.
    """

    print(f"\nGenerazione di {num_accidents} incidenti fittizi...")
    #print(f"Buffer di rischio per POI: {buffer_distance}m.")

    nodes_gdf_geographic = nodes_gdf_full_area.copy()

    if nodes_gdf_geographic.empty:
        print("AVVISO: nodes_gdf_full_area è vuoto. Impossibile generare incidenti.")
        return pd.DataFrame(columns=['Latitudine', 'Longitudine', 'Data', 'Ora', 'Gravita', 'Causa', 'Tipo Veicolo'])

    #print(f"Base di {len(nodes_gdf_geographic)} nodi (incroci) reali per l'area di studio.")

    # Proietta i GeoDataFrame per i calcoli spaziali
    target_crs_proj = nodes_gdf_geographic.estimate_utm_crs()
    nodes_gdf_proj = nodes_gdf_geographic.to_crs(target_crs_proj)

    if pois_gdf is not None and not pois_gdf.empty:
        if 'original_poi_id' not in pois_gdf.columns:
            pois_gdf['original_poi_id'] = pois_gdf.index.map(str) + '_poi'

        pois_gdf_proj = pois_gdf.to_crs(target_crs_proj)
    else:
        pois_gdf_proj = gpd.GeoDataFrame(geometry=[], crs=target_crs_proj)
        print("AVVISO: pois_gdf è vuoto o None.")

    # Prepara i buffer dei nodi per l'analisi del rischio
    nodes_buffered_gdf = gpd.GeoDataFrame(
        {'geometry': nodes_gdf_proj.geometry.buffer(buffer_distance)},
        index=nodes_gdf_proj.index,
        crs=target_crs_proj
    )
    nodes_buffered_gdf.index.name = 'osmid'
    print(f"Creati {len(nodes_buffered_gdf)} buffer di {buffer_distance}m attorno ai nodi.")

    # Calcola i "punteggi di rischio" per ogni nodo filtrato
    # Inizializza un punteggio di rischio base per tutti i nodi nell'area filtrata
    node_risk_scores = pd.Series(1.0, index=nodes_gdf_proj.index, name='risk_score') # Parto da 1.0 per evitare probabilità zero

    if not pois_gdf_proj.empty and not nodes_buffered_gdf.empty:
        # Effettua un sjoin tra i POI proiettati e i buffer dei nodi
        sjoin_pois_temp = gpd.sjoin(pois_gdf_proj, nodes_buffered_gdf, how="inner", predicate="intersects")

        if not sjoin_pois_temp.empty:
            def add_risk_from_pois(poi_filter_df, weight, current_risk_scores):
                if not poi_filter_df.empty:
                    counts = poi_filter_df.groupby('osmid')['original_poi_id'].nunique()
                    return current_risk_scores.add(counts * weight, fill_value=0)
                return current_risk_scores

            WEIGHT_PED_CROSSING = 5
            WEIGHT_SCHOOL = 10
            WEIGHT_COMMERCIAL = 3

            pedestrian_related_pois = sjoin_pois_temp[
                (sjoin_pois_temp.get('crossing_ref') == 'zebra') |
                (sjoin_pois_temp.get('crossing') == 'marked') |
                (sjoin_pois_temp.get('crossing') == 'traffic_signals')
                ]
            node_risk_scores = add_risk_from_pois(pedestrian_related_pois, WEIGHT_PED_CROSSING, node_risk_scores)

            schools_pois = sjoin_pois_temp[
                (sjoin_pois_temp.get('amenity') == 'school') |
                (sjoin_pois_temp.get('amenity') == 'kindergarten')
                ]
            node_risk_scores = add_risk_from_pois(schools_pois, WEIGHT_SCHOOL, node_risk_scores)

            commercial_pois = sjoin_pois_temp[
                (sjoin_pois_temp.get('shop').notna())
            ]
            node_risk_scores = add_risk_from_pois(commercial_pois, WEIGHT_COMMERCIAL, node_risk_scores)
        else:
            print("AVVISO: Nessuna intersezione tra POI e buffer dei nodi. I punteggi di rischio rimangono al valore base.")
    else:
        print("AVVISO: pois_gdf è vuoto o nessun POI nell'area.")

    # Normalizza i punteggi di rischio per usarli come probabilità di selezione
    node_risk_scores = node_risk_scores.clip(lower=1.0)
    total_risk_score = node_risk_scores.sum()

    if total_risk_score == 0:
        probabilities = pd.Series(1.0 / len(nodes_gdf_proj), index=nodes_gdf_proj.index) if not nodes_gdf_proj.empty else pd.Series([])
        print("Avviso: Tutti i punteggi di rischio sono zero. Distribuzione incidenti uniforme nell'area.")
    else:
        probabilities = node_risk_scores / total_risk_score
        if not np.isclose(probabilities.sum(), 1.0):
            print(f"Avviso: Somma delle probabilità non 1.0: {probabilities.sum()}. Re-normalizzazione forzata.")
            probabilities = probabilities / probabilities.sum()

    # Genera gli incidenti campionando i nodi in base alle probabilità
    accidents_data_list = []

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    days_in_year = (end_date - start_date).days + 1

    SEASON_RISK_MULTIPLIERS = {
        'SPRING': {'min': 0.9, 'max': 1.0},
        'SUMMER': {'min': 0.8, 'max': 1.0},
        'AUTUMN': {'min': 1.1, 'max': 1.3},
        'WINTER': {'min': 1.2, 'max': 1.4}
    }

    WEEKDAY_RISK_MULTIPLIERS = {
        'FERIALE': {'min': 1.0, 'max': 1.1},
        'SABATO': {'min': 0.9, 'max': 1.0},
        'DOMENICA': {'min': 0.7, 'max': 0.9}
    }

    daily_risk_factors = np.ones(days_in_year)

    for day_offset in range(days_in_year):
        current_date = start_date + timedelta(days=day_offset)
        month = current_date.month
        day_of_week = current_date.weekday()

        if month in [3, 4, 5]:
            season_factor = np.random.uniform(SEASON_RISK_MULTIPLIERS['SPRING']['min'], SEASON_RISK_MULTIPLIERS['SPRING']['max'])
        elif month in [6, 7, 8]:
            season_factor = np.random.uniform(SEASON_RISK_MULTIPLIERS['SUMMER']['min'], SEASON_RISK_MULTIPLIERS['SUMMER']['max'])
        elif month in [9, 10, 11]:
            season_factor = np.random.uniform(SEASON_RISK_MULTIPLIERS['AUTUMN']['min'], SEASON_RISK_MULTIPLIERS['AUTUMN']['max'])
        else:
            season_factor = np.random.uniform(SEASON_RISK_MULTIPLIERS['WINTER']['min'], SEASON_RISK_MULTIPLIERS['WINTER']['max'])

        daily_risk_factors[day_offset] *= season_factor

        if day_of_week < 5:
            weekly_factor = np.random.uniform(WEEKDAY_RISK_MULTIPLIERS['FERIALE']['min'], WEEKDAY_RISK_MULTIPLIERS['FERIALE']['max'])
        elif day_of_week == 5:
            weekly_factor = np.random.uniform(WEEKDAY_RISK_MULTIPLIERS['SABATO']['min'], WEEKDAY_RISK_MULTIPLIERS['SABATO']['max'])
        else:
            weekly_factor = np.random.uniform(WEEKDAY_RISK_MULTIPLIERS['DOMENICA']['min'], WEEKDAY_RISK_MULTIPLIERS['DOMENICA']['max'])

        daily_risk_factors[day_offset] *= weekly_factor

    if daily_risk_factors.sum() == 0:
        daily_accident_probabilities = np.ones(days_in_year) / days_in_year
    else:
        daily_accident_probabilities = daily_risk_factors / daily_risk_factors.sum()

    # Campiona i nodi in base alle probabilità di rischio all'interno dell'area filtrata
    if not probabilities.empty and len(nodes_gdf_proj) > 0:
        chosen_day_offsets = random.choices(
            population=range(days_in_year),
            weights=daily_accident_probabilities.tolist(),
            k=num_accidents
        )
        fake_timestamps = [start_date + timedelta(days=offset) for offset in chosen_day_offsets]

        # Campiona gli OSMID dei nodi (cioè le posizioni)
        sampled_nodes_osmid = random.choices(
            population=nodes_gdf_proj.index.tolist(),
            weights=probabilities.tolist(),
            k=num_accidents
        )
    else:
        print("Nessuna probabilità calcolata o nessun nodo disponibile per la generazione di incidenti.")
        return pd.DataFrame(columns=['Latitudine', 'Longitudine', 'Data', 'Ora', 'Gravita', 'Causa', 'Tipo Veicolo'])

    # Riproeitta le geometrie dei nodi campionati in massa
    sampled_geometries_proj = nodes_gdf_proj.loc[sampled_nodes_osmid].geometry

    # Riproeitta l'intera GeoSeries in EPSG:4326 in un'unica operazione
    sampled_geometries_4326 = sampled_geometries_proj.to_crs("EPSG:4326")

    # Genera gli attributi non spaziali degli incidenti (gravità, causa, tipo veicolo)
    # in base al numero totale di incidenti
    causes = ['eccesso_velocita', 'distrazione', 'mancato_rispetto_stop', 'mancanza_precedenza', 'condizioni_stradali', 'guida_sotto_effetto_sostanze']
    vehicle_types = ['auto', 'moto', 'bicicletta', 'camion', 'furgone']

    # Popola la lista degli incidenti con i dati e le coordinate dei nodi campionati
    for i, incident_point_geom_4326 in enumerate(sampled_geometries_4326):
        current_date = fake_timestamps[i]
        accident_hour = np.random.randint(0, 24)
        accident_minute = np.random.randint(0, 60)
        accident_time = f"{accident_hour:02d}:{accident_minute:02d}"

        # Modulazione della Gravità e Condizioni Meteo in base a Ora/Stagione
        p_severity_1, p_severity_2, p_severity_3, p_severity_4 = 0.7, 0.2, 0.08, 0.02

        if 22 <= accident_hour or accident_hour <= 6:
            p_severity_1, p_severity_2, p_severity_3, p_severity_4 = 0.4, 0.35, 0.2, 0.05
        elif 7 <= accident_hour <= 9 or 17 <= accident_hour <= 19:
            p_severity_1, p_severity_2, p_severity_3, p_severity_4 = 0.6, 0.25, 0.1, 0.05

        month = current_date.month
        weather_condition = random.choice(['Sereno', 'Pioggia', 'Nebbia', 'Neve'])
        if month in [12, 1, 2]:
            p_severity_1, p_severity_2, p_severity_3, p_severity_4 = p_severity_1 * 0.8, p_severity_2 * 1.1, p_severity_3 * 1.2, p_severity_4 * 1.5
            weather_condition = random.choices(['Sereno', 'Pioggia', 'Nebbia', 'Neve'], weights=[0.4, 0.3, 0.2, 0.1], k=1)[0]
        elif month in [9, 10, 11]:
            p_severity_1, p_severity_2, p_severity_3, p_severity_4 = p_severity_1 * 0.9, p_severity_2 * 1.05, p_severity_3 * 1.15, p_severity_4 * 1.2
            weather_condition = random.choices(['Sereno', 'Pioggia', 'Nebbia'], weights=[0.6, 0.3, 0.1], k=1)[0]

        total_p = p_severity_1 + p_severity_2 + p_severity_3 + p_severity_4
        if total_p == 0:
            p_severity_1, p_severity_2, p_severity_3, p_severity_4 = 0.7, 0.2, 0.08, 0.02
        else:
            p_severity_1 /= total_p
            p_severity_2 /= total_p
            p_severity_3 /= total_p
            p_severity_4 /= total_p

        severity = random.choices([1, 2, 3, 4], weights=[p_severity_1, p_severity_2, p_severity_3, p_severity_4], k=1)[0]

        cause = random.choice(causes)
        vehicle_type = random.choice(vehicle_types)

        accidents_data_list.append({
            'Latitudine': incident_point_geom_4326.y,
            'Longitudine': incident_point_geom_4326.x,
            'Data': current_date.strftime('%d/%m/%Y'),
            'Ora': accident_time,
            'Gravita': severity,
            'Causa': cause,
            'Tipo Veicolo': vehicle_type,
            'CondizioniMeteo': weather_condition
        })

    if not accidents_data_list:
        print("Nessun incidente fittizio generato. Controlla i parametri, i dati in input e l'area geografica.")
        # Ho aggiunto 'CondizioniMeteo' alle colonne di default per coerenza con l'output effettivo.
        return pd.DataFrame(columns=['Latitudine', 'Longitudine', 'Data', 'Ora', 'Gravita', 'Causa', 'Tipo Veicolo', 'CondizioniMeteo'])

    df_incidents = pd.DataFrame(accidents_data_list)
    print(f"Generati {len(df_incidents)} incidenti fittizi.")
    return df_incidents


def generate_fake_traffic_data(G, nodes_gdf_full_area, pois_gdf, city_name, num_sensors, num_readings_per_sensor,
                               buffer_distance_poi=100,
                               road_type_weights=None, # Pesi per i tipi di strada
                               ):
    """
    Genera dati di traffico fittizi più realistici per una data città,
    posizionando sensori su nodi OSM e modulando il traffico in base a:
    - Gerarchia delle strade
    - Densità di POI nell'area circostante.

    Args:
        G (networkx.MultiDiGraph): Il grafo OSM della città, completo.
        nodes_gdf_full_area (gpd.GeoDataFrame): GeoDataFrame di *tutti* i nodi OSM del grafo per l'area di studio.
                                            Deve essere già in un CRS geografico (EPSG:4326).
        pois_gdf (gpd.GeoDataFrame): GeoDataFrame di *tutti* i POI della città.
                                     Deve essere già in un CRS geografico (EPSG:4326).
        city_name (str): Nome della città (per gli ID dei sensori).
        num_sensors (int): Numero di sensori di traffico da simulare.
        num_readings_per_sensor (int): Numero di letture da generare per ciascun sensore.
        buffer_distance_poi (int): Distanza in metri per il buffer attorno ai nodi per valutare l'influenza dei POI.
        road_type_weights (dict, optional): Dizionario che mappa i tipi di strada OSM (es. 'motorway') a un peso numerico.
                                            I tipi di strada non specificati avranno peso 1.0.
                                            Esempio: {'motorway': 5, 'primary': 3, 'secondary': 2}.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame dei dati di traffico fittizi.
    """

    print(f"\nGenerazione di {num_sensors} sensori di traffico con {num_readings_per_sensor} letture ciascuno...")
    #print(f"Buffer di influenza POI: {buffer_distance_poi}m.")

    # Pesi di default per i tipi di strada se non forniti
    if road_type_weights is None:
        road_type_weights = {
            'motorway': 2.0,
            'trunk': 1.8,
            'primary': 4.0,
            'secondary': 3.0,
            'tertiary': 2.0,
            'residential': 1.0,
            'unclassified': 0.8,
            'service': 0.5,
            'footway': 0.01,
            'pedestrian': 0.01,
            'path': 0.01,
            'track': 0.1
        }

    # Pesi per i POI (simili a quelli usati per gli incidenti ma potenzialmente diversi)
    POI_WEIGHTS = {
        'school': 5, 'kindergarten': 4, 'university': 3, 'college': 2,
        'restaurant': 2, 'cafe': 1.5, 'bar': 1, 'fast_food': 1.5,
        'shop': 2.5, 'commercial': 3, 'retail': 2.5, 'mall': 4,
        'office': 3.5, 'industrial': 2, 'station': 3, 'bus_station': 1.5,
        'public_transport': 1.5, 'hospital': 3.5, 'clinic': 2, 'parking': 1.5
    }

    # Utilizza tutti i nodi OSM e POI dell'area di interesse geografica (non più filtrati da box ristretto)
    # Assicurati che nodes_gdf_full_area sia già in EPSG:4326.
    nodes_gdf_geographic = nodes_gdf_full_area.copy()

    # Usa tutti i nodi dell'area completa
    nodes_in_area_gdf = nodes_gdf_geographic.copy()

    if nodes_in_area_gdf.empty:
        print("AVVISO: nodes_gdf_full_area è vuoto. Impossibile generare dati di traffico.")
        return gpd.GeoDataFrame(columns=['IDSensore', 'Latitudine', 'Longitudine', 'DataRilevamento',
                                         'OraRilevamento', 'ConteggioVeicoli', 'VelocitaMedia', 'IndiceCongestione', 'geometry'], crs="EPSG:4326")
    #print(f"Base di {len(nodes_in_area_gdf)} nodi (incroci) reali per l'area di studio del traffico.")

    if pois_gdf is not None and not pois_gdf.empty:
        pois_gdf_geographic = pois_gdf.to_crs("EPSG:4326") if pois_gdf.crs is None or not pois_gdf.crs.is_geographic else pois_gdf.copy()

        # Uso tutti i POI nell'area
        pois_gdf_filtered_geographic = pois_gdf_geographic.copy()

        if 'original_poi_id' not in pois_gdf_filtered_geographic.columns:
            if isinstance(pois_gdf_filtered_geographic.index, pd.MultiIndex):
                pois_gdf_filtered_geographic['original_poi_id'] = pois_gdf_filtered_geographic.index.map(str) + '_poi'
            else:
                pois_gdf_filtered_geographic['original_poi_id'] = pois_gdf_filtered_geographic.index.astype(str) + '_poi'
    else:
        pois_gdf_filtered_geographic = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        print("AVVISO: pois_gdf è vuoto o None. Nessun fattore di traffico basato sui POI verrà considerato.")

    # Proietta i GeoDataFrame per i calcoli spaziali
    # Stima il CRS proiettato dall'intera area per coerenza.
    target_crs_proj = nodes_gdf_geographic.estimate_utm_crs() # Usa nodes_gdf_geographic (l'intera area)
    nodes_gdf_proj = nodes_gdf_geographic.to_crs(target_crs_proj) # Proietta tutti i nodi

    pois_gdf_proj = pois_gdf_filtered_geographic.to_crs(target_crs_proj) # Proietta tutti i POI

    # Crea buffer dei nodi per l'analisi POI
    nodes_buffered_gdf = gpd.GeoDataFrame(
        {'geometry': nodes_gdf_proj.geometry.buffer(buffer_distance_poi)},
        index=nodes_gdf_proj.index,
        crs=target_crs_proj
    )
    nodes_buffered_gdf.index.name = 'osmid'

    # Calcola i "punteggi di potenziale traffico" per ogni nodo
    node_traffic_potential = pd.Series(1.0, index=nodes_gdf_proj.index, name='traffic_potential')

    # Aggiungo il contributo della gerarchia stradale
    for osmid in nodes_gdf_proj.index:
        highways_touching_node = []
        for neighbor in list(G.neighbors(osmid)) + list(G.predecessors(osmid)):
            if (osmid, neighbor) in G.edges:
                edge_data = G.get_edge_data(osmid, neighbor)
                for k_edge in edge_data:
                    if 'highway' in edge_data[k_edge]:
                        highway_type = edge_data[k_edge]['highway']
                        if isinstance(highway_type, list):
                            highways_touching_node.extend(highway_type)
                        else:
                            highways_touching_node.append(highway_type)
            if (neighbor, osmid) in G.edges:
                edge_data = G.get_edge_data(neighbor, osmid)
                for k_edge in edge_data:
                    if 'highway' in edge_data[k_edge]:
                        highway_type = edge_data[k_edge]['highway']
                        if isinstance(highway_type, list):
                            highways_touching_node.extend(highway_type)
                        else:
                            highways_touching_node.append(highway_type)

        if highways_touching_node:
            max_road_weight = max([road_type_weights.get(ht, 1.0) for ht in highways_touching_node])
            node_traffic_potential.loc[osmid] += max_road_weight * 0.4

    # Aggiungo il contributo dei POI
    if not pois_gdf_proj.empty and not nodes_buffered_gdf.empty:
        sjoin_pois_traffic = gpd.sjoin(pois_gdf_proj, nodes_buffered_gdf, how="inner", predicate="intersects")

        if not sjoin_pois_traffic.empty:
            def add_traffic_potential_from_pois(poi_filter_df, weights_map, current_potential_scores):
                if poi_filter_df.empty:
                    return current_potential_scores

                poi_contributions = []
                for idx, row in poi_filter_df.iterrows():
                    poi_type_found = None
                    for tag_key, tag_value in row.items():
                        if tag_key in ['amenity', 'shop', 'highway', 'crossing', 'office', 'industrial', 'public_transport'] and tag_value in weights_map:
                            poi_type_found = tag_value
                            break
                        elif tag_key in ['retail'] and isinstance(tag_value, str) and tag_value in weights_map:
                            poi_type_found = tag_value
                            break
                        elif tag_key == 'shop' and tag_value is True and 'shop' in weights_map:
                            poi_type_found = 'shop'
                            break

                    if poi_type_found:
                        poi_contributions.append({
                            'osmid': row['osmid'],
                            'contribution': weights_map[poi_type_found]
                        })

                if poi_contributions:
                    poi_contributions_df = pd.DataFrame(poi_contributions)
                    grouped_contributions = poi_contributions_df.groupby('osmid')['contribution'].sum()
                    return current_potential_scores.add(grouped_contributions, fill_value=0)
                return current_potential_scores

            relevant_pois_in_sjoin = sjoin_pois_traffic[
                sjoin_pois_traffic.apply(lambda row: any(tag in POI_WEIGHTS for tag in [row.get('amenity'), row.get('shop'), row.get('highway'), row.get('crossing'), row.get('office'), row.get('industrial'), row.get('public_transport')]), axis=1) |
                sjoin_pois_traffic['shop'].notna()
                ]
            node_traffic_potential = add_traffic_potential_from_pois(relevant_pois_in_sjoin, POI_WEIGHTS, node_traffic_potential)
        else:
            print("AVVISO: Nessuna intersezione tra POI e buffer dei nodi. I punteggi di traffico dai POI rimangono al valore base.")
    else:
        print("AVVISO: pois_gdf è vuoto o nessun POI nell'area, quindi nessun fattore di traffico basato sui POI verrà considerato.")

    # Normalizza i punteggi di traffico per usarli come probabilità di selezione dei sensori
    node_traffic_potential = node_traffic_potential.clip(lower=1.0)
    total_potential_score = node_traffic_potential.sum()
    if total_potential_score == 0:
        sensor_probabilities = pd.Series(1.0 / len(nodes_gdf_proj), index=nodes_gdf_proj.index)
        print("AVVISO: Tutti i punteggi di potenziale traffico sono zero. Sensori distribuiti uniformemente.")
    else:
        sensor_probabilities = node_traffic_potential / total_potential_score

    # Seleziona i nodi per i sensori in base alle probabilità di potenziale traffico
    if not sensor_probabilities.empty and len(nodes_gdf_proj) > 0:
        sampled_sensor_nodes_osmid = random.choices(
            population=nodes_gdf_proj.index.tolist(),
            weights=sensor_probabilities.tolist(),
            k=num_sensors
        )
    else:
        print("AVVISO: Nessuna probabilità calcolata o nessun nodo disponibile per il posizionamento dei sensori.")
        return gpd.GeoDataFrame(columns=['IDSensore', 'Latitudine', 'Longitudine', 'DataRilevamento',
                                         'OraRilevamento', 'ConteggioVeicoli', 'VelocitaMedia', 'IndiceCongestione', 'geometry'], crs="EPSG:4326")

    # Mappa gli OSMID dei sensori ai loro punteggi di traffico effettivi
    sensor_node_potential_map = node_traffic_potential.loc[sampled_sensor_nodes_osmid].to_dict()

    # Recupera le geometrie dei nodi dei sensori
    sensor_node_geometries_proj = nodes_gdf_proj.loc[sampled_sensor_nodes_osmid].geometry
    sensor_node_geometries_4326 = sensor_node_geometries_proj.to_crs("EPSG:4326")

    # Crea una lista di tuple (osmid, lat, lon) per i sensori scelti
    chosen_sensors_info = []
    for i, osmid in enumerate(sampled_sensor_nodes_osmid):
        geom_4326 = sensor_node_geometries_4326.iloc[i]
        chosen_sensors_info.append({
            'osmid': osmid,
            'lat': geom_4326.y,
            'lon': geom_4326.x,
            'traffic_potential': sensor_node_potential_map.get(osmid, 1.0)
        })

    # Assegna ID sensore univoci (non basati sull'OSMID, ma un ID "fisico" del sensore)
    for i, sensor_info in enumerate(chosen_sensors_info):
        sensor_info['IDSensore'] = f"SENSOR_{city_name.upper()}_{i:03d}"

    # Genera le letture di traffico per ciascun sensore
    traffic_data = []
    start_date = datetime(2024, 1, 1)

    for sensor_info in chosen_sensors_info:
        sensor_id = sensor_info['IDSensore']
        sensor_lat = sensor_info['lat']
        sensor_lon = sensor_info['lon']
        sensor_potential = sensor_info['traffic_potential']

        potential_multiplier = 1 + (sensor_potential - node_traffic_potential.min()) / (node_traffic_potential.max() - node_traffic_potential.min() + 1e-6) * 0.5

        for reading_idx in range(num_readings_per_sensor):
            random_days = np.random.randint(0, 365)
            reading_date = start_date + timedelta(days=int(random_days))
            reading_hour = np.random.randint(0, 24)
            reading_minute = np.random.randint(0, 60)
            reading_time = f"{reading_hour:02d}:{reading_minute:02d}"

            day_of_week = reading_date.weekday()
            weekly_multiplier = 1.0

            if day_of_week < 5:
                weekly_multiplier *= np.random.uniform(0.95, 1.05)
            elif day_of_week == 5:
                weekly_multiplier *= np.random.uniform(0.7, 0.95)
            else:
                weekly_multiplier *= np.random.uniform(0.4, 0.7)

            month = reading_date.month
            season_multiplier = 1.0

            if month in [7, 8]:
                season_multiplier *= np.random.uniform(0.8, 1.0)
            elif month in [12]:
                season_multiplier *= np.random.uniform(1.0, 1.2)
            elif month in [1, 2]:
                season_multiplier *= np.random.uniform(0.9, 1.05)
            elif month in [3, 4, 5, 6, 9, 10, 11]:
                season_multiplier *= np.random.uniform(0.98, 1.02)

            count_base = np.random.randint(30, 150) * potential_multiplier
            count_base *= season_multiplier
            count_base *= weekly_multiplier

            if 7 <= reading_hour <= 9 or 17 <= reading_hour <= 19:
                count_base = count_base * np.random.uniform(1.3, 1.8)
            elif 23 <= reading_hour or reading_hour <= 5:
                count_base = count_base * np.random.uniform(0.4, 0.8)

            count_base = max(1, int(count_base))

            speed_base = np.random.uniform(20, 70)
            speed = speed_base / (1 + (count_base / 500))
            congestion_index = 1 - (speed / 70) + (count_base / 1000) * 0.5
            congestion_index = max(0.0, min(1.0, congestion_index))

            traffic_data.append({
                'IDSensore': sensor_id,
                'Latitudine': sensor_lat,
                'Longitudine': sensor_lon,
                'DataRilevamento': reading_date.strftime('%d/%m/%Y'),
                'OraRilevamento': reading_time,
                'ConteggioVeicoli': int(count_base),
                'VelocitaMedia': round(speed, 2),
                'IndiceCongestione': round(congestion_index, 2)
            })

    if not traffic_data:
        print("AVVISO: Nessun dato di traffico generato. Controlla i parametri e l'area geografica.")
        return gpd.GeoDataFrame(columns=['IDSensore', 'Latitudine', 'Longitudine', 'DataRilevamento',
                                         'OraRilevamento', 'ConteggioVeicoli', 'VelocitaMedia', 'IndiceCongestione', 'geometry'], crs="EPSG:4326")

    df = pd.DataFrame(traffic_data)
    geometry = gpd.points_from_xy(df['Longitudine'], df['Latitudine'])
    gdf_traffic_fake = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    print(f"Generati {len(gdf_traffic_fake)} punti di traffico fittizi.")
    return gdf_traffic_fake
