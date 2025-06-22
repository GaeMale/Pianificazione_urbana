#import folium
#import os
#import pandas as pd
#import geopandas as gpd
#import numpy as np # Importa numpy per gestire valori numerici
#
# def plot_osm_data_on_map(nodes_gdf, G, filepath="reports/osm_data_map.html"):
#     """
#     Genera una mappa interattiva dei dati OSM, inclusi i nodi e gli archi.
#     I nodi possono essere colorati in base a una feature (es. densita_popolazione_vicina).
#
#     Args:
#         nodes_gdf (gpd.GeoDataFrame): GeoDataFrame dei nodi con 'geometry' e features integrate.
#         G (networkx.MultiDiGraph): Grafo della rete stradale OSM.
#         filepath (str): Percorso dove salvare lappa HTML.
#     """
#     print(f"Generazione mappa dati OSM in {filepath}...")
#     if nodes_gdf.empty:
#         print("Nodes GeoDataFrame è vuoto. Impossibile generare la mappa dei dati OSM.")
#         # Creare una mappa vuota o con un messaggio di avviso
#         m = folium.Map(location=[41.12, 16.87], zoom_start=12) # Posizione di default (es. Bari)
#         folium.Marker(
#             location=[41.12, 16.87],
#             popup="Nessun dato nodo disponibile per la visualizzazione."
#         ).add_to(m)
#         m.save(filepath)
#         return
#
#     # Calcola il centro della mappa
#     center_lat = nodes_gdf.geometry.y.mean() if not nodes_gdf.empty else 41.12
#     center_lon = nodes_gdf.geometry.x.mean() if not nodes_gdf.empty else 16.87
#
#     m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
#
#     # Aggiungi gli archi del grafo
#     ox.plot_graph_folium(G, graph_map=m, color="#808080", weight=1, opacity=0.5, tiles="OpenStreetMap")
#
#     # Aggiungi i nodi. Potremmo colorarli in base a una feature (es. densità popolazione)
#     if 'densita_popolazione_vicina' in nodes_gdf.columns:
#         # Normalizza la densità per la colorazione
#         # Sostituisci NaN con 0 prima di calcolare max()
#         nodes_gdf['densita_popolazione_vicina_clean'] = nodes_gdf['densita_popolazione_vicina'].fillna(0)
#         max_density = nodes_gdf['densita_popolazione_vicina_clean'].max()
#
#         if max_density > 0:
#             nodes_gdf['color_intensity'] = nodes_gdf['densita_popolazione_vicina_clean'] / max_density
#         else:
#             nodes_gdf['color_intensity'] = 0.1 # Colore di base se la densità è 0 per tutti
#
#         for _, row in nodes_gdf.iterrows():
#             if row.geometry:
#                 color = "red" if row['color_intensity'] > 0.7 else ("orange" if row['color_intensity'] > 0.3 else "green")
#                 folium.CircleMarker(
#                     location=[row.geometry.y, row.geometry.x],
#                     radius=3,
#                     color=color,
#                     fill=True,
#                     fill_color=color,
#                     fill_opacity=0.7,
#                     popup=f"OSMID: {row.name}<br>Density: {row['densita_popolazione_vicina']:.0f}"
#                 ).add_to(m)
#     else:
#         # Colore default se la feature non è presente
#         for _, row in nodes_gdf.iterrows():
#             if row.geometry:
#                 folium.CircleMarker(
#                     location=[row.geometry.y, row.geometry.x],
#                     radius=2,
#                     color="blue",
#                     fill=True,
#                     fill_color="blue",
#                     fill_opacity=0.5,
#                     popup=f"OSMID: {row.name}"
#                 ).add_to(m)
#
#     m.save(filepath)
#     print(f"Mappa dati OSM salvata in {filepath}.")
#
# def plot_recommendations_on_map(map_data_df, G, filepath="reports/recommendations_map.html", default_color='gray'):
#     """
#     Genera una mappa interattiva delle raccomandazioni di sicurezza, colorando i nodi.
#
#     Args:
#         map_data_df (gpd.GeoDataFrame): GeoDataFrame dei nodi con 'geometry', 'danger_score' e 'recommendation'.
#         G (networkx.MultiDiGraph): Grafo della rete stradale OSM.
#         filepath (str): Percorso dove salvare la mappa HTML.
#         default_color (str): Colore per i nodi senza danger_score o raccomandazioni.
#     """
#     print(f"Generazione mappa raccomandazioni in {filepath}...")
#     if map_data_df.empty or 'danger_score' not in map_data_df.columns:
#         print("Map data GeoDataFrame è vuoto o manca 'danger_score'. Impossibile generare la mappa delle raccomandazioni.")
#         # Creare una mappa vuota o con un messaggio di avviso
#         m = folium.Map(location=[41.12, 16.87], zoom_start=12)
#         folium.Marker(
#             location=[41.12, 16.87],
#             popup="Nessun dato di raccomandazione disponibile per la visualizzazione."
#         ).add_to(m)
#         m.save(filepath)
#         return
#
#     # Calcola il centro della mappa
#     center_lat = map_data_df.geometry.y.mean() if not map_data_df.empty else 41.12
#     center_lon = map_data_df.geometry.x.mean() if not map_data_df.empty else 16.87
#
#     m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
#
#     # Aggiungi gli archi del grafo
#     ox.plot_graph_folium(G, graph_map=m, color="#808080", weight=1, opacity=0.5, tiles="OpenStreetMap")
#
#     # Aggiungi i nodi colorati in base al 'danger_score'
#     # Sostituisci NaN con 0 prima di calcolare max()
#     map_data_df['danger_score_clean'] = map_data_df['danger_score'].fillna(0)
#     max_score = map_data_df['danger_score_clean'].max()
#     if max_score == 0: # Evita divisione per zero se tutti i danger_score sono 0
#         max_score = 1
#
#     for _, row in map_data_df.iterrows():
#         if row.geometry:
#             score = row['danger_score']
#             # Mappa il punteggio di pericolo a un colore (es. da verde a rosso)
#             if pd.isna(score) or score == 0:
#                 color = default_color # Grigio per nessun pericolo o non calcolato
#             else:
#                 # Esempio di scala di colore: verde (basso) -> giallo (medio) -> rosso (alto)
#                 normalized_score = score / max_score
#                 if normalized_score < 0.33:
#                     color = "green"
#                 elif normalized_score < 0.66:
#                     color = "orange"
#                 else:
#                     color = "red"
#
#             popup_text = f"OSMID: {row.name}<br>Danger Score: {score:.2f}"
#             if 'recommendation' in row and pd.notna(row['recommendation']):
#                 popup_text += f"<br>Recommendation: {row['recommendation']}"
#
#             folium.CircleMarker(
#                 location=[row.geometry.y, row.geometry.x],
#                 radius=5,
#                 color=color,
#                 fill=True,
#                 fill_color=color,
#                 fill_opacity=0.7,
#                 popup=popup_text
#             ).add_to(m)
#
#     m.save(filepath)
#     print(f"Mappa raccomandazioni salvata in {filepath}.")
#
#
# def generate_html_report(unified_features_df, report_filepath, map_osm_filepath, map_recs_filepath):
#     """
#     Genera un report HTML riassuntivo con statistiche chiave e link alle mappe.
#
#     Args:
#         unified_features_df (pd.DataFrame): DataFrame unificato con tutte le features.
#         report_filepath (str): Percorso dove salvare il report HTML.
#         map_osm_filepath (str): Percorso del file HTML della mappa dati OSM.
#         map_recs_filepath (str): Percorso del file HTML della mappa delle raccomandazioni.
#     """
#     print(f"Generazione report HTML in {report_filepath}...")
#
#     # Assicurati che i percorsi delle mappe siano relativi o accessibili correttamente
#     # Usiamo os.path.basename per ottenere solo il nome del file se i report sono nella stessa cartella
#     osm_map_name = os.path.basename(map_osm_filepath)
#     recs_map_name = os.path.basename(map_recs_filepath)
#
#     # --- Generazione di statistiche riassuntive ---
#     num_nodes = len(unified_features_df)
#
#     # Tentativo di recuperare il nome della città dal contesto, altrimenti usa 'la città selezionata'
#     city_name = "la città selezionata"
#     # Una soluzione più robusta sarebbe passare city_name a questa funzione da main.py.
#     # Per ora, usiamo una stringa generica.
#
#     report_title = f"Report di Analisi della Sicurezza Stradale - {city_name.replace('_', ' ').title()}"
#
#     summary_html = "<h2>Riepilogo Generale</h2>"
#     summary_html += f"<p>Questo report presenta un'analisi della sicurezza stradale per **{city_name}**, basata sull'integrazione di dati OpenStreetMap, POI, dati demografici, incidenti stradali e dati sul traffico.</p>"
#     summary_html += f"<p>Il dataset unificato contiene informazioni per <b>{num_nodes}</b> nodi (incroci o punti importanti della rete stradale).</p>"
#
#     # Statistiche sugli incidenti (se disponibili)
#     if 'num_incidenti' in unified_features_df.columns:
#         total_accidents = unified_features_df['num_incidenti'].sum()
#         avg_accidents_per_node = unified_features_df['num_incidenti'].mean()
#         nodes_with_accidents = (unified_features_df['num_incidenti'] > 0).sum()
#
#         summary_html += "<h3>Dati Incidenti</h3>"
#         summary_html += f"<p>Numero totale di incidenti registrati nei nodi analizzati: <b>{int(total_accidents)}</b></p>"
#         summary_html += f"<p>Numero medio di incidenti per nodo: <b>{avg_accidents_per_node:.2f}</b></p>"
#         summary_html += f"<p>Nodi con almeno un incidente registrato: <b>{nodes_with_accidents}</b> su {num_nodes}</p>"
#
#         if 'gravita_media_incidente' in unified_features_df.columns:
#             # Assicurati che sia numerico e gestisci NaN
#             avg_severity = unified_features_df['gravita_media_incidente'].dropna().mean()
#             if not pd.isna(avg_severity):
#                 summary_html += f"<p>Gravità media degli incidenti (scala arbitraria): <b>{avg_severity:.2f}</b></p>"
#             else:
#                 summary_html += "<p>Gravità media degli incidenti: Dati non disponibili.</p>"
#
#     # Statistiche su densità popolazione (se disponibili)
#     if 'densita_popolazione_vicina' in unified_features_df.columns:
#         # Assicurati che sia numerico e gestisci NaN
#         avg_density = unified_features_df['densita_popolazione_vicina'].dropna().mean()
#         if not pd.isna(avg_density):
#             summary_html += "<h3>Dati Demografici</h3>"
#             summary_html += f"<p>Densità media di popolazione vicino ai nodi: <b>{avg_density:.2f}</b> persone/km² (valore stimato/fallback)</p>"
#         else:
#             summary_html += "<p>Densità di popolazione: Dati non disponibili.</p>"
#
#     # Statistiche sui POI (se disponibili)
#     # Controlla l'esistenza di ALMENO UNA colonna POI prima di stampare il titolo
#     if any(col in unified_features_df.columns for col in ['num_scuole_vicine', 'num_ospedali_vicine', 'num_stazioni_bus_vicine']):
#         summary_html += "<h3>Punti di Interesse (POI)</h3>"
#         if 'num_scuole_vicine' in unified_features_df.columns:
#             total_schools = unified_features_df['num_scuole_vicine'].sum()
#             summary_html += f"<p>Numero totale di scuole vicine ai nodi: <b>{int(total_schools)}</b></p>"
#         if 'num_ospedali_vicine' in unified_features_df.columns:
#             total_hospitals = unified_features_df['num_ospedali_vicine'].sum()
#             summary_html += f"<p>Numero totale di ospedali vicini ai nodi: <b>{int(total_hospitals)}</b></p>"
#         if 'num_stazioni_bus_vicine' in unified_features_df.columns:
#             total_bus_stops = unified_features_df['num_stazioni_bus_vicine'].sum()
#             summary_html += f"<p>Numero totale di stazioni bus vicine ai nodi: <b>{int(total_bus_stops)}</b></p>"
#
#     # Statistiche sul traffico (se disponibili)
#     # Controlla l'esistenza di ALMENO UNA colonna traffico prima di stampare il titolo
#     if any(col in unified_features_df.columns for col in ['avg_conteggio_veicoli', 'avg_velocita', 'indice_congestione']):
#         summary_html += "<h3>Dati Traffico</h3>"
#         if 'avg_conteggio_veicoli' in unified_features_df.columns:
#             avg_traffic_count = unified_features_df['avg_conteggio_veicoli'].dropna().mean()
#             if not pd.isna(avg_traffic_count):
#                 summary_html += f"<p>Conteggio medio veicoli per nodo: <b>{avg_traffic_count:.2f}</b></p>"
#         if 'avg_velocita' in unified_features_df.columns:
#             avg_speed = unified_features_df['avg_velocita'].dropna().mean()
#             if not pd.isna(avg_speed):
#                 summary_html += f"<p>Velocità media per nodo: <b>{avg_speed:.2f}</b> km/h</p>"
#         if 'indice_congestione' in unified_features_df.columns:
#             avg_congestion = unified_features_df['indice_congestione'].dropna().mean()
#             if not pd.isna(avg_congestion):
#                 summary_html += f"<p>Indice di congestione medio per nodo: <b>{avg_congestion:.2f}</b></p>"
#
#
#     # --- Contenuto del report HTML ---
#     html_content = f"""
#     <!DOCTYPE html>
#     <html lang="it">
#     <head>
#         <meta charset="UTF-8">
#         <meta name="viewport" content="width=device-width, initial-scale=1.0">
#         <title>{report_title}</title>
#         <style>
#             body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
#             .container {{ max-width: 900px; margin: auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
#             h1, h2, h3 {{ color: #0056b3; }}
#             .map-link {{ display: inline-block; margin: 10px 0; padding: 10px 15px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; }}
#             .map-link:hover {{ background-color: #0056b3; }}
#             .section {{ margin-bottom: 20px; padding: 15px; background-color: #e9ecef; border-left: 5px solid #007bff; border-radius: 4px; }}
#             table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
#             th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
#             th {{ background-color: #f2f2f2; }}
#         </style>
#     </head>
#     <body>
#         <div class="container">
#             <h1>{report_title}</h1>
#             <p>Data di generazione: {pd.Timestamp.now().strftime('%d-%m-%Y %H:%M:%S')}</p>
#
#             {summary_html}
#
#             <div class="section">
#                 <h2>Mappe Interattive</h2>
#                 <p>Esplora i dati e le raccomandazioni sulle mappe interattive:</p>
#                 <a href="{os.path.basename(map_osm_filepath)}" class="map-link">Visualizza Mappa Dati OSM</a>
#                 <a href="{os.path.basename(map_recs_filepath)}" class="map-link">Visualizza Mappa Raccomandazioni</a>
#             </div>
#
#             <div class="section">
#                 <h2>Note Metodologiche</h2>
#                 <p>
#                     Questo report è generato da un'analisi della sicurezza stradale che integra diverse fonti di dati.
#                     I dati spaziali sono stati acquisiti da OpenStreetMap (OSM).
#                     Le feature sono state generate per i nodi (incroci o punti importanti) della rete stradale.
#                     I dati demografici, POI, incidenti e traffico sono stati integrati spazialmente con i nodi OSM.
#                     Per i dati demografici, se non disponibili a livello granulare, è stata utilizzata una densità di popolazione media stimata per l'intera città.
#                     Il modello predittivo è stato addestrato per identificare le aree a rischio.
#                 </p>
#             </div>
#
#             <div class="section">
#                 <h2>Dettagli Dataset Unificato (Prime 5 Righe)</h2>
#                 {unified_features_df.head().to_html()}
#             </div>
#
#         </div>
#     </body>
#     </html>
#     """
#
#     # Salva il report
#     with open(report_filepath, "w", encoding="utf-8") as f:
#         f.write(html_content)
#
#     print(f"Report HTML generato e salvato in {report_filepath}.")

# src/visualization.py
import os

import folium
from folium.plugins import MarkerCluster # Per gestire molti POI
import osmnx as ox
import pandas as pd
import geopandas as gpd
import webbrowser

def plot_recommendations_on_map(G, recommendations_df, filepath="reports/recommendations_map.html"):
    """
    Crea una mappa interattiva con le raccomandazioni evidenziate.
    """
    if G is None or recommendations_df.empty:
        print("Impossibile generare la mappa delle raccomandazioni: dati mancanti o vuoti.")
        return

    priority_colors = {
        "Priorità Alta": "red",
        "Priorità Media": "orange",
        "Priorità Bassa": "green",
    }

    priority_order = [
        "Priorità Alta",
        "Priorità Media",
        "Priorità Bassa",
    ]

    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    if nodes_gdf.empty:
        print("Il grafo non contiene nodi. Impossibile calcolare il centro della mappa.")
        return

    center_lat = nodes_gdf['y'].mean()
    center_lon = nodes_gdf['x'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="cartodbpositron")

    # Aggiungi gli edges (strade) per contesto, ma in un layer separato e meno invasivo
    nodes, edges = ox.graph_to_gdfs(G)
    folium.GeoJson(
        edges.geometry.__geo_interface__,
        name="Rete Stradale", # Nome del layer
        style_function=lambda x: {
            "color": "#cccccc", # Colore grigio chiaro per le strade di sfondo
            "weight": 0.8,
            "opacity": 0.5
        },
        show=True
    ).add_to(m)

    # Crea FeatureGroup separati per ogni categoria di priorità
    feature_groups = {}
    for category in priority_order:
        if category == "Priorità Alta":
            show_default = True # Visibile di default
        else:
            show_default = False # Nascosto di default
        feature_groups[category] = folium.FeatureGroup(name=category, show=show_default).add_to(m)


    for idx, row in recommendations_df.iterrows():
        osmid = row['osmid']
        try:
            node_data = G.nodes[osmid]
            lat = node_data['y']
            lon = node_data['x']

            # Recupera la categoria di priorità e il colore
            current_priority_category = row.get('categoria_priorita', 'Priorità Bassa')
            marker_color = priority_colors.get(current_priority_category, "gray") # Fallback a grigio se la categoria non è mappata


            popup_html = f"""
            <h4>Raccomandazione Incrocio: {osmid}</h4>
            <p><strong>Punteggio Priorità:</strong> <span style="color:{marker_color}; font-weight:bold;">{row['indice_rischio']:.2f}</span></p>
            <p><strong>Categoria Priorità:</strong> <span style="color:{marker_color}; font-weight:bold;">{current_priority_category}</span></p>
            <p><strong>Rischio Incidente Predetto:</strong> {row['probabilità_rischio_incidente_predetta']:.3f}</p>
            <p><strong>Gravità Media Incidenti Storici:</strong> {row['gravita_media_incidente']:.2f}</p>
            <p><strong>N. Incidenti (storico):</strong> {row['num_incidenti_vicini']:.0f}</p>
            <p><strong>Raccomandazione Specifico:</strong> <span style="color:blue; font-weight:bold;">{row['intervento']}</span></p>
            """

            iframe = folium.IFrame(html=popup_html, width=350, height=250)
            popup = folium.Popup(iframe, max_width=450)

            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.8,
                popup=popup,
                tooltip=f"{current_priority_category}: {row['indice_rischio']:.2f}"
            ).add_to(feature_groups[current_priority_category]) # Aggiungi al FeatureGroup corretto

        except KeyError as e:
            print(f"KeyError: Colonna o dato '{e}' mancante nel popup o nel tooltip per nodo {osmid}. Saltando la visualizzazione.")
            continue
        except Exception as e:
            print(f"Errore generico durante l'elaborazione del nodo {osmid}: {e}. Saltando la visualizzazione.")
            continue

    folium.LayerControl().add_to(m)

    m.save(filepath)
    print(f"Mappa delle raccomandazioni salvata in {filepath}")

    try:
        webbrowser.open(f'file://{os.path.abspath(filepath)}')
        print(f"Apertura della mappa delle raccomandazioni nel browser: {os.path.abspath(filepath)}")
    except Exception as e:
        print(f"Impossibile aprire automaticamente la mappa nel browser: {e}")


def plot_osm_data_on_map(G, pois_gdf, filepath="reports/osm_data_map.html"):
    """
    Crea una mappa interattiva per visualizzare la rete stradale OSM e i POI scaricati.
    """
    if G is None:
        print("Impossibile generare la mappa dei dati OSM: grafo non disponibile.")
        return

    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    if nodes_gdf.empty:
        print("Il grafo non contiene nodi. Impossibile calcolare il centro della mappa.")
        return

    center_lat = nodes_gdf['y'].mean()
    center_lon = nodes_gdf['x'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="cartodbpositron")

    # --- Rete Stradale ---
    nodes, edges = ox.graph_to_gdfs(G)
    # Assicurati che edges sia in un CRS geografico per Folium
    if edges.crs and not edges.crs.is_geographic:
        edges = edges.to_crs(epsg=4326)

    folium.GeoJson(
        edges.geometry.__geo_interface__,
        name="Rete Stradale OSM", # Nome del layer per la leggenda
        style_function=lambda x: {
            "color": "#666666",
            "weight": 0.8,
            "opacity": 0.5
        }
    ).add_to(m)

    # --- POI con Legenda e Layer Controllabili ---

    # Dizionario per mappare i tipi di POI a icone e colori
    poi_icons_config = {
        'traffic_signals': {'icon': 'traffic-light', 'color': 'orange', 'prefix': 'fa', 'name': 'Semaforo'}, # Personalizzato da 'lightbulb' a 'traffic-light'
        #'crossing': {'icon': 'male', 'color': 'green', 'prefix': 'fa', 'name': 'Attraversamento Pedonale'}, # Personalizzato da 'walking' a 'walk'
        'crossing': {'icon': 'walking', 'color': 'blue', 'prefix': 'fa', 'name': 'Strisce Pedonali'}, #Attraversamenti Pedonali Segnalati e Controllati
        'school': {'icon': 'school', 'color': 'purple', 'prefix': 'fa', 'name': 'Scuola'},
        'hospital': {'icon': 'hospital', 'color': 'red', 'prefix': 'fa', 'name': 'Ospedale'}, # Personalizzato da 'h-sign' a 'hospital'

        #'police': {'icon': 'shield-halved', 'color': 'cadetblue', 'prefix': 'fa', 'name': 'Polizia'},
        #'fire_station': {'icon': 'fire-extinguisher', 'color': 'darkred', 'prefix': 'fa', 'name': 'Vigili del Fuoco'},

        #'shop': {'icon': 'cart-shopping', 'color': 'darkgreen', 'prefix': 'fa', 'name': 'Negozio'},
        ##'supermarket': {'icon': 'store', 'color': 'darkblue', 'prefix': 'fa', 'name': 'Supermercato'},
        ##'bakery': {'icon': 'bread-slice', 'color': 'cadetblue', 'prefix': 'fa', 'name': 'Panificio'},
        ##'clothes': {'icon': 'shirt', 'color': 'pink', 'prefix': 'fa', 'name': 'Negozio di Vestiti'},
        ##'pharmacy': {'icon': 'truck-medical', 'color': 'lightred', 'prefix': 'fa', 'name': 'Farmacia'},
        ##'hairdresser': {'icon': 'scissors', 'color': 'darkpurple', 'prefix': 'fa', 'name': 'Parrucchiere'},
        ##'car_shop': {'icon': 'car', 'color': 'gray', 'prefix': 'fa', 'name': 'Negozi di Auto'},
        ##'shop_default': {'icon': 'cart-shopping', 'color': 'darkgreen', 'prefix': 'fa', 'name': 'Negozio (generico)'},
        'leisure_area': {'icon': 'tree', 'color': 'darkgreen', 'prefix': 'fa', 'name': 'Area Ricreativa'}, #Considera parchi, ecc.
        # Puoi aggiungere altri tipi di POI qui
        'default': {'icon': 'info-circle', 'color': 'blue', 'prefix': 'fa', 'name': 'Altro POI'} # Personalizzato da 'info-sign' a 'info-circle'
    }

    # Creiamo un MarkerCluster per i POI di default per evitare l'affollamento
    default_poi_cluster = MarkerCluster(name="Altri Punti di Interesse").add_to(m)

    # Creiamo FeatureGroup per POI specifici per la leggenda
    traffic_signals_layer = folium.FeatureGroup(name="Semafori").add_to(m)
    #crossings_layer = folium.FeatureGroup(name="Attraversamenti Pedonali").add_to(m)
    crossings_layer = folium.FeatureGroup(name="Strisce Pedonali").add_to(m)
    schools_layer = folium.FeatureGroup(name="Scuole").add_to(m)
    hospitals_layer = folium.FeatureGroup(name="Ospedali").add_to(m)
    #police_layer = folium.FeatureGroup(name="Polizia").add_to(m)
    #fire_station_layer = folium.FeatureGroup(name="Vigili del Fuoco").add_to(m)
    parks_playgrounds_layer = folium.FeatureGroup(name="Parchi").add_to(m)
    ##shops_layer = folium.FeatureGroup(name="Negozi").add_to(m)
    # Aggiungi layer per altri POI se vuoi che siano separabili

    if pois_gdf is not None and not pois_gdf.empty:
        #pois_gdf = pois_gdf[pois_gdf.geometry.geom_type == 'Point'] #Non si visualizzavano alcuni POI
        # Assicurati che pois_gdf sia in un CRS geografico (WGS84, EPSG:4326) per Folium
        if pois_gdf.crs and not pois_gdf.crs.is_geographic:
            pois_gdf = pois_gdf.to_crs(epsg=4326)

        #####################################
        for _, row in pois_gdf.iterrows():
            lat, lon = None, None
            geom_type = row.geometry.geom_type

            if geom_type == 'Point':
                lat, lon = row.geometry.y, row.geometry.x
            elif geom_type in ('Polygon', 'LineString'):
                if row.geometry.centroid.is_valid and not row.geometry.centroid.is_empty:
                    lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
                else:
                    bounds = row.geometry.bounds
                    if bounds:
                        lat, lon = (bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2

            if lat is None or lon is None:
                # Questo print è utile solo se hai ancora problemi con le coordinate
                # In un'applicazione stabile potresti anche rimuoverlo.
                # print(f"Impossibile ottenere coordinate valide per il POI {row.get('osmid', idx)} con geometria {geom_type}. Saltando.")
                continue

            popup_html = f"""
            <b>Tipo:</b> {row.get('highway', row.get('amenity', row.get('leisure', row.get('shop', row.get('building', 'N/A')))))}<br>
            <b>Nome:</b> {row.get('name', 'N/A')}<br>
            <b>Indirizzo:</b> {row.get('addr:street', 'N/A')} {row.get('addr:housenumber', '')}<br>
            <b>Dettagli:</b> {', '.join([f"{k}:{v}" for k, v in row.items() if pd.notna(v) and k not in ['geometry', 'osmid', 'name', 'highway', 'amenity', 'addr:street', 'addr:housenumber', 'leisure', 'shop', 'building']])}
            """

            ####popup_html = f"""
            ####<b>OSMID:</b> {row.get('osmid', 'N/A')}<br>
            ####<b>Tipo:</b> {row.get('highway', row.get('amenity', 'N/A'))}<br>
            ####<b>Nome:</b> {row.get('name', 'N/A')}<br>
            ####<b>Indirizzo:</b> {row.get('addr:street', 'N/A')} {row.get('addr:housenumber', '')}<br>
            ####<b>Dettagli:</b> {', '.join([f"{k}:{v}" for k, v in row.items() if pd.notna(v) and k not in ['geometry', 'osmid', 'name', 'highway', 'amenity', 'addr:street', 'addr:housenumber']])}
            ####"""

            ##popup_html = f"""
            ##<b>OSMID:</b> {row.get('osmid', 'N/A')}<br>
            ##<b>Tipo:</b> {row.get('highway', row.get('amenity', row.get('shop', 'N/A')))}<br>
            ##<b>Nome:</b> {row.get('name', 'N/A')}<br>
            ##<b>Indirizzo:</b> {row.get('addr:street', 'N/A')} {row.get('addr:housenumber', '')}<br>
            ##<b>Dettagli:</b> {', '.join([f"{k}:{v}" for k, v in row.items() if pd.notna(v) and k not in ['geometry', 'osmid', 'name', 'highway', 'amenity', 'addr:street', 'addr:housenumber', 'shop']])}
            ##"""

            # Determina icona e colore in base al tipo di POI
            icon_config = poi_icons_config['default']
            target_layer = default_poi_cluster

            if 'highway' in row and pd.notna(row['highway']):
                highway_value = str(row['highway']).strip().lower()
                if highway_value == 'traffic_signals':
                    icon_config = poi_icons_config['traffic_signals']
                    target_layer = traffic_signals_layer
                elif highway_value == 'crossing':
                    #if row.get('crossing_ref') == 'zebra':  # Check for zebra crossing
                    #Alcune strisce pedonali vengono considerate anche con altro tag OSM
                    if str(row.get('crossing_ref', '')).strip().lower() == 'zebra' or \
                            str(row.get('crossing', '')).strip().lower() == 'marked' or \
                            str(row.get('crossing', '')).strip().lower() == 'traffic_signals':
                        icon_config = poi_icons_config['crossing']
                        target_layer = crossings_layer
                    ##else:
                        ##icon_config = poi_icons_config['crossing']
                        ##target_layer = crossings_layer

                    #Commentare la parte di sopra e decommentare le seguenti righe per considerare TUTTI gli attraversamenti pedonali
                    #icon_config = poi_icons_config['crossing']
                    #target_layer = crossings_layer
            elif 'amenity' in row and pd.notna(row['amenity']):
                amenity_value = str(row['amenity']).strip().lower()
                if amenity_value == 'school' or amenity_value == 'kindergarten':
                    icon_config = poi_icons_config['school']
                    target_layer = schools_layer
                elif amenity_value == 'hospital':
                    icon_config = poi_icons_config['hospital']
                    target_layer = hospitals_layer
                #elif amenity_value == 'police':
                #    icon_config = poi_icons_config['police']
                #    target_layer = police_layer
                #elif amenity_value == 'fire_station':
                #    icon_config = poi_icons_config['fire_station']
                #    target_layer = fire_station_layer
                # Gli altri tipi di 'amenity' (es. fuel, cafe) finiranno nel default_poi_cluster
            elif 'leisure' in row and pd.notna(row['leisure']):
                leisure_value = str(row['leisure']).strip().lower()
                if leisure_value in ['park', 'playground']: # Raggruppa entrambi i valori
                    icon_config = poi_icons_config['leisure_area'] # Usa la configurazione comune
                    target_layer = parks_playgrounds_layer
            #elif 'shop' in row and pd.notna(row['shop']):
                #shop_value = str(row['shop']).strip().lower()
                # I POI con tag 'shop' finiranno nel default_poi_cluster

                ### Assegna icona e colore in base al tipo specifico di shop
                ##if shop_value == 'supermarket':
                ##    icon_config = poi_icons_config['supermarket']
                ##elif shop_value == 'bakery':
                ##    icon_config = poi_icons_config['bakery']
                ##elif shop_value == 'clothes':
                ##    icon_config = poi_icons_config['clothes']
                ##elif shop_value == 'pharmacy':
                ##    icon_config = poi_icons_config['pharmacy']
                ##elif shop_value == 'hairdresser':
                ##    icon_config = poi_icons_config['hairdresser']
                ##elif shop_value in ['car', 'dealership']: # Controlla entrambi i tag comuni
                ##    icon_config = poi_icons_config['car_shop']
                ##else: # Per tutti gli altri tipi di shop non specificati
                ##    icon_config = poi_icons_config['shop_default']

                ##target_layer = shops_layer # Tutti i negozi vanno comunque nel layer "Negozi"
            #elif 'building' in row and str(row['building']).strip().lower() == 'public' and pd.notna(row['building']):
                # I POI con tag 'building=public' finiranno nel default_poi_cluster
                #pass # Non facciamo nulla qui, il target_layer è già default_poi_cluster
            # Se nessun 'if/elif' precedente è stato catturato, il POI resta nel default_poi_cluster

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(
                    color=icon_config['color'],
                    icon=icon_config['icon'],
                    prefix=icon_config['prefix']
                )
            ).add_to(target_layer)
        #####################################


        ####if not pois_gdf.empty:
        ####    for _, row in pois_gdf.iterrows():
        ####        ####################################
        ####        # Gestione di geometrie Point e Polygon (e potenzialmente LineString)
        ####        if row.geometry.geom_type == 'Point':
        ####            lat, lon = row.geometry.y, row.geometry.x
        ####        elif row.geometry.geom_type in ('Polygon', 'LineString'):
        ####            # Prendi il centroide per Polygon e LineString
        ####            lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
        ####        else:
        ####            print(f"ATTENZIONE: Geometria non supportata: {row.geometry.geom_type}. Saltando.")
        ####            continue  # Salta questa iterazione
        ####        ####################################

        ####        #################################### lat, lon = row.geometry.y, row.geometry.x

        ####        popup_html = f"""
        ####        <b>OSMID:</b> {row.get('osmid', 'N/A')}<br>
        ####        <b>Tipo:</b> {row.get('highway', row.get('amenity', 'N/A'))}<br>
        ####        <b>Nome:</b> {row.get('name', 'N/A')}<br>
        ####        <b>Indirizzo:</b> {row.get('addr:street', 'N/A')} {row.get('addr:housenumber', '')}<br>
        ####        <b>Dettagli:</b> {', '.join([f"{k}:{v}" for k,v in row.items() if pd.notna(v) and k not in ['geometry', 'osmid', 'name', 'highway', 'amenity', 'addr:street', 'addr:housenumber']])}
        ####        """

        ####        # Determina icona e colore in base al tipo di POI
        ####        icon_config = poi_icons_config['default']
        ####        target_layer = default_poi_cluster # Default: Vanno nel cluster "Altri Punti di Interesse"

        ####        if 'highway' in row:
        ####            if row['highway'] == 'traffic_signals':
        ####                icon_config = poi_icons_config['traffic_signals']
        ####                target_layer = traffic_signals_layer
        ####            elif row['highway'] == 'crossing':
        ####                icon_config = poi_icons_config['crossing']
        ####                target_layer = crossings_layer
        ####        elif 'amenity' in row:
        ####            if row['amenity'] == 'school':
        ####                icon_config = poi_icons_config['school']
        ####                target_layer = schools_layer
        ####            elif row['amenity'] == 'hospital':
        ####                icon_config = poi_icons_config['hospital']
        ####                target_layer = hospitals_layer
        ####            elif row['amenity'] == 'police':
        ####                icon_config = poi_icons_config['police']
        ####                target_layer = police_layer
        ####            elif row['amenity'] == 'fire_station':
        ####                icon_config = poi_icons_config['fire_station']
        ####                target_layer = fire_station_layer
        ####        # Se nessun 'if/elif' precedente è stato catturato, il POI resta nel default_poi_cluster

        ####        folium.Marker(
        ####            location=[lat, lon],
        ####            popup=folium.Popup(popup_html, max_width=300),
        ####            icon=folium.Icon(
        ####                color=icon_config['color'],
        ####                icon=icon_config['icon'],
        ####                prefix=icon_config['prefix']
        ####            )
        ####        ).add_to(target_layer)
        ####else:
        ####    print("GeoDataFrame dei POI è vuoto dopo il filtro per geometria Point. Nessun POI da plottare.")
    else:
        print("GeoDataFrame dei POI non disponibile o vuoto. Nessun POI da plottare.")

    # Aggiungi il controllo dei layer alla mappa
    folium.LayerControl().add_to(m)

    m.save(filepath)
    print(f"Mappa interattiva dei dati OSM salvata in  {filepath}")

    try:
        webbrowser.open(f'file://{os.path.abspath(filepath)}')
        print(f"Apertura della mappa dei dati OSM nel browser: {os.path.abspath(filepath)}")
    except Exception as e:
        print(f"Impossibile aprire automaticamente la mappa nel browser: {e}")
#
# # # src/visualization.py
# # import folium
# # import osmnx as ox
# # import pandas as pd
# # import geopandas as gpd
# #
# # def plot_recommendations_on_map(G, recommendations_df, filepath="reports/recommendations_map.html"):
# #     """
# #     Crea una mappa interattiva con le raccomandazioni evidenziate.
# #     """
# #     if G is None or recommendations_df.empty:
# #         print("Impossibile generare la mappa delle raccomandazioni: dati mancanti o vuoti.")
# #         return
# #
# #     # Calcola il centro della mappa in base ai nodi del grafo
# #     nodes_gdf = ox.graph_to_gdfs(G, edges=False) # Corretto per expected 1 value
# #     if nodes_gdf.empty:
# #         print("Il grafo non contiene nodi. Impossibile calcolare il centro della mappa.")
# #         return
# #
# #     center_lat = nodes_gdf['y'].mean()
# #     center_lon = nodes_gdf['x'].mean()
# #
# #     m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="cartodbpositron")
# #
# #     # Aggiungi i nodi raccomandati con un colore e un popup per i dettagli
# #     for idx, row in recommendations_df.iterrows():
# #         node_osmid = row['osmid']
# #         try:
# #             node_data = G.nodes[node_osmid]
# #             lat = node_data['y']
# #             lon = node_data['x']
# #
# #             # Crea il popup HTML con le informazioni
# #             popup_html = f"""
# #             <h4>Raccomandazione Incrocio OSMID: {node_osmid}</h4>
# #             <p><strong>Punteggio Priorità:</strong> {row['priority_score']:.2f}</p>
# #             <p><strong>Rischio Incidente Predetto:</strong> {row['predicted_accident_risk']:.2f}</p>
# #             <p><strong>Traffico Predetto:</strong> {row['predicted_traffic']:.0f} veicoli/ora</p>
# #             <p><strong>N. Incidenti (storico):</strong> {row['num_incidenti']:.0f}</p>
# #             <p><strong>Semaforo Esistente:</strong> {'Sì' if row['semaforo_esistente'] == 1 else 'No'}</p>
# #             <p><strong>Strisce Esistenti:</strong> {'Sì' if row['strisce_esistenti'] == 1 else 'No'}</p>
# #             <p><strong>Raccomandazione:</strong> {row['recommended_intervention']}</p>
# #             """
# #             iframe = folium.IFrame(html=popup_html, width=300, height=200)
# #             popup = folium.Popup(iframe, max_width=400)
# #
# #             folium.CircleMarker(
# #                 location=[lat, lon],
# #                 radius=10,
# #                 color='red',
# #                 fill=True,
# #                 fill_color='red',
# #                 fill_opacity=0.7,
# #                 popup=popup
# #             ).add_to(m)
# #
# #         except KeyError:
# #             print(f"Nodo OSMID {node_osmid} non trovato nel grafo, saltando la visualizzazione.")
# #             continue
# #
# #     m.save(filepath)
# #     print(f"Mappa delle raccomandazioni salvata in {filepath}")
# #
# #
# # def plot_osm_data_on_map(G, pois_gdf, filepath="reports/osm_data_map.html"):
# #     """
# #     Crea una mappa interattiva per visualizzare la rete stradale OSM e i POI scaricati.
# #     """
# #     if G is None:
# #         print("Impossibile generare la mappa dei dati OSM: grafo non disponibile.")
# #         return
# #
# #     # Calcola il centro della mappa in base ai nodi del grafo
# #     nodes_gdf = ox.graph_to_gdfs(G, edges=False) # Corretto per expected 1 value
# #     if nodes_gdf.empty:
# #         print("Il grafo non contiene nodi. Impossibile calcolare il centro della mappa.")
# #         return
# #
# #     center_lat = nodes_gdf['y'].mean()
# #     center_lon = nodes_gdf['x'].mean()
# #
# #     m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="cartodbpositron")
# #
# #     # --- SOSTITUZIONE DI ox.plot_graph_folium ---
# #     # Converti il grafo in GeoDataFrames per nodi e edges
# #     nodes, edges = ox.graph_to_gdfs(G) # Questa volta vogliamo entrambi
# #
# #     # Aggiungi gli edges (strade) alla mappa
# #     # Se il grafo è molto grande, questa operazione può rallentare o rendere il file enorme.
# #     # Puoi considerare di plottare solo un campione o usare un'altra libreria per grafici molto grandi.
# #     # Per semplicità, aggiungiamo tutte le geometrie degli edges.
# #     folium.GeoJson(
# #         edges.geometry.__geo_interface__,
# #         style_function=lambda x: {
# #             "color": "#666666", # Colore grigio per le strade
# #             "weight": 1.5,
# #             "opacity": 0.7
# #         }
# #     ).add_to(m)
# #
# #     # Aggiungi i nodi del grafo alla mappa (opzionale, ma utile per incroci)
# #     # folium.GeoJson(
# #     #     nodes.geometry.__geo_interface__,
# #     #     marker=folium.CircleMarker(radius=2, color="blue", fill=True, fill_color="blue", fill_opacity=0.6),
# #     #     tooltip=folium.features.GeoJsonTooltip(fields=['osmid'], aliases=['Node ID:']),
# #     # ).add_to(m)
# #     # Ho commentato i nodi per non appesantire troppo la mappa se ci sono molti incroci.
# #     # Li abbiamo già per le raccomandazioni e POI.
# #     # --- FINE SOSTITUZIONE ---
# #
# #     # Aggiungi i POI alla mappa
# #     if pois_gdf is not None and not pois_gdf.empty:
# #         pois_gdf = pois_gdf[pois_gdf.geometry.geom_type == 'Point']
# #         if not pois_gdf.empty:
# #             for _, row in pois_gdf.iterrows():
# #                 lat, lon = row.geometry.y, row.geometry.x
# #
# #                 popup_text = f"<b>Tipo:</b> {row.get('highway', '')} {row.get('amenity', '')}<br>"
# #                 for prop in ['name', 'addr:street', 'addr:housenumber', 'operator']:
# #                     if prop in row and pd.notna(row[prop]):
# #                         popup_text += f"<b>{prop.replace('addr:', '').replace('_', ' ').title()}:</b> {row[prop]}<br>"
# #
# #                 icon_name = 'info-sign'
# #                 icon_color = 'blue'
# #                 if 'highway' in row and row['highway'] == 'traffic_signals':
# #                     icon_name = 'lightbulb'
# #                     icon_color = 'orange'
# #                 elif 'highway' in row and row['highway'] == 'crossing':
# #                     icon_name = 'walking'
# #                     icon_color = 'green'
# #                 elif 'amenity' in row and row['amenity'] == 'school':
# #                     icon_name = 'school'
# #                     icon_color = 'purple'
# #                 elif 'amenity' in row and row['amenity'] == 'hospital':
# #                     icon_name = 'h-sign'
# #                     icon_color = 'red'
# #
# #                 folium.Marker(
# #                     location=[lat, lon],
# #                     popup=folium.Popup(popup_text, max_width=300),
# #                     icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa')
# #                 ).add_to(m)
# #         else:
# #             print("GeoDataFrame dei POI è vuoto dopo il filtro per geometria Point. Nessun POI da plottare.")
# #     else:
# #         print("GeoDataFrame dei POI non disponibile o vuoto. Nessun POI da plottare.")
# #
# #     m.save(filepath)
# #     print(f"Mappa dei dati OSM salvata in {filepath}")