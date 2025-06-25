import pandas as pd


def score_candidate_locations(df):
    """
    Calcola un 'indice_rischio' per ogni location candidata basandosi su vari fattori.

    Args:
        df (pd.DataFrame): DataFrame contenente le feature dei nodi, incluso 'osmid'
                          e le colonne rilevanti per il calcolo del rischio.

    Returns:
        pd.DataFrame: Il DataFrame originale con l'aggiunta della colonna 'indice_rischio'.
    """

    if df.index.name == 'osmid':
        df = df.reset_index()
    elif 'osmid' not in df.columns:
        print("Errore: la colonna 'osmid' non è stata trovata nel DataFrame per lo scoring.")
        return pd.DataFrame()

    w_num_incidenti = 0.25        # Numero di incidenti storici (molto significativo)
    w_avg_conteggio_veicoli = 0.15 # Volume di traffico
    w_grado_incrocio = 0.10       # Complessità dell'incrocio
    w_num_attraversamenti_pedonali = 0.10 # Rischio pedonale
    w_num_scuole = 0.10           # Presenza di scuole (utenti vulnerabili)
    w_avg_velocita = 0.10         # Velocità media (influenza la gravità)
    w_avg_indice_congestione = 0.05 # Congestione (rischio di tamponamenti minori)
    w_num_pois = 0.075            # Punti di interesse (indica attività generale)
    w_num_negozi = 0.075          # Negozi (indica attività pedonale/veicolare)

    # La somma dei pesi dovrebbe essere 1.0 (o si dovrebbe normalizzare il risultato finale)

    # Normalizzazione delle feature tramite ranking

    required_columns = [
        'grado_incrocio', 'num_incidenti_vicini', 'num_pois_vicini',
        'num_attraversamenti_pedonali_vicini', 'num_scuole_vicine', 'num_negozi_vicini',
        'avg_conteggio_veicoli', 'avg_velocita', 'avg_indice_congestione'
    ]
    for col in required_columns:
        if col not in df.columns:
            print(f"ATTENZIONE: Colonna '{col}' mancante nel DataFrame. Verrà trattata come zeros o NaN da rank().")

    ranked_grado_incrocio = df['grado_incrocio'].rank(pct=True, method='average')
    ranked_num_incidenti = df['num_incidenti_vicini'].rank(pct=True, method='average')
    ranked_num_pois = df['num_pois_vicini'].rank(pct=True, method='average')
    ranked_num_attraversamenti_pedonali = df['num_attraversamenti_pedonali_vicini'].rank(pct=True, method='average')
    ranked_num_scuole = df['num_scuole_vicine'].rank(pct=True, method='average')
    ranked_num_negozi = df['num_negozi_vicini'].rank(pct=True, method='average')
    ranked_avg_conteggio_veicoli = df['avg_conteggio_veicoli'].rank(pct=True, method='average')
    ranked_avg_velocita = df['avg_velocita'].rank(pct=True, method='average')
    ranked_avg_indice_congestione = df['avg_indice_congestione'].rank(pct=True, method='average')

    # Calcolo dell'indice di rischio combinato
    df['indice_rischio'] = (
            w_num_incidenti * ranked_num_incidenti +
            w_avg_conteggio_veicoli * ranked_avg_conteggio_veicoli +
            w_grado_incrocio * ranked_grado_incrocio +
            w_num_attraversamenti_pedonali * ranked_num_attraversamenti_pedonali +
            w_num_scuole * ranked_num_scuole +
            w_num_negozi * ranked_num_negozi +
            w_avg_velocita * ranked_avg_velocita +
            w_avg_indice_congestione * ranked_avg_indice_congestione +
            w_num_pois * ranked_num_pois
    )

    return df


def recommend_interventions(scored_df):
    top_candidates = scored_df.sort_values(by='indice_rischio', ascending=False)

    recommendations = []
    for index, row in top_candidates.iterrows():
        osmid = row['osmid']
        num_inc = row['num_incidenti_vicini']
        avg_sev = row['gravita_media_incidente']
        risk_proba = row['probabilità_rischio_incidente_predetta']
        score = row['indice_rischio']

        # Esempio di logica per le raccomandazioni
        recommendation_text = ""
        if score >= 0.8: # Esempio: score da 0.8 a 1.0 (o oltre) è Alta
            priority_category = "Priorità Alta"
            if num_inc > 3 and avg_sev > 2: # Molti incidenti, alta gravità
                recommendation_text = "AZIONI URGENTI: Riprogettazione complessiva dell'intersezione, installazione/aggiornamento semafori intelligenti, revisione drastica dei limiti di velocità e controlli rafforzati. Considerare interventi di moderazione del traffico."
            elif num_inc > 0: # Incidenti presenti, anche se meno gravi/frequenti
                recommendation_text = "ALTA PRIORITÀ: Valutazione dettagliata per modifiche geometriche (es. rotonda, canalizzazione), potenziamento segnaletica orizzontale/verticale, analisi e gestione dei flussi veicolari per il controllo della velocità."
            else: # Rischio calcolato alto ma senza incidenti storici
                recommendation_text = "ALTA PRIORITÀ (PREVENTIVA): Monitoraggio approfondito dei comportamenti di guida e dei flussi di traffico. Indagine per identificare anomalie latenti nel design stradale o nel contesto urbano che generano alto rischio teorico."

        elif score >= 0.5: # Esempio: score da 0.5 a <0.8 è Medio
            priority_category = "Priorità Media"
            if num_inc > 0: # Incidenti presenti
                recommendation_text = "PRIORITÀ MEDIA: Interventi di miglioramento della segnaletica (anche luminosa), potenziamento dell'illuminazione pubblica, analisi punti ciechi e visibilità per utenti vulnerabili (pedoni, ciclisti)."
            else: # Rischio moderato senza incidenti storici
                recommendation_text = "PRIORITÀ MEDIA (PREVENTIVA): Revisione periodica della segnaletica, piccole migliorie alla visibilità in prossimità dell'incrocio, analisi di potenziali conflitti veicolari/pedonali."
        else: # Score < 0.5 = Rischio Basso
            priority_category = "Priorità Bassa"
            recommendation_text = "PRIORITÀ BASSA: Mantenimento dello stato attuale con manutenzione ordinaria della segnaletica e del manto stradale. Controlli di routine per garantire la sicurezza standard."

        recommendations.append({
            'osmid': osmid,
            'indice_rischio': score,
            'probabilità_rischio_incidente_predetta': risk_proba,
            'gravita_media_incidente': avg_sev,
            'num_incidenti_vicini': num_inc,
            'intervento': recommendation_text,
            'categoria_priorita': priority_category
        })
    return recommendations
