# Installazione e avvio

1.  **Prerequisiti:** Prima di iniziare, assicurati di avere installati sul tuo sistema:
    * Python 3.9 o superiore: Si raccomanda una versione recente di Python 3. (Puoi scaricarlo da [python.org](https://www.python.org/)). Durante l'installazione su Windows, assicurati di selezionare l'opzione "Add Python to PATH".
    * Git: Per clonare il repository del progetto. (Puoi scaricarlo da [git-scm.com](https://git-scm.com/)).
2.  **Clonazione del Repository:** Apri il tuo terminale (Prompt dei Comandi su Windows, Terminale su macOS/Linux) e segui questi passaggi:
    * Naviga nella directory dove desideri clonare il progetto: `cd C:\Users\USER\Desktop\` (Sostituisci il percorso con la tua directory preferita)
    * Clona il repository del progetto: `git clone https://github.com/GaeMale/Pianificazione_urbana.git`
    * Naviga nella directory del progetto appena clonata: `cd Pianificazione_urbana`
3.  **Configurazione dell'Ambiente Virtuale (Raccomandato):** È altamente consigliato utilizzare un ambiente virtuale per isolare le dipendenze del progetto.
    * Crea l’ambiente virtuale: Assicurati di essere nella directory principale del progetto (`Progetto_pianificazione_urbana`) e lancia da terminale il comando `python -m venv venv` <br>
        Questo creerà una sottocartella chiamata `venv` all'interno della directory
    * Attiva l'ambiente virtuale:
        * Su Windows (Prompt dei Comandi): `.\venv\Scripts\activate`
        * Su macOS / Linux (Terminale): `source venv/bin/activate` <br>
        Dovresti vedere `(venv)` (o il nome che hai dato al tuo ambiente virtuale) apparire all'inizio del tuo prompt, indicando che l'ambiente è attivo.
4.  **Installazione delle Dipendenze:** Una volta attivato l'ambiente virtuale, installa tutte le librerie Python necessarie elencate nel file `requirements.txt`, mediante il seguente comando:
    `pip install -r requirements.txt`
    <br><b>N.B:</b> Assicurati di essere nella directory principale del progetto
5.  **Avvio del Progetto:** Il progetto può essere avviato eseguendo lo script principale, che è `main.py`.
    * Avvio tramite Terminale (con ambiente virtuale attivo):
        * Assicurati che il tuo ambiente virtuale sia attivo (dovresti vedere `(venv)` nel prompt).
        * Naviga alla directory principale del progetto.
        * Esegui lo script principale: `python main.py`

Il progetto inizierà a eseguire le sue operazioni, che includeranno il caricamento dei dati, l'analisi, la modellazione e la generazione dei risultati/raccomandazioni.
