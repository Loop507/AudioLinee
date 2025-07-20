# 🎨 AudioLinee

**AudioLinee** è un'applicazione web basata su Streamlit che genera video visivi sincronizzati con la musica. Carica un file audio e osserva come gli effetti grafici si muovono e reagiscono al volume e al tempo del brano, creando un'esperienza audiovisiva dinamica e accattivante.

## Caratteristiche

* **Sincronizzazione Audio-Visiva:** Gli effetti grafici sono strettamente legati all'analisi del volume (RMS) e del tempo musicale (onset strength) dell'audio.
* **Effetti Artistici Multipli:** Scegli tra diverse modalità di visualizzazione:
    * `connessioni`: Linee casuali che si connettono, pulsando al ritmo della musica.
    * `rettangoli_griglia`: Una griglia di punti che formano rettangoli interconnessi.
    * `geometriche`: Una rete complessa di forme geometriche che reagiscono all'audio.
    * `linee_orizzontali`: Linee orizzontali che pulsano e si muovono.
    * `linee_verticali`: Linee verticali che pulsano e si muovono.
    * `linee_casuali_verticali_orizzontali`: Una combinazione di linee verticali e orizzontali casuali.
    * `quadrati`: Quadrati di varie dimensioni che appaiono e pulsano.
    * `rettangoli`: Rettangoli di diverse proporzioni che si animano con la musica.
    * `forme_casuali_quadrati_rettangoli`: Una miscela casuale di quadrati e rettangoli.
* **Personalizzazione:**
    * Seleziona il formato del video (16:9, 9:16, 1:1, 4:3).
    * Regola l'intensità degli effetti (`soft`, `medium`, `hard`).
    * Scegli il colore di sfondo e i colori per le bande di frequenza (basse, medie, alte).
    * Imposta i fotogrammi al secondo (FPS) per il video.
* **Anteprima Video:** Visualizza una breve anteprima del video generato direttamente nell'applicazione.
* **Download:** Scarica il video finale con l'audio sincronizzato.

## Requisiti

Per eseguire questa applicazione, avrai bisogno di:

* **Python 3.7+**
* **pip** (gestore di pacchetti Python)
* **FFmpeg** (per la sincronizzazione dell'audio e la generazione dell'anteprima video). Assicurati che FFmpeg sia installato e disponibile nel tuo PATH di sistema.

## Installazione

1.  **Clona il repository (o scarica il codice):**
    ```bash
    git clone [https://github.com/tuo-utente/audiolinee.git](https://github.com/tuo-utente/audiolinee.git) # Sostituisci con il tuo link al repository
    cd audiolinee
    ```

2.  **Crea un ambiente virtuale (consigliato):**
    ```bash
    python -m venv venv
    source venv/bin/activate # Su Linux/macOS
    # venv\Scripts\activate # Su Windows
    ```

3.  **Installa le dipendenze Python:**
    ```bash
    pip install -r requirements.txt
    ```
    (Se non hai un `requirements.txt`, puoi crearlo con `pip freeze > requirements.txt` dopo aver installato le librerie, oppure installarle manualmente: `pip install streamlit numpy opencv-python librosa`)

## Come eseguire l'applicazione

Dopo aver installato le dipendenze, puoi avviare l'applicazione Streamlit:

```bash
streamlit run app.py

L'applicazione si aprirà automaticamente nel tuo browser web.

Utilizzo
Carica un file audio: Fai clic sul pulsante "🎧 Carica un file audio (.wav o .mp3)" e seleziona il tuo brano.

Configura le opzioni: Scegli il formato video, il livello degli effetti, gli FPS, l'effetto artistico desiderato e i colori.

Genera il video: Fai clic sul pulsante "🎬 Genera Video Sincronizzato". L'applicazione elaborerà l'audio e genererà il video.

Scarica/Anteprima: Una volta completata la generazione, potrai scaricare il video finale o vederne un'anteprima.

Pulisci: Puoi usare il pulsante "🧹 Pulisci file temporanei" per rimuovere i file audio e video intermedi.

Crediti e Menzione
Questo codice è stato sviluppato con il supporto di Loop507.

Se utilizzi o condividi questo codice e le sue derivazioni, ti chiedo gentilmente di menzionare e accreditare il sottoscritto.
Apprezzo molto il riconoscimento per il lavoro svolto!

Spero che ti piaccia usare AudioLinee!
