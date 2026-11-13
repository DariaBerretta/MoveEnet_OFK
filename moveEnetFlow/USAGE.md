# moveEnetFlow — guida all'uso

> Ambito: cartella `moveEnetFlow/` della repository `MoveEnet_OFK`, branch `main`, commit `af3a1fc420a7689fe47416a21c6f2a39bb64bfbf` (26 ottobre 2026). I percorsi sono quelli configurati nel codice: possono richiedere adattamenti su altre macchine.
>
> Nei comandi seguenti, la directory corrente è **`moveEnetFlow/`**, salvo indicazione diversa.

## 1. Scopo e componenti

La repository esegue **stima offline di posa umana 2D (13 joint)** su registrazioni event-based o video RGB. Separa, ove previsto, la frequenza delle inferenze dalla frequenza di aggiornamento/salvataggio della posa. Le pipeline producono predizioni da confrontare con il ground truth (GT) e supportano esperimenti di accuratezza, latenza, consumo energetico e tempo di esecuzione.

| Metodo / eseguibile | Input | Funzionamento | Dipendenze specifiche |
| --- | --- | --- | --- |
| **MoveEnet only** / `moveEnetOFK_offline --moveenet_only` | Eventi (`data.log`) | Converte eventi in superficie EROS, invia le immagini a MoveNet; non applica l'aggiornamento di velocità da optical flow. È il baseline event-based. | `hpe-core` + checkpoint MoveNet + server Python/YARP di MoveNet. |
| **MoveEnetOFK** / `moveEnetOFK_offline` | Eventi (`data.log`) | Rilevazioni MoveNet intermittenti; velocità articolari da **SAE + optical flow locale**; fusione con filtro di Kalman multi-joint, con opzione di compensazione della latenza. | Come sopra, più componenti `hpe-core` per velocità e fusione. |
| **MoveEnetOFK con MPJPE integrato** / `moveEnetOFK_offline_mpjpe` | Eventi + GT | Variante della pipeline che calcola anche MPJPE nel processo C++, oltre agli output ordinari. | GT in formato YARP skeleton. |
| **OpenPose** / `OpenPose_offline` | Video RGB (`.mp4`) | Inferenza con **libreria C++ OpenPose**; mantiene l'ultima posa valida tra inferenze (zero-order hold, ZOH). | OpenPose C++ e relativi pesi. |
| **YOLOPose** / `YoloPose_offline` | Video RGB (`.mp4`) | Inferenza Ultralytics YOLO-Pose tramite server Python/YARP; output ZOH. | `model_mounts/YoloPose/`, Ultralytics, Python/YARP. |
| **EventPointPose (EPP)** / `eventPointPose_offline` | Eventi DHP19, camere 2/3 | Il server Python mantiene una **FIFO scorrevole di 7.500 eventi**, applica RasEPC, campiona 2.048 punti ed esegue PointNet. Il C++ richiede pose a `net_period`; l'output usa ZOH. | `model_mounts/eventpointpose/PointNet/models/`, PyTorch e Python/YARP. |

**Flusso MoveEnetOFK:** `data.log` → EROS → MoveNet (posizione) + SAE → optical flow per joint (velocità) → filtro di Kalman → posa 2D → CSV/video. La pipeline EPP **non** usa l'optical flow o il filtro di Kalman di MoveEnetOFK.

Sorgenti principali: `moveEnetOFK_offline.cpp`, `moveEnetOFK_offline_mpjpe.cpp`, `OpenPose_offline.cpp`, `YoloPose_offline.cpp`, `eventPointPose_offline.cpp`. `moveEnetOFK_offline_v0.cpp` è una variante storica presente nel repository, **non compilata** dal `CMakeLists.txt` principale.

## 2. Parametri e convenzioni

Tutti i periodi sono espressi in **secondi**; la frequenza nominale è `f [Hz] = 1 / periodo [s]`.

| Parametro | Significato |
| --- | --- |
| `--net_period` | Intervallo tra richieste di inferenza alla rete. Esempio: `0.05` = 20 Hz. |
| `--flow_period` | Intervallo di aggiornamento optical flow/Kalman; riguarda MoveEnetOFK. Esempio: `0.005` = 200 Hz. |
| `--output_period` | Intervallo nominale di campionamento della posa per CSV/video, indipendente dal periodo di rete nei metodi che lo prevedono. |
| `--roi` | Dimensione della regione locale impiegata nella stima della velocità articolare. |
| `--pu`, `--muD`, `--muV` | Incertezze del filtro di Kalman: processo, misura di posizione, misura di velocità. |
| `--use_lc true` | Abilita la compensazione della latenza in MoveEnetOFK. Il binario la abilita per impostazione predefinita. |
| `--moveenet_only` | Seleziona il baseline senza aggiornamento di velocità OFK. |
| `--dhp19` / `--moveiit` | Seleziona le convenzioni/dimensioni di input specifiche del dataset nel binario MoveEnet; sono mutuamente esclusivi. |
| `--device` | Dispositivo d'inferenza (`cuda:0`, ecc.; alcuni eseguibili supportano `cpu`). |
| `--no_csv`, `--no_video` | Disabilitano rispettivamente la scrittura delle predizioni e del video, quando supportati dall'eseguibile. |

**Ordine dei 13 joint nei CSV:** `0 head`, `1 shoulder_right`, `2 shoulder_left`, `3 elbow_right`, `4 elbow_left`, `5 hip_left`, `6 hip_right`, `7 wrist_right`, `8 wrist_left`, `9 knee_right`, `10 knee_left`, `11 ankle_right`, `12 ankle_left`.

**Formato base delle predizioni:** `timestamp,latency,joint0_x,joint0_y,...,joint12_x,joint12_y`. La riga contiene timestamp relativo alla registrazione, latenza associata alla posa e coordinate in pixel. Alcune opzioni aggiungono colonne (ad esempio velocità): non assumere sempre esattamente 28 colonne. **Le predizioni non sono il GT**; le metriche si calcolano negli script/notebook di analisi.

## 3. Installazione ed esecuzione

### 3.1 Requisiti

- Linux, CMake (C++17), YARP con name server, OpenCV, `event-driven` e `hpe-core` con pacchetti CMake installati.
- **OpenPose C++ + GFlags** sono richiesti dall'attuale `CMakeLists.txt` principale anche se si intende compilare soltanto MoveEnetOFK: `find_package(OpenPose REQUIRED)` non è condizionale.
- Python, binding YARP e dipendenze dei modelli: MoveNet (`hpe-core/example/movenet/`), Ultralytics per YOLOPose, PyTorch/PointNet per EPP. `requirements.txt` si trova nella **radice della repository**, non in `moveEnetFlow/`.
- Checkpoint MoveNet di eH36M/DHP19, modelli OpenPose e pesi YOLO/EPP. La repository contiene i pesi YOLO/EPP in `model_mounts/`; i due checkpoint MoveNet sono referenziati da percorsi di `hpe-core`.
- Per i benchmark energetici: host con **Intel RAPL**, GPU NVIDIA con contatore cumulativo **NVML**, Python host con `pynvml`, e Docker configurato come nei runner. Questi benchmark **non sono portabili senza modifiche** su qualunque host.

### 3.2 Compilazione

```bash
# Dalla cartella moveEnetFlow/
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Eseguibili attesi:
# build/moveEnetOFK_offline
# build/moveEnetOFK_offline_mpjpe
# build/OpenPose_offline
# build/YoloPose_offline
# build/eventPointPose_offline
```

Prima di avviare una pipeline che usa YARP, verificare `yarp where`; se non esiste un name server, avviare `yarpserver` nell'ambiente in cui sono raggiungibili le porte YARP. I binari di MoveNet e YOLOPose avviano i rispettivi server Python; evitare processi concorrenti sulle stesse porte. Il binario MoveNet esegue anche una pulizia dei processi `movenet_online.py` esistenti all'avvio: **non lanciarlo mentre altri esperimenti MoveNet sono in uso**.

**Docker è opzionale per i normali esperimenti di accuratezza** se tutte le dipendenze sono installate nativamente. Gli script assumono tuttavia spesso i percorsi del container (ad esempio `/workspace/moveEnetFlow/build`): usare `--binary`, `--checkpoint_path`, `--data_root` o altre opzioni disponibili per correggerli. Il file `docker/docker-compose.yml` monta la repository in `/workspace` e `${HOME}/data` in `/data`.

### 3.3 Prova minima: una registrazione DHP19

```bash
bash experiments/ExpA/expA_accuracy_vs_network_period/run_experiment_a.sh \
  --dataset dhp19 \
  --binary "$PWD/build/moveEnetOFK_offline" \
  --data_file /data/dhp19_testing_set_S13toS17/S13_1_1/ch2dvs/data.log \
  --net_period 0.05 --flow_period 0.005 \
  --checkpoint_path /usr/local/src/hpe-core/example/movenet/models/dhp19_allcams_e33_valacc0.87996.pth
```

Questo comando esegue **MoveEnetOFK e MoveEnet only** sulla stessa registrazione. Per verificare subito i parametri di uno script: `bash <script> --help`. **Attenzione:** alcune descrizioni di default stampate da `--help` sono rimaste storiche; i valori effettivi sono quelli assegnati nelle variabili all'inizio dello script.

## 4. Dataset e ground truth

| Dataset | Percorso input predefinito | Risoluzione | Uso |
| --- | --- | --- | --- |
| **eH36M** | `/data/eh36m_testing_set_S9S11/events/<sequenza>/ch0dvs/data.log` e `/data/eh36m_testing_set_S9S11/rgb/<sequenza>.mp4` | 640×480 | Eventi per MoveEnet/OFK; RGB per OpenPose/YOLOPose. |
| **DHP19** | `/data/dhp19_testing_set_S13toS17/<sequenza>/ch*dvs/data.log` | 346×260 | MoveEnet/OFK; EPP usa soltanto `ch2dvs` e `ch3dvs`. |
| **move-iit-hpe** | `/data/move-iit-it_testing/{luna,simon}/<sequenza>/atis-s/data.log` | 1280×720 | Esperimenti dedicati MoveEnet/OFK, con GT Vicon 2D associato. |

I notebook utilizzano GT della relativa sequenza. Per eH36M è previsto, dove presente, un log `ch0GT200Hzskeleton/data.log`; per DHP19 si usano i log GT in formato `SKLT` secondo `hpe-core`. In move-iit-hpe il protocollo dedicato richiede file `GT/<soggetto>/<sequenza>_joint_projections.csv`.

Il tool `utils/gt_XXXhz_interpolation.py` può creare log `ch0GT200Hzskeleton` interpolando quelli a 50 Hz: `python3 utils/gt_XXXhz_interpolation.py --dataset-root <dir> --source-hz 50 --target-hz 200`. **Non** sostituisce un GT mancante con una nuova misura: interpola dati esistenti.

## 5. Esperimenti di accuratezza

I runner pubblici di ExpA/ExpB sono piccoli **wrapper**: la logica comune è in `experiments/shared/`. Per nuove griglie modificare/riusare il runner condiviso, non duplicarne il contenuto.

### 5.1 ExpA — accuratezza rispetto alla frequenza di rete

**Domanda:** come variano **MPJPE e PCK** quando cambia `net_period`? Per MoveEnetOFK il flow è controllato separatamente; sono disponibili baseline MoveEnet only, OpenPose, YOLOPose ed EPP.

| Metodo | Runner | Configurazione effettiva iniziale | Dataset |
| --- | --- | --- | --- |
| MoveEnetOFK + MoveEnet only | `experiments/ExpA/expA_accuracy_vs_network_period/run_experiment_a.sh` | `net=0.01 s`; `flow=0.005,0.01 s`; output `0.005 s` | DHP19 di default; `--dataset eh36m` disponibile. |
| OpenPose | `experiments/ExpA/OP_expA_accuracy_vs_network_period/OP_run_experiment_a.sh` | `net=0.02,0.05,0.1,0.2,0.5 s`; output `0.005 s` | eH36M RGB. |
| YOLOPose | `experiments/ExpA/YOLO_expA_accuracy_vs_network_period/YOLO_run_experiment_a.sh` | Come OpenPose | eH36M RGB. |
| EventPointPose | `experiments/ExpA/EPP_expA_accuracy_vs_network_period/EPP_run_expA.sh` | **Attualmente solo `net=0.005 s`**; output `0.005 s`; FIFO 7.500, 2.048 punti | DHP19, camere 2/3. |

Esempi (da `moveEnetFlow/`):

```bash
# OFK + baseline: più periodi rete, optical flow fisso a 200 Hz
bash experiments/ExpA/expA_accuracy_vs_network_period/run_experiment_a.sh \
  --dataset eh36m --network_periods 0.02,0.05,0.1,0.2 \
  --flow_period 0.005 --binary "$PWD/build/moveEnetOFK_offline"

# OpenPose (RGB)
bash experiments/ExpA/OP_expA_accuracy_vs_network_period/OP_run_experiment_a.sh \
  --net_period 0.05 --binary "$PWD/build/OpenPose_offline"

# YOLOPose (RGB)
bash experiments/ExpA/YOLO_expA_accuracy_vs_network_period/YOLO_run_experiment_a.sh \
  --net_period 0.05 --binary "$PWD/build/YoloPose_offline"

# EPP (DHP19, FIFO scorrevole)
bash experiments/ExpA/EPP_expA_accuracy_vs_network_period/EPP_run_expA.sh \
  --net_period 0.05 --binary "$PWD/build/eventPointPose_offline"
```

OpenPose e YOLOPose utilizzano video RGB e, **nei rispettivi runner ExpA**, richiedono un dispositivo GPU (un `--device cpu` viene rifiutato dal wrapper). I runner hanno `--data_file`, `--data_root`, `--raw_dir`, `--log_dir` per limitare i dati o cambiare destinazione. Per EPP sono presenti anche `--model_path`, `--epp_script`, `--dump_root`, `--skip_existing`, `--max_packets`.

### 5.2 ExpB — accuratezza rispetto al periodo optical flow

**Domanda:** come cambia la qualità di MoveEnetOFK al variare di `flow_period` con `net_period` fissato?

Runner: `experiments/expB_accuracy_vs_flow_period/run_experiment_b.sh` (usa `shared/run_movenet_grid.sh` con profilo `expB`). Configurazione iniziale: eH36M, `net_period=0.01 s`, `flow_period=0.01,0.02 s`, solo **MoveEnetOFK** (senza baseline MoveEnet only).

```bash
bash experiments/expB_accuracy_vs_flow_period/run_experiment_b.sh \
  --dataset eh36m --net_period 0.05 \
  --flow_periods 0.005,0.01,0.02 \
  --binary "$PWD/build/moveEnetOFK_offline"
```

### 5.3 ExpA move-iit-hpe

`experiments/ExpA/expA_accuracy_vs_network_period/run_moveiit_expa.sh` è un **protocollo separato**, non un wrapper della griglia comune. Seleziona automaticamente le **59 registrazioni abbinate al GT Vicon** dei soggetti `luna` e `simon`, richiedendo che tutte siano presenti. Esegue MoveEnet only e OFK per `net_period = 0.01,0.02,0.05,0.1,0.2,0.5 s`, `flow=output=0.005 s`, `ROI=35`, configurazione **gain=1** e risoluzione ATIS nativa 1280×720. Controlla la configurazione tramite hash e salta solo i run già completi.

```bash
# Variabili personalizzabili via ambiente: DATA_ROOT, BINARY, CHECKPOINT, RESULTS, DEVICE
BINARY="$PWD/build/moveEnetOFK_offline" \
  bash experiments/ExpA/expA_accuracy_vs_network_period/run_moveiit_expa.sh
```

Questo protocollo è vincolato a una versione del binario compatibile con le sue impostazioni; **non** mescolare i risultati con run ottenuti usando un eseguibile differente.

### 5.4 Analisi ExpA/ExpB

| Notebook | Uso |
| --- | --- |
| `ExpA/expA_accuracy_vs_network_period/notebooks/ExperimentA_MPJPE_PCK_vs_network_period.ipynb` | MoveEnet only/OFK: errore e accuratezza rispetto al periodo rete. |
| `ExpA/OP_expA_accuracy_vs_network_period/notebooks/OP_ExperimentA_MPJPE_PCK_vs_network_period.ipynb` | ExpA OpenPose. |
| `ExpA/YOLO_expA_accuracy_vs_network_period/notebooks/YOLOPose_MPJPE_PCK_vs_network_period.ipynb` | ExpA YOLOPose. |
| `ExpA/EPP_expA_accuracy_vs_network_period/Notebooks/EPP_MPJPE_PCK_vs_network_period.ipynb` | ExpA EventPointPose. |
| `ExpA/Comparison/ExperimentA_Method_Comparison_SELECTABLE_MODELS.ipynb` | Confronto selezionabile tra metodi e dataset; mantiene separate le varianti di flow e deduplica i baseline ripetuti. |
| `ExpA/Comparison/ExperimentA_Joint_MPJPE_Comparison.ipynb` | Confronto degli errori **per joint**. |
| `expB_accuracy_vs_flow_period/notebooks/experimentB_flow_period_analysis_SHARED_HEATMAPS.ipynb` | ExpB: analisi/heatmap con periodo optical flow variabile. |

I percorsi della tabella sono **relativi a `experiments/`**. Nei notebook, verificare le celle di configurazione (`RESULTS_ROOT`, dataset, metodi, filtri `np_*`/`fp_*` e cartella di esportazione) **prima** di eseguire tutte le celle. Il notebook di confronto crea una directory distinta sotto `MoveEnet_OFK_results/Comparison/` per ciascuna selezione.

**Attenzione EPP:** il runner EPP attuale salva per default in `EPP_dhp19_full_test/results/eng_200Hz/raw`, mentre il notebook di confronto generale cerca `EPP_dhp19_full_test/results/raw`. Occorre allineare esplicitamente il percorso nel notebook oppure passare `--raw_dir` al runner; i risultati non vengono trovati automaticamente fra queste due directory.

## 6. Benchmark di latenza

Runner: `experiments/ExpLatency/run_latency_experiment.sh`.

- **Protocollo:** 12 campioni eH36M (`H01–H12`) e 12 DHP19 (`D01–D12`), definiti in `ExpLatency/manifests/`. eH36M confronta MoveEnet only, OFK, OpenPose, YOLOPose; DHP19 confronta MoveEnet only, OFK, EPP. Gli ordini dei modelli sono bilanciati; predefiniti `net=0.02 s` e `flow=0.005 s` (dove applicabile).
- **Misura:** un CSV di latenza **per richiesta d'inferenza**, distinto dai CSV di predizione. Il runner registra anche durata del processo, temperature GPU, stato del run e configurazione.
- **Ambiente:** MoveEnet/OpenPose fuori da `/workspace/.nbvenv`, YOLO/EPP dentro; regolabile con `VENV_DIR`.

```bash
bash experiments/ExpLatency/run_latency_experiment.sh --preflight-only
bash experiments/ExpLatency/run_latency_experiment.sh \
  --dataset h36m --sample H01 --session prova_latenza --cooldown 0
# Ripresa della stessa sessione:
bash experiments/ExpLatency/run_latency_experiment.sh \
  --dataset h36m --sample H01 --session prova_latenza --resume
```

Output: `/data/MoveEnet_OFK_results/Latency/<session>/run_manifest.csv`, `metadata.txt` e `<dataset>/<sample>/<model>/{latency.csv,run.log}` (per EPP anche `epp_server.log`). Notebook: `ExpLatency/Notebooks/Latency_Analysis.ipynb`.

**Incompatibilità da verificare prima dell'uso:** il runner passa `--latency_csv` a `moveEnetOFK_offline`, ma **il sorgente principale attuale non implementa la scrittura di quel file** (l'opzione si trova nella variante storica `moveEnetOFK_offline_v0.cpp`, non compilata). I run MoveEnet/OFK possono quindi non produrre righe di latenza e risultare `NO_LATENCY_ROWS`. Correggere/ricompilare il binario o usare una versione verificata prima di interpretare questo benchmark.

## 7. Benchmark energetici e runtime

### 7.1 Sweep energia × frequenza (`ExpEnergy`)

`experiments/ExpEnergy/run_energy_experiment.sh` esegue una griglia comune di `net_period = 0.02,0.05,0.1,0.2,0.5 s` sui campioni dei manifest di latenza. Tiene `flow_period=0.005 s` per OFK. Per il confronto di questa griglia, i periodi nominali di output **non sono uniformi**: MoveEnet/OFK e EPP `0.005 s`, OpenPose/YOLOPose `0.02 s`.

```bash
# Sul computer host, con container moveEnetOFK già avviato
bash experiments/ExpEnergy/run_energy_experiment.sh --preflight-only
bash experiments/ExpEnergy/run_energy_experiment.sh \
  --dataset h36m --sample H01 --net_period 0.05 \
  --session test_energy --cooldown 20
```

Output host: `~/data/MoveEnet_OFK_results/Energy/<session>/run_manifest.csv`, `metadata.txt`, e `<dataset>/<sample>/np_<period>/<model>/{energy.csv,run.log,OK}`. `--resume` riusa soltanto i run con marker `OK`; `--idle_baseline`, `--dry-run` e `--continue_on_error` sono disponibili. Gli script si eseguono **sull'host**, ma lanciano le pipeline attraverso `docker exec`.

### 7.2 Protocollo energetico finale a punti operativi fissi

`experiments/ExpEnergy/run_energy_accuracy_final.sh` confronta metodi in **punti operativi prestabiliti** (non esegue lo sweep). Tutti hanno output nominale a **200 Hz**:

| Dataset | MoveEnet only | MoveEnetOFK | Altri metodi |
| --- | --- | --- | --- |
| eH36M | rete **50 Hz** | rete **5 Hz**, OF **200 Hz** | OpenPose **50 Hz**, YOLOPose **50 Hz** |
| DHP19 | rete **200 Hz** | rete **5 Hz**, OF **200 Hz** | EventPointPose **200 Hz** |

Il protocollo usa gli stessi **12+12 campioni** dei manifest e **5 ripetizioni per coppia campione–metodo**, con rotazione dell'ordine dei modelli e cooldown predefinito di 20 s: **84 coppie, 420 misure**.

```bash
# Eseguire sull'host, dopo aver verificato i contatori energia e il container
bash experiments/ExpEnergy/run_energy_accuracy_final.sh --preflight-only
bash experiments/ExpEnergy/run_energy_accuracy_final.sh \
  --dataset both --repeats 5 --session energy_accuracy_final_5hz_ofk
# Se interrotto:
bash experiments/ExpEnergy/run_energy_accuracy_final.sh \
  --dataset both --repeats 5 --session energy_accuracy_final_5hz_ofk --resume
```

Output host: `~/data/MoveEnet_OFK_results/Energy/<session>/energy_measurements.csv`, `run_manifest.csv`, `metadata.txt`, `runs/*.log`. Le misure comprendono energia CPU package (**Intel RAPL**), energia dell'intera GPU (**NVML**), somma `measured_compute_energy_j`, potenza media e `wall_seconds`. **Non rappresentano l'energia dell'intero computer né la sola energia attribuita al processo HPE**. Il tempo include avvio/caricamento/arresto del comando misurato, ma non il cooldown tra run.

### 7.3 ExpFinal — unione energia, accuratezza e runtime

`experiments/ExpFinal/finish_experiment.sh` **non ripete le misure energetiche**. Verifica la sessione finale da 420 misure, conserva snapshot/provenienza, riesegue le **84 inferenze necessarie alle predizioni** e associa accuratezza, energia e runtime con controlli di configurazione/hash. **Richiede il container Docker** e va lanciato sull'host.

```bash
bash experiments/ExpFinal/finish_experiment.sh \
  --session energy_accuracy_final_5hz_ofk --prepare-only

bash experiments/ExpFinal/finish_experiment.sh \
  --session energy_accuracy_final_5hz_ofk --resume

# Solo rigenerazione analisi da risultati Final esistenti:
bash experiments/ExpFinal/finish_experiment.sh \
  --session energy_accuracy_final_5hz_ofk --analysis-only
```

Default output: `~/data/MoveEnet_OFK_results/Final/<session>/`. È separato dai CSV Energy originali. Contiene `source_energy/`, `audit_energy.json`, `plan.json`, `preflight.json`, `provenance/`, `attempts/`, `predictions/`, `metrics/`, `tables/`, `figures/`, `completion.json` e `ANALYSIS_README.txt`. Tabelle principali: `energy_validated_runs.csv`, `energy_runtime_by_sample.csv`, `energy_runtime_summary.csv`, `sequence_runtime_by_sample.csv`, `accuracy_by_sample.csv`, `energy_accuracy_by_sample.csv`, `energy_accuracy_summary.csv` e diagnostica di copertura. Le figure vengono esportate in PNG/SVG. Notebook di lettura: `ExpFinal/notebooks/`.

**Metriche usate da ExpFinal:** GT interpolato linearmente ai timestamp delle predizioni (le predizioni **non** sono interpolate); **MPJPE** in pixel; **PCK@0.2/0.4** con scala del torso GT tra joint 1 (spalla destra) e joint 5 (anca sinistra), in percentuale. L'accuratezza aggregata principale pesa i singoli punti valutati; le medie per sequenza sono esportate separatamente. L'energia media è calcolata prima sulle cinque ripetizioni di ogni sequenza e poi sulle sequenze; intervalli bootstrap su sequenze, non sulle ripetizioni.

**Runtime:** `RTF = tempo totale di esecuzione / durata temporale dei dati in input`; `RTF < 1` indica throughput offline mediamente più rapido del tempo registrato, **non** prova una latenza inferiore al vincolo real-time. I tempi normalizzati per slot di output sono quantità derivate, non nuove misure di inferenza.

### 7.4 move-iit-hpe: energia e real-time factor

`experiments/ExpEnergy/run_moveiit_energy_rtf.py` usa **12 campioni** e tre condizioni: MoveEnet only **50 Hz**, MoveEnet only **5 Hz**, MoveEnetOFK **5 Hz + OF 200 Hz**; 5 ripetizioni predefinite. È un runner **host+Docker**, con misure RAPL/NVML e verifica dei risultati ExpA move-iit. Richiede un file di campioni passato con `--samples-csv`: il default punta a `ExpEnergy/moveiit_energy_samples.csv`, **non presente nel commit esaminato**. Prepararlo/fornirlo prima dell'esecuzione.

```bash
python3 experiments/ExpEnergy/run_moveiit_energy_rtf.py \
  --samples-csv /percorso/moveiit_energy_samples.csv \
  --preflight-only

python3 experiments/ExpEnergy/analyze_moveiit_energy_rtf.py \
  --session-dir /percorso/alla/sessione --strict
```

Analisi: `ExpEnergy/Notebooks/MoveIIT_Energy_RTF_Analysis.ipynb`; tabelle prodotte dallo script: `energy_runtime_by_sample.csv`, `energy_runtime_dataset_summary.csv`, `paired_differences.csv` e alias `rtf_energy_*.csv`. In `ExpEnergy/Notebooks/` si trovano anche `Energy_Accuracy_Frequency_Analysis.ipynb`, `Energy_Accuracy_Tradeoff.ipynb` e `Sequence_Runtime_Analysis.ipynb` per i confronti energia/accuratezza/frequenza/runtime. Per aggregazioni da CLI, `ExpEnergy/analyze_energy_accuracy.py` prende `--energy`, `--accuracy`, `--out`. Verificare i percorsi input delle prime celle: alcuni notebook puntano a sessioni specifiche.

## 8. Output: struttura e nomi

La maggior parte degli esperimenti scrive fuori dalla repository. Nei runner ExpA/ExpB e video la variabile `MOVENET_RESULTS_ROOT` sostituisce la root predefinita **`/data/MoveEnet_OFK_results`**. Il mapping Docker predefinito è `${HOME}/data` (host) ↔ `/data` (container). I runner Energy, Latency, Final e move-iit usano **variabili/opzioni proprie** (`RESULTS_BASE`, `--results_base`, `RESULTS`, `--results-dir`, ecc.).

| Esperimento | Cartella risultati predefinita (relativa alla root `/data/MoveEnet_OFK_results`) |
| --- | --- |
| ExpA/ExpB MoveEnet, eH36M | `MOFK_eh36m_full_test/results/raw/` |
| ExpA/ExpB MoveEnet, DHP19 | `MOFK_dhp19_full_test/results/raw/` |
| ExpA OpenPose / YOLOPose | `OP_h36m_full_test/results/raw/` / `YOLO_h36m_full_test/results/raw/` |
| ExpA EventPointPose | `EPP_dhp19_full_test/results/eng_200Hz/raw/` |
| Video OFK / OpenPose / YOLOPose | `MOFK_<dataset>_full_test/videos/` / `OP_h36m_full_test/videos/` / `YOLO_h36m_full_test/videos/` |
| Video EPP rolling | `EPP_dhp19_full_test/videos/rolling/` |
| ExpLatency | `Latency/<session>/` |
| ExpEnergy | `Energy/<session>/` (runner eseguito sull'host: default `~/data/...`) |
| ExpFinal | `Final/<session>/` (host) |
| ExpA move-iit | `Final_ROI_Gain1/Accuracy/moveiit/ExpA/` |

Schema dei risultati della griglia MoveEnet:

```text
MOFK_dhp19_full_test/results/
├── raw/
│   └── fp_0p005/
│       └── np_0p05/
│           ├── S13_1_1__ch2dvs_moveenet_ofk_np_0p05_fp_0p005.csv
│           └── S13_1_1__ch2dvs_moveenet_only_np_0p05_fp_0p005.csv
└── logs/
    ├── S13_1_1__ch2dvs_moveenet_ofk_np_0p05_fp_0p005.log
    └── S13_1_1__ch2dvs_moveenet_ofk_np_0p05_fp_0p005_gpu.csv
```

- `np_0p05` significa `net_period=0.05 s` (20 Hz); `fp_0p005` significa `flow_period=0.005 s` (200 Hz). Il punto decimale è sostituito con `p`.
- ExpA OpenPose/YOLOPose: `raw/np_<period>/<sequenza>_<metodo>_np_<period>.csv`; i log stanno in `results/logs/`.
- ExpA EPP: `results/eng_200Hz/raw/np_<period>/<sequenza>__<camera>_net_flow_np_<period>.csv` e log C++/server in `results/eng_200Hz/logs/`. Il nome storico `net_flow` **non indica** che EPP usi l'optical flow.
- Output video: script dedicati salvano CSV di predizione, MP4 ed eventuale overlay GT, log e talvolta CSV GPU nella cartella `videos/`.
- CSV predizioni, CSV latenza e CSV energia sono **file differenti**: non sostituirli l'uno con l'altro nell'analisi.

## 9. Video e strumenti di debug

| Strumento | Cosa fa | Utilizzo |
| --- | --- | --- |
| `experiments/ExpA/expA_accuracy_vs_network_period/MOFK_video.sh` | Video di una sequenza con predizione MoveEnetOFK e/o baseline, overlay GT. | `--dataset eh36m|dhp19`, `--mode ofk|moveenet_only|both`, `--data_file`, `--output_dir`, `--print_config`. |
| `experiments/ExpA/OP_expA_accuracy_vs_network_period/OpenPose_video.sh` | Video RGB, GT e predizione OpenPose. | `--data_file`, `--gt_root`, `--output_dir`, `--net_period`. |
| `experiments/ExpA/YOLO_expA_accuracy_vs_network_period/YOLO_video.sh` | Video RGB, GT e predizione YOLOPose. | Stesse opzioni principali. |
| `experiments/ExpA/EPP_expA_accuracy_vs_network_period/EPP_video.sh` | Video EPP FIFO scorrevole; eventuale overlay GT e metriche; ricostruisce il video **dopo** l'inferenza. | `--data_file`, `--net_period`, `--output_dir`, `--prediction_only`, `--skip_metrics`. |
| `experiments/ExpA/EPP_expA_accuracy_vs_network_period/EPP_video_no_rolling.sh` | Ablazione EPP con finestre indipendenti (senza FIFO rolling). | Richiede binario e server Python `*_no_rolling` **non inclusi/compilati** nel checkout esaminato. |
| `experiments/ExpA/EPP_expA_accuracy_vs_network_period/EPP_generate_hpm.sh` | Genera maschere hot pixel PNG/NPY e coordinate CSV per registrazione. | `--data_root`, `--mask_root`, `--threshold`, `--force`; il binario `eventPointPose_hotpixel_mask` **non** è definito dal CMake principale. |
| `utils/of_sanity/` + `utils/optical_flow_sanity.cpp` | Diagnostica OFK standalone: EROS, pose rilevate/fuse, vettori optical flow, ROI. | Compilazione con il suo CMake, poi `--data_file ... --output_video ...`. |
| `utils/of_gui/` | Viewer interattivo locale del flusso OFK con engine C++ e server web Python. | Compilare l'engine nel sottoprogetto; `python3 utils/of_gui/server.py --host 127.0.0.1 --port 8765`. |
| `utils/run_batch_processing.sh` | Elaborazione batch generica più vecchia, con parametri propri. | Preferire i runner ExpA/ExpB per esperimenti riproducibili. |
| `analysis/epp_coordinate_sweep.py` | Verifica trasformazioni di coordinate EPP confrontando predizioni e GT CSV. | Controllare l'intestazione CSV: accetta convenzioni specifiche, non tutti gli output nuovi. |
| `analysis/visualize_hpe_results.ipynb`, `utils/imgs_paper.ipynb`, `utils/MoveEnetOFK_image_for_pipeline.ipynb` | Visualizzazioni e figure per analisi/documentazione. | Notebook di post-processing: controllare le celle iniziali per i path e le opzioni di salvataggio. |

Per una verifica non invasiva dell'optical flow:

```bash
cmake -S utils/of_sanity -B utils/of_sanity/build
cmake --build utils/of_sanity/build -j
utils/of_sanity/build/optical_flow_sanity \
  --data_file /data/eh36m_testing_set_S9S11/events/<sequenza>/ch0dvs/data.log \
  --output_video /tmp/of_sanity.mp4
```

Le maschere hot pixel generate da `EPP_generate_hpm.sh` sono pertinenti alla modalità EPP **raw**: il protocollo ExpA corrente usa `already_filtered` e non deve riapplicare i filtri ai `data.log` già preprocessati.

Molti script video hanno come default un binario in `build2/`, mentre i runner di accuratezza usano `build/`: se è stato compilato solo `build/`, passare **`--binary "$PWD/build/<eseguibile>"`**.

## 10. Procedura consigliata e controlli

1. **Compilare** e verificare i pesi, il Python corretto e `yarp where`. Usare una sola pipeline per volta sulle porte YARP condivise.
2. **Prova su una singola registrazione:** `--data_file`, un solo `--net_period`, un solo `--flow_period` e, dove disponibile, `--raw_dir`/`--log_dir` dedicati.
3. **Griglia completa:** selezionare dataset, periodi, checkpoint e path; conservare gli script/comandi e le versioni dei binari. Evitare cartelle condivise tra configurazioni incompatibili.
4. **Controllare gli output:** CSV non vuoti, nomi `np_`/`fp_` corretti, assenza di errori nei `.log`, esistenza dei GT corrispondenti e distribuzione temporale dei timestamp.
5. **Analisi:** aprire il notebook specifico, aggiornare i percorsi, verificare sequenze escluse e criteri di aggregazione; esportare le figure in una cartella dedicata.
6. **Energia:** usare `--preflight-only`, GPU il più possibile libera, stessa macchina e stesso protocollo; non confrontare direttamente misure basate su hardware o periodi di output diversi senza dichiararlo.
7. **Risultati finali:** mantenere intatta la sessione Energy originale e usare `ExpFinal` per generare tabelle e figure riconciliate.

**Limiti noti della versione esaminata:** (i) ExpLatency richiede un output non implementato dal binario MoveEnet attualmente compilato; (ii) la directory EPP dell'ExpA differisce da quella attesa dal notebook generale di confronto; (iii) i componenti EPP `*_no_rolling` e l'eseguibile per le maschere hot pixel non sono target CMake correnti; (iv) lo script move-iit energetico attende un manifest CSV non versionato; (v) alcuni messaggi `--help` e default di `build`/`build2` non sono aggiornati. Verificare questi punti **prima** di considerare un esperimento riproducibile.

---

**Riferimenti nel repository:** [`experiments/README.md`](experiments/README.md), [`CMakeLists.txt`](CMakeLists.txt), [`experiments/shared/`](experiments/shared/), [`experiments/ExpLatency/manifests/`](experiments/ExpLatency/manifests/), [`experiments/ExpFinal/final_metrics.py`](experiments/ExpFinal/final_metrics.py), [`experiments/ExpFinal/final_analysis.py`](experiments/ExpFinal/final_analysis.py).
