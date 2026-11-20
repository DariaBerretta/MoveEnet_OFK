# moveEnetFlow — guida all'uso

> Ambito: documentazione generale di `moveEnetFlow/`; ExpA ristrutturato nella branch `experiment-cleanup`. Alcune sezioni degli altri esperimenti descrivono ancora la versione storica.
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

### 3.3 Experiment A — test minimo

La nuova implementazione si trova in `experiments/ExpA/`.

```bash
python3 experiments/ExpA/scripts/run_experiment.py --dataset dhp19 --dry-run
python3 -m unittest discover -s experiments/ExpA/tests -v
```

Per le configurazioni, il runner completo e gli output, consultare
[`experiments/ExpA/README.md`](experiments/ExpA/README.md).

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

### 5.1 ExpA — accuracy vs neural-network frequency

ExpA è stato ristrutturato. Usare esclusivamente
`experiments/ExpA/scripts/run_experiment.py` per nuove esecuzioni.

- Dataset: eH36M, DHP19, Move-IIT-HPE.
- Rate rete: 2, 5, 10, 20, 50, 100, 200 Hz per metodi event-based;
  2, 5, 10, 20, 50 Hz per OpenPose/YOLOPose.
- Output 200 Hz; optical flow di MoveEnetOFK a 200 Hz.
- ROI: 20 (eH36M), **10 (DHP19)**, 35 (Move-IIT-HPE); of_gain=1.0.
- Risultati sotto `/data/MoveEnet_OFK_results/experiment_cleanup/ExpA/`.
- I notebook specialistici rimasti in `ExpA/analysis/validation`
  sono riferimenti diagnostici e non i runner ufficiali.

Consultare [`experiments/ExpA/README.md`](experiments/ExpA/README.md).

### 5.2 ExpB — accuratezza rispetto al periodo optical flow

**Domanda:** come cambia la qualità di MoveEnetOFK al variare di `flow_period` con `net_period` fissato?

Runner: `experiments/expB_accuracy_vs_flow_period/run_experiment_b.sh` (usa `shared/run_movenet_grid.sh` con profilo `expB`). Configurazione iniziale: eH36M, `net_period=0.01 s`, `flow_period=0.01,0.02 s`, solo **MoveEnetOFK** (senza baseline MoveEnet only).

```bash
bash experiments/expB_accuracy_vs_flow_period/run_experiment_b.sh \
  --dataset eh36m --net_period 0.05 \
  --flow_periods 0.005,0.01,0.02 \
  --binary "$PWD/build/moveEnetOFK_offline"
```

### 5.3 ExpA — Move-IIT-HPE

Le configurazioni sono in `experiments/ExpA/config/moveiit.yaml`.
Il GT 2D Vicon è tuttora soggetto alla verifica del mapping anatomico
e dell'offset temporale; le analisi in `ExpA/analysis/validation/`
sono mantenute per confronto, non come protocollo definitivo.

### 5.4 Analisi ExpA/ExpB

Per nuove acquisizioni ExpA, usare:

```bash
python3 experiments/ExpA/analysis/accuracy_metrics.py --run-dir <RUN_DIR>
python3 experiments/ExpA/visualization/plot_accuracy.py --run-dir <RUN_DIR>
```

Le analisi Move-IIT e di confronto articolare sono in
`experiments/ExpA/analysis/validation/` e restano diagnostiche.
Per ExpB resta il notebook specifico sotto
`experiments/expB_accuracy_vs_flow_period/notebooks/`.

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

**Nuovo ExpA:** risultati centralizzati in
`/data/MoveEnet_OFK_results/experiment_cleanup/ExpA/<run_id>/`.
Impostare `HPE_RESULTS_ROOT` per cambiare la radice; un run conserva
`config.yaml`, `metadata.json`, `manifest.csv`, `excluded.csv`, `audit.jsonl`,
`predictions/`, `logs/`, `telemetry/`, `metrics/` e `figures/`.

Lo schema delle predizioni è
`predictions/<dataset>/<method>/nn_<Hz>Hz/<sample_id>__<hash>.csv`.
I risultati precedenti in cartelle come `MOFK_dhp19_full_test/`,
`OP_h36m_full_test/`, `EPP_dhp19_full_test/` e `Final_ROI_Gain1/`
sono **storici**, non devono essere cancellati né uniti ai nuovi output.

**ExpB, Latency, Energy ed ExpFinal** conservano temporaneamente i loro
output storici fino alla rispettiva ristrutturazione.

## 9. Video e strumenti di debug

| Strumento | Uso |
| --- | --- |
| `experiments/ExpA/visualization/movenet_video.sh` | Video MoveEnetOFK/MoveEnet-only con GT; eH36M e DHP19. |
| `experiments/ExpA/visualization/rgb_pose_video.sh` | RGB, GT e predizione OpenPose/YOLOPose (`POSE_METHOD=openpose` o `POSE_METHOD=yolo`). |
| `experiments/ExpA/visualization/eventpointpose_video.sh` | EPP rolling FIFO, overlay GT e metriche opzionali. |
| `experiments/ExpA/preprocessing/eventpointpose_generate_hpm.sh` | Maschere hot pixel, **solo** per analisi raw. DHP19 già preprocessato non deve essere filtrato nuovamente. |
| `utils/of_sanity/` e `utils/of_gui/` | Diagnostica ottica e viewer interattivo. |

Gli script video scrivono per default in
`/data/MoveEnet_OFK_results/experiment_cleanup/ExpA/visualization/`,
ma non sostituiscono il runner ExpA usato per le metriche del paper.
Alcuni video utilizzano di default binari in `build2/`; passare `--binary`
se sono compilati in `build/`.

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
