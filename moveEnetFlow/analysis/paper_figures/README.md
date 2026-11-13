# MoveEnetOFK — figure per il paper (versione completa v2)

Questa versione **sostituisce lo starter che produceva soltanto una composizione di anteprima**. I file già presenti nello ZIP sono esempi con dati numerici autentici salvati nelle uscite dei notebook GitHub; non sono valori inventati o estratti da screenshot.

## Avvio rapido

Da `moveEnetFlow/analysis`:

```bash
unzip ~/Downloads/MoveEnetOFK_Paper_Figures_Complete_v2.zip
cd paper_figures
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt ipykernel
python generate_all_figures.py
```

Apri `Paper_Figures.ipynb` in VS Code con il kernel `.venv` e scegli **Run All**. Il notebook elenca e visualizza tutti i PNG separatamente.

**Non usare `render_preview.py` per produrre tutte le figure.** Quel comando è mantenuto solo per compatibilità e salva una singola anteprima.

## Cosa ottieni immediatamente

Il comando crea **14 figure** in `output/`, per un totale di **42 file grafici** (`.pdf`, `.svg`, `.png`) più un `.meta.json` ciascuno.

| Famiglia | Dati inclusi | Figure |
| --- | --- | ---: |
| Energy vs MPJPE/PCK | eH36M, DHP19, Move-IIT-HPE | 10 (6 singole, 3 coppie, 1 complessiva) |
| Network rate vs MPJPE/PCK — MoveEnet-only e OFK | Move-IIT-HPE, 2/5/10/20/50/100 Hz | 3 (2 singole + 1 coppia) |
| Energy (J / input s) vs RTF | Move-IIT-HPE | 1 |

I risultati per Move-IIT-HPE ablation sono quelli della tabella eseguita nel notebook `MoveIIT_ExpA_2DGT_Analysis.ipynb` (59 registrazioni, ROI35, OF 200 Hz, PCK 1/5, GT time offset 0 s **provvisorio**). I risultati per Move-IIT-HPE energia/RTF provengono dal notebook `MoveIIT_EventPointPose_Energy_RTF_Analysis.ipynb` (12 registrazioni, acquisizioni energia EPP e MoveEnet in sessioni diverse). Per eH36M/DHP19 energia-accuratezza si usa il subset validato di ExpFinal (12 sequenze eH36M, 10 DHP19, D09/D10 esclusi). Gli intervalli che non compaiono nelle tabelle cached **non vengono disegnati**.

## Come ottenere le altre immagini eH36M/DHP19

Lo ZIP **non contiene** le tabelle ExpA eH36M/DHP19 alle diverse frequenze o la tabella originale energia normalizzata/RTF eH36M/DHP19. Per generare quelle curve scientificamente valide, bisogna indicare i CSV locali degli esperimenti.

1. Cerca gli export nella macchina/container dove sono stati generati:

```bash
python generate_all_figures.py --scan /data/MoveEnet_OFK_results
```

2. Apri `sources.json` e sostituisci i `null` con i percorsi **assoluti** dei file CSV, per esempio:

```json
{
  "accuracy_rate": {
    "eh36m": "/data/.../eh36m_dataset_metrics.csv",
    "dhp19": "/data/.../dhp19_dataset_metrics.csv",
    "moveiit": "data/cached_moveiit_ablation_59.csv"
  },
  "energy_rtf": {
    "eh36m": "/data/.../energy_runtime_summary_valid_subset.csv",
    "dhp19": "/data/.../energy_runtime_summary_valid_subset.csv",
    "moveiit": "data/cached_moveiit_rtf_12.csv"
  }
}
```

Questa è solo la sezione rilevante, **non sostituire l'intero JSON** con questo esempio: mantieni `energy_accuracy` e gli altri campi già presenti.

3. Riesegui:

```bash
python generate_all_figures.py
```

**Verifica PCK:** la definizione definitiva è `GT[1]`–`GT[5]` (spalla destra–anca sinistra). Nel notebook ExpA eH36M/DHP19 precedente si usavano `GT[2]`–`GT[6]`. Il generatore produce la curva MPJPE per quegli export, ma **non la curva PCK** finché `expa_pck_1_5_recomputed` non è impostato a `true` dopo aver effettivamente ricalcolato i CSV con GT 1/5. I numeri aggregati PCK 2/6 non si possono convertire in 1/5 con una proporzione.

Per usare **dati definitivi** di energia-accuratezza, puoi anche sostituire `energy_accuracy.all` con il CSV originale, se contiene tutti e tre i dataset e la stessa coorte; in alternativa conserva l'anteprima. La versione cached non è un'analisi riprodotta dai raw.

## Opzioni stilistiche

Modifica `style.json` per un cambiamento globale. Per tutti i metodi, colore e marker rimangono sempre uguali nei vari dataset e figure. Se vuoi agire separatamente sulle metriche nelle tue analisi Python:

```python
from paper_figures import load_style, render_accuracy_rate, export_figure
style = load_style('style.json')
figure = render_accuracy_rate(data, style, datasets=['moveiit'], metrics=['pck_percent'])
export_figure(figure, 'output/my_moveiit_pck', style)
```

Per utilizzare una figura esportata nel paper, importa il `.pdf` in LaTeX con `\includegraphics`; lo `.svg` conserva il testo modificabile. Ogni PNG è un'anteprima della corrispondente figura vettoriale.

`output/FIGURE_MANIFEST.md` viene aggiornato dopo ogni esecuzione e contiene sia i file generati che le figure per cui mancano i dati.
