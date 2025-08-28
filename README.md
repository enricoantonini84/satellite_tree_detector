# Satellite tree detector

Collection of Python scripts to analyze satellite map tiles and generate GeoJSON files with detected tree locations.
This is part of a project made for my BSc in computer science at UniVr.

## Description

This repository provides tools for manipulating and processing satellite map tiles downloaded from external providers, aimed at facilitating detection tasks (e.g., trees/vegetation using YOLO or similar models) and generating georeferenced outputs.

## Requirements

All dependencies are listed in `requirements.txt`.  
**Using a virtual environment** (`venv`, `conda`, or `miniconda`) is strongly recommended.

## Project Structure

### `tiles/` folder

Scripts for satellite tile manipulation and processing:

- **[`merge_tiles.py`](tiles/merge_tiles.py)**
  Merge tiles from a folder (named in slippy format: `tile_[latitude]_[longitude]_z[zoom].jpg`) into a single GeoTIFF file. Supports RGB images (JPG/PNG) and automatically calculates geographic bounds from tile coordinates. Features include:
  - Parse tile filenames with lat/lon/zoom information
  - Convert coordinates from lat/lon to slippy map format
  - Calculate geographic bounds for proper georeferencing
  - Generate compressed GeoTIFF with WGS84 coordinate system
  - Handle mixed tile collections and validate zoom levels

- **[`split_geotiff_tiles.py`](tiles/split_geotiff_tiles.py)**
  Split large GeoTIFF satellite images into smaller sub-tiles of specific resolution. Preserves geospatial information for each tile. Features include:
  - Configurable tile size (default 640x640 pixels)
  - Automatic calculation of required tile grid
  - Preserve original GeoTIFF metadata and coordinate system
  - Handle edge tiles that don't fit standard dimensions
  - Generate compressed output with LZW compression

- **[`align_geotiff.py`](tiles/align_geotiff.py)**
  Realign (translate) GeoTIFF tiles from current coordinates to target coordinates. Useful for correcting misaligned satellite imagery. Features include:
  - Calculate lat/lon offsets between current and target positions
  - Apply affine transformation to preserve pixel data
  - Maintain original image quality and metadata
  - Support for any coordinate reference system

> **Example workflow**: Download satellite tiles at 512x512px → merge with [`merge_tiles.py`](tiles/merge_tiles.py) → split with [`split_geotiff_tiles.py`](tiles/split_geotiff_tiles.py) at 640x640px for YOLO training, preserving image definition.

### `yolo/` folder

Ultralytics YOLO-based scripts for tree detection training and inference:

- **[`train.py`](yolo/train.py)**
  Train YOLO11x model for tree detection with optimized configurations. Supports multiple devices (CPU/CUDA/MPS) and includes:
  - Automatic device detection (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
  - Pre-configured training parameters (300 epochs, batch size 16, cosine learning rate)
  - Single class detection setup for tree/vegetation detection
  - Validation metrics reporting (mAP50, mAP50-95)
  - Model checkpointing every 10 epochs with patience of 100

- **[`inference.py`](yolo/inference.py)**
  Perform tree detection inference on images/GeoTIFF files using trained YOLO models. Features include:
  - Support for regular images and georeferenced GeoTIFF files
  - Pixel-to-coordinate transformation for GeoTIFF inputs
  - Configurable confidence threshold (default 0.05)
  - JSON export with detection results and geographic coordinates
  - Optional visualization output with bounding boxes
  - Detailed detection metadata (confidence, bbox, center coordinates)

### `detectree2/` folder

DetecTree2-based instance segmentation for precise tree detection:

- **[`inference.py`](detectree2/inference.py)**
  Advanced tree detection using Detectron2-based DetecTree2 for instance segmentation. Features include:
  - Polygon-based tree detection (not just bounding boxes)
  - Confidence threshold filtering
  - Contour extraction and polygon simplification
  - Visualization with filled polygons and transparency
  - CPU/GPU inference support with fallback handling
  - Comprehensive error handling for different environments

### Configuration Files

- **[`requirements.txt`](requirements.txt)**
  Complete dependency list for the project including:
  - Image processing: `pillow`, `opencv-python`, `matplotlib`
  - Geospatial: `rasterio` for GeoTIFF handling
  - Machine learning: `torch`, `ultralytics`, `scikit-learn`
  - Tree detection models: `detectron2`, `detectree2`
  - Utilities: `numpy`, `requests`, `pyyaml`

- **[`LICENSE`](LICENSE)**
  This project is licensed under the GNU Affero General Public License version 3.0 (AGPL-3.0).
  
You can find the full license text in the LICENSE file.

If you use, modify, or distribute this software, or make it accessible over a network (for example, as a web service), you must provide the complete source code of the project, including any modifications, under the same AGPL-3.0 license.

For more information, please visit: https://www.gnu.org/licenses/agpl-3.0.html

- **[`inference_pipeline.py`](inference_pipeline.py)**
  Complete end-to-end pipeline for tree detection on satellite imagery. Features include:
  - Support for both YOLO and DetecTree2 models
  - Automatic processing of single images, tile folders, or large GeoTIFF files
  - Tile splitting for large images with configurable tile size
  - Batch processing with progress tracking
  - Automatic GeoJSON generation with proper coordinate transformation
  - Merged annotated GeoTIFF output
  - Comprehensive error handling and logging
  - Flexible output options (images, JSON, GeoJSON, merged GeoTIFF)

  You can find the full license text in the LICENSE file.

  If you use, modify, or distribute this software, or make it accessible over a network (for example, as a web service), you must provide the complete source code of the project, including any modifications, under the same AGPL-3.0 license.

  For more information, please visit: https://www.gnu.org/licenses/agpl-3.0.html

- **[`.gitignore`](.gitignore)**
  Git ignore patterns for the project

## Usage

1. Install requirements:

pip install -r requirements.txt

2. Run the desired scripts from the `tiles/` folder.

## Complete Pipeline Usage

For end-to-end processing, use the [`inference_pipeline.py`](inference_pipeline.py) script which orchestrates the entire workflow:

```bash
# Process a large GeoTIFF with DetecTree2
python inference_pipeline.py --input large_image.tif --model detectree2 --model-path path/to/model.pth --output results

# Process a folder of tiles with YOLO
python inference_pipeline.py --input tile_folder/ --model yolo --model-path path/to/model.pt --output results

# Process with custom parameters
python inference_pipeline.py --input image.tif --model detectree2 --model-path model.pth --confidence 0.3 --tile-size 512 --output results
```

The pipeline automatically handles:
- Tile splitting for large images
- Batch detection on all tiles
- GeoJSON generation with proper coordinate transformation
- Merged annotated GeoTIFF creation
- Summary statistics and error reporting

## GeoJSON Generation

The pipeline generates GeoJSON files containing detected tree locations from the processed satellite imagery. These files can be directly imported into GIS software like QGIS for visualization and analysis.

### `json_to_geojson.py`

Script to convert detection JSON files to GeoJSON format with proper geographic coordinates:

- **Coordinate Transformation**: Converts pixel coordinates to geographic coordinates using affine transformation parameters
- **CRS Specification**: Includes explicit EPSG:4326 coordinate reference system for GIS compatibility
- **Multi-format Support**: Handles both YOLO bounding box and DetecTree2 polygon detection formats
- **Batch Processing**: Can process individual JSON files or entire folders of detection results
- **QGIS Compatible**: Generates properly formatted GeoJSON files that load correctly in QGIS without coordinate issues

**Usage Examples:**

```bash
# Process all JSON files in a folder
python json_to_geojson.py --input detectree2_out/json --output output.geojson

# Process specific JSON files
python json_to_geojson.py --files file1.json file2.json --output output.geojson
```

## Disclaimer

These datasets are provided "as is" without any warranties of any kind, express or implied. The authors make no representations or warranties regarding the accuracy, completeness, or reliability of the tree detection data. Users should verify the data for their specific use cases and are solely responsible for any consequences arising from the use of these datasets.

The detection algorithms can generate content that may contain errors, and the datasets should not be considered reliable without proper validation.

## Citations

If you use this software, please consider citing the following libraries and tools that make this project possible:

```bibtex
@software{ball2022detectree2,
  author = {James G. C. Ball},
  title = {detectree2: Python package for automatic tree crown delineation based on Detectron2},
  version = {2.0.1},
  year = {2024},
  url = {https://github.com/PatBall1/detectree2}
}

@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics}
}

```

---

# Satellite tree detector (Italiano)

Collezione di script Python per analizzare tiles di mappe satellitari e generare file GeoJSON con le posizioni degli alberi rilevati.
Questo fa parte di un progetto realizzato per la mia laurea triennale in informatica presso UniVr.

## Descrizione

Questo repository fornisce strumenti per manipolare e processare tiles di mappe satellitari scaricate da fornitori esterni, finalizzati a facilitare compiti di rilevamento (ad esempio, alberi/vegetazione usando YOLO o modelli simili) e generare output georeferenziati.

## Requisiti

Tutte le dipendenze sono elencate in `requirements.txt`.
**L'uso di un ambiente virtuale** (`venv`, `conda`, o `miniconda`) è fortemente raccomandato.

## Struttura del Progetto

### Cartella `tiles/`

Script per manipolazione e processamento di tiles satellitari:

- **[`merge_tiles.py`](tiles/merge_tiles.py)**
  Unisce tiles da una cartella (nominate in formato slippy: `tile_[latitude]_[longitude]_z[zoom].jpg`) in un singolo file GeoTIFF. Supporta immagini RGB (JPG/PNG) e calcola automaticamente i confini geografici dalle coordinate delle tiles. Le funzionalità includono:
  - Analisi dei nomi file delle tiles con informazioni lat/lon/zoom
  - Conversione delle coordinate da lat/lon al formato mappa slippy
  - Calcolo dei confini geografici per un georeferenziamento corretto
  - Generazione di GeoTIFF compresso con sistema di coordinate WGS84
  - Gestione di collezioni miste di tiles e validazione dei livelli di zoom

- **[`split_geotiff_tiles.py`](tiles/split_geotiff_tiles.py)**
  Divide grandi immagini satellitari GeoTIFF in sotto-tiles più piccole di risoluzione specifica. Preserva le informazioni geospaziali per ogni tessera. Le funzionalità includono:
  - Dimensione tessera configurabile (default 640x640 pixel)
  - Calcolo automatico della griglia di tiles necessaria
  - Preservazione dei metadati GeoTIFF originali e del sistema di coordinate
  - Gestione delle tiles di bordo che non si adattano alle dimensioni standard
  - Generazione di output compresso con compressione LZW

- **[`align_geotiff.py`](tiles/align_geotiff.py)**
  Reallinea (trasla) tiles GeoTIFF dalle coordinate attuali alle coordinate target. Utile per correggere immagini satellitari disallineate. Le funzionalità includono:
  - Calcolo degli offset lat/lon tra posizioni attuali e target
  - Applicazione di trasformazione affine per preservare i dati pixel
  - Mantenimento della qualità dell'immagine originale e dei metadati
  - Supporto per qualsiasi sistema di riferimento delle coordinate

> **Flusso di lavoro di esempio**: Scarica tiles satellitari a 512x512px → unisci con [`merge_tiles.py`](tiles/merge_tiles.py) → dividi con [`split_geotiff_tiles.py`](tiles/split_geotiff_tiles.py) a 640x640px per addestramento YOLO, preservando la definizione dell'immagine.

### Cartella `yolo/`

Script basati su Ultralytics YOLO per addestramento e inferenza del rilevamento alberi:

- **[`train.py`](yolo/train.py)**
  Addestra il modello YOLO11x per rilevamento alberi con configurazioni ottimizzate. Supporta dispositivi multipli (CPU/CUDA/MPS) e include:
  - Rilevamento automatico del dispositivo (MPS per Apple Silicon, CUDA per NVIDIA, fallback CPU)
  - Parametri di addestramento pre-configurati (300 epoche, batch size 16, tasso di apprendimento cosinusoidale)
  - Setup per rilevamento classe singola per rilevamento alberi/vegetazione
  - Reportistica delle metriche di validazione (mAP50, mAP50-95)
  - Checkpoint del modello ogni 10 epoche con pazienza di 100

- **[`inference.py`](yolo/inference.py)**
  Esegue inferenza di rilevamento alberi su immagini/file GeoTIFF usando modelli YOLO addestrati. Le funzionalità includono:
  - Supporto per immagini normali e file GeoTIFF georeferenziati
  - Trasformazione pixel-coordinate per input GeoTIFF
  - Soglia di confidenza configurabile (default 0.05)
  - Esportazione JSON con risultati di rilevamento e coordinate geografiche
  - Output di visualizzazione opzionale con riquadri di delimitazione
  - Metadati dettagliati di rilevamento (confidenza, bbox, coordinate centrali)

### Cartella `detectree2/`

Segmentazione di istanza basata su DetecTree2 per rilevamento preciso degli alberi:

- **[`inference.py`](detectree2/inference.py)**
  Rilevamento avanzato alberi usando DetecTree2 basato su Detectron2 per segmentazione di istanza. Le funzionalità includono:
  - Rilevamento alberi basato su poligoni (non solo riquadri di delimitazione)
  - Filtraggio soglia di confidenza
  - Estrazione contorni e semplificazione poligoni
  - Visualizzazione con poligoni riempiti e trasparenza
  - Supporto inferenza CPU/GPU con gestione fallback
  - Gestione errori comprensiva per diversi ambienti

### File di Configurazione

- **[`requirements.txt`](requirements.txt)**
  Lista completa dipendenze per il progetto inclusi:
  - Processamento immagini: `pillow`, `opencv-python`, `matplotlib`
  - Geospaziale: `rasterio` per gestione GeoTIFF
  - Machine learning: `torch`, `ultralytics`, `scikit-learn`
  - Modelli rilevamento alberi: `detectron2`, `detectree2`
  - Utilità: `numpy`, `requests`, `pyyaml`

- **[`LICENSE`](LICENSE)**
  Questo progetto è licenziato sotto la GNU Affero General Public License versione 3.0 (AGPL-3.0).
  
Puoi trovare il testo completo della licenza nel file LICENSE.

Se usi, modifichi, o distribuisci questo software, o lo rendi accessibile attraverso una rete (ad esempio, come servizio web), devi fornire il codice sorgente completo del progetto, incluse eventuali modifiche, sotto la stessa licenza AGPL-3.0.

Per maggiori informazioni, visita: https://www.gnu.org/licenses/agpl-3.0.html

- **[`inference_pipeline.py`](inference_pipeline.py)**
  Pipeline completa end-to-end per rilevamento alberi su immagini satellitari. Le funzionalità includono:
  - Supporto per modelli sia YOLO che DetecTree2
  - Processamento automatico di immagini singole, cartelle di tiles, o grandi file GeoTIFF
  - Divisione in tiles per immagini grandi con dimensione tessera configurabile
  - Processamento batch con tracciamento progresso
  - Generazione automatica GeoJSON con trasformazione coordinate appropriata
  - Output GeoTIFF annotato unito
  - Gestione errori comprensiva e logging
  - Opzioni di output flessibili (immagini, JSON, GeoJSON, GeoTIFF unito)

  Puoi trovare il testo completo della licenza nel file LICENSE.

  Se usi, modifichi, o distribuisci questo software, o lo rendi accessibile attraverso una rete (ad esempio, come servizio web), devi fornire il codice sorgente completo del progetto, incluse eventuali modifiche, sotto la stessa licenza AGPL-3.0.

  Per maggiori informazioni, visita: https://www.gnu.org/licenses/agpl-3.0.html

- **[`.gitignore`](.gitignore)**
  Pattern ignore Git per il progetto

## Utilizzo

1. Installa i requisiti:

```bash
pip install -r requirements.txt
```

2. Esegui gli script desiderati dalla cartella `tiles/`.

## Utilizzo Pipeline Completa

Per processamento end-to-end, usa lo script [`inference_pipeline.py`](inference_pipeline.py) che orchestra l'intero flusso di lavoro:

```bash
# Processa un GeoTIFF grande con DetecTree2
python inference_pipeline.py --input large_image.tif --model detectree2 --model-path path/to/model.pth --output results

# Processa una cartella di tiles con YOLO
python inference_pipeline.py --input tile_folder/ --model yolo --model-path path/to/model.pt --output results

# Processa con parametri personalizzati
python inference_pipeline.py --input image.tif --model detectree2 --model-path model.pth --confidence 0.3 --tile-size 512 --output results
```

La pipeline gestisce automaticamente:
- Divisione in tiles per immagini grandi
- Rilevamento batch su tutte le tiles
- Generazione GeoJSON con trasformazione coordinate appropriata
- Creazione GeoTIFF annotato unito
- Statistiche riassuntive e reportistica errori

## Generazione GeoJSON

La pipeline genera file GeoJSON contenenti posizioni alberi rilevati dalle immagini satellitari processate. Questi file possono essere importati direttamente in software GIS come QGIS per visualizzazione e analisi.

### `json_to_geojson.py`

Script per convertire file JSON di rilevamento in formato GeoJSON con coordinate geografiche appropriate:

- **Trasformazione Coordinate**: Converte coordinate pixel in coordinate geografiche usando parametri di trasformazione affine
- **Specificazione CRS**: Include sistema di riferimento coordinate EPSG:4326 esplicito per compatibilità GIS
- **Supporto Multi-formato**: Gestisce formati rilevamento sia riquadri delimitazione YOLO che poligoni DetecTree2
- **Processamento Batch**: Può processare file JSON individuali o intere cartelle di risultati rilevamento
- **Compatibile QGIS**: Genera file GeoJSON formattati correttamente che si caricano correttamente in QGIS senza problemi di coordinate

**Esempi di Utilizzo:**

```bash
# Processa tutti i file JSON in una cartella
python json_to_geojson.py --input detectree2_out/json --output output.geojson

# Processa file JSON specifici
python json_to_geojson.py --files file1.json file2.json --output output.geojson
```

## Disclaimer

Questi dataset sono forniti "così come sono" senza garanzie di alcun tipo, espresse o implicite. Gli autori non forniscono rappresentazioni o garanzie riguardo all'accuratezza, completezza o affidabilità dei dati di rilevamento alberi. Gli utenti dovrebbero verificare i dati per i loro casi d'uso specifici e sono i soli responsabili per qualsiasi conseguenza derivante dall'uso di questi dataset.

Gli algoritmi di rilevamento possono generare contenuti che potrebbero contenere errori, e i dataset non dovrebbero essere considerati affidabili senza una validazione appropriata.

## Citazioni

Se usi questo software, considera di citare le seguenti librerie e strumenti che rendono possibile questo progetto:

```bibtex
@software{ball2022detectree2,
  author = {James G. C. Ball},
  title = {detectree2: Python package for automatic tree crown delineation based on Detectron2},
  version = {2.0.1},
  year = {2024},
  url = {https://github.com/PatBall1/detectree2}
}

@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics}
}

```
