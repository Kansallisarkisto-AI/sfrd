# Changelog

All _major_ changes to this project will be documented in this file.

The project currently uses date-based versioning. Version numbers may be introduced in the future.

## 2026-10-15

### Added
- Text recognition can use a cyrillic model alongside the main one, with script classification choosing between them per line (optional; see the nafrec options `--cyrillic_recognition_model_path`, `--cyrillic_processor_path`, `--script_classification_model_path`).
- Options not recognized by `structuralizer` are passed on to nafrec, so all nafrec options (e.g. `--device`, `--gpu_ids`, `--detector paddle`, PP-OCR models, fp16) are available on the SfRD command line. Mistakes in them are reported before the alignment step starts.
- SfRD can run on CPU (nafrec's `--device cpu`); a GPU is no longer required.

### Changed
- Text line detection and recognition now use [nafrec](https://github.com/Kansallisarkisto-AI/nafrec) instead of the old nafhtr (RF-DETR + TrOCR) pipeline. nafrec starts and manages the model workers; SfRD keeps its own CPU workers.
- The pipeline options of SfRD were removed in favor of the equivalent nafrec options: `--text_batch_size` is now `--trocr_batch_size`, `--gpu_in_flight_limit` is `--inference_in_flight_limit`, `--segmentation_tilesize` is `--tile_size`, `--segmentation_tileoverlap` is `--tile_overlap`, and `--segmentation_overallsize` is derived from `--tile_size`, `--tiles_across` and `--tile_overlap` (768 without tiling). `--detection_model_path`, `--recognition_model_path`, `--workers_per_gpu` and the threshold options keep their names, but nafrec now requires `--processor_path` as well as `--recognition_model_path` (SfRD used to use the model path for both). `--recognition_model_revision` was removed.
- `--cpu_processes` now only sets the number of SfRD CPU workers (the default is what nafrec would use).
- Abbreviated options (e.g. `--disreg`) are no longer accepted, so that they can't swallow an option meant for nafrec.

### Fixed
- N/A

## 2026-08-20

### Added
- Added heuristic and exact solvers for refining bounding boxes
- Added template suggestion feature, which finds the most central nodes in the connected components of the alignment graph.

### Changed
- N/A

### Fixed
- N/A

## 2026-07-22

### Added
- Added option to disregard RF-DETR results for regions, text lines or both (using only SfRD and annotations)

### Changed
- N/A

### Fixed
- N/A

## 2026-06-24

### Added
- Added Label Studio ML backend for preannotations
- Added CVAT preannotation script. CVAT is less suitable for SfRD annotations because it doesn't have multiselect functionality.

### Changed
- N/A

### Fixed
- N/A

## 2026-06-12

### Added
- Added fine-grained deformation correction based on thin plate splines (TPS)
- Improved config loading and made more tolerant config for deformed materials

### Changed
- Modified changelog format.

### Fixed
- N/A

## 2026-05-22

### Added
- Initial project structure.
- Basic configuration and documentation.

### Changed
- N/A

### Fixed
- N/A