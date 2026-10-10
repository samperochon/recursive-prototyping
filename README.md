# recursive-prototyping

Code for **"Recursive prototyping for computational behavioral analysis from egocentric
videos"** (Perochon & Oudre, ICPR 2026), which develop a framework for constructing interpretable symbolic representations of egocentric video captured during procedural tasks. The method combines:

- **Recursive prototype discovery via iterative latent subspace refinement with minimal human feedback**: a recursive human-in-the-loop loop mines a compact vocabulary of visual prototypes: cosine k-means proposes clusters, a human annotator accepts or rejects each from a visual probe image, poorly explained samples are re-clustered, and accepted centroids are consolidated by hierarchical clustering

- **Vocabulary consolidation through hierarchical clustering to reduce redundancy**

- **Unsupervised temporal segmentation using kernel-based change-point detection**

- **Symbolization**: each window takes the prototype of its nearest centroid (or a background symbol if it is too far), and each segment takes the majority symbol of its central half.

The resulting symbolic representations are used in the paper to explore fine-grained behavioral analysis and cross-subject comparison without requiring dense ground-truth annotations. The input is any precomputed embedding sequence (the paper uses D=1408 VideoMAE-v2 features [1]); the clinical dataset is private, so everything here runs on a bundled synthetic generator.


[1] Limin Wang, undefined., et al, "VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking," 2023.


## Install

```bash
pip install -e .                 # Python >= 3.10, < 3.13
pip install -e ".[annotation]"   # + the interactive annotation widget (needs Jupyter)
pip install -e ".[notebook,dev]" # + the demo notebook and tests
```

## Quickstart

The whole pipeline on synthetic data, with a scripted annotator in place of a person:

```python
import itertools
from recursive_prototyping.config.schema import (
    PrototypingConfig, SegmentationConfig, SymbolizationConfig)
from recursive_prototyping.pipeline import run_full_pipeline
from recursive_prototyping.prototyping.annotation_widget import fake_annotate
from recursive_prototyping.synthetic.frames import SyntheticFrameSource
from recursive_prototyping.synthetic.generator import (
    SyntheticDatasetConfig, generate_synthetic_embeddings)

participants = generate_synthetic_embeddings(SyntheticDatasetConfig(
    n_participants=3, n_action_clusters=3, min_windows=60, max_windows=100,
    embedding_dim=32, seed=47))
embeddings = {pid: p.embeddings for pid, p in participants.items()}

# A scripted annotator standing in for the widget.
labels = itertools.cycle(["task-definitive", "exo-definitive", "task-ambiguous"])
annotate = lambda paths, categories: fake_annotate(paths, lambda _p: next(labels))

result = run_full_pipeline(
    embeddings, SyntheticFrameSource(participants), annotate,
    PrototypingConfig(n_clusters_per_round=12, n_rounds=2, faiss_niter=50, faiss_nredo=1,
                      n_probe_neighbors=6, n_probe_participants_target=3, probe_image_ncols=5),
    SegmentationConfig(), SymbolizationConfig(), "outputs/demo_run", annotator_id="demo")

print(result.vocabulary.shape, result.participants["synth_000"].symbols.tolist())
```

## Guided tour

1. [notebooks/demo_end_to_end.ipynb](notebooks/demo_end_to_end.ipynb) runs the same pipeline
   step by step, shows a probe image, and plots the segmentation and the symbolic timeline.
2. [docs/METHOD.md](docs/METHOD.md) states the method as implemented, step by step, and
   lists every choice the paper leaves open.

## Repository map

```
src/recursive_prototyping/
  pipeline.py     run_full_pipeline: the three steps in one call
  prototyping/    step 1
    clustering.py         cosine k-means and nearest-centroid assignment
    vpm.py                visual probe images
    annotation_widget.py  the interactive widget, and a scripted annotator for tests
    refinement.py         the recursive loop, one round at a time or all at once
    consolidation.py      hierarchical clustering of the accepted centroids
    meta_prototypes.py    composite images and verdicts for the consolidated groups
  segmentation/   step 2: kernel change-point detection and penalty calibration
  symbolization/  step 3: assignment, background filter, segment symbols
  synthetic/      synthetic embeddings and frames
  viz/            figures
  config/         dataclass configs and their JSON snapshots
  io/             npy/npz and parquet helpers
notebooks/        the end-to-end demo
docs/METHOD.md    the method as implemented
tests/            run with: pytest tests -q
```

Feature extraction and the paper's evaluation are not part of this repository.

## Running on your own data

`run_full_pipeline` needs three things:

- **Embeddings**: a dict `participant_id -> array [T, D]`, one row per window in temporal
  order.
- **A frame source**: any object with a method
  `central_frame(participant_id, window_index) -> HxWx3 uint8 array` returning the video
  frame at the centre of that window. It is what the probe images are built from.
- **`hop_seconds`**: the time between two consecutive windows (default 0.32 s), used to
  bound the number of change points.

The three config dataclasses in `config/schema.py` hold every parameter, with the paper's
values as defaults; the quickstart only shrinks them to fit the tiny synthetic dataset.

## Annotating with a person in the loop

`run_full_pipeline` takes the annotator as a function, which suits a scripted one. A person
clicks through a widget in a notebook, and the widget returns before the clicks are made, so
the first step has to be driven round by round, one notebook cell per action. Install the
`annotation` extra first.

**Setup.** Pool the embeddings into the three aligned arrays the loop works on:

```python
import numpy as np
from recursive_prototyping.config.schema import PrototypingConfig
from recursive_prototyping.prototyping.annotation_widget import (
    annotate_probe_images, collect_annotations)
from recursive_prototyping.prototyping.refinement import (
    finish_round, init_refinement_state, start_round)

ids = list(embeddings)
X_train = np.concatenate([embeddings[p] for p in ids])
participant_ids = np.concatenate([np.full(len(embeddings[p]), p) for p in ids])
local_indices = np.concatenate([np.arange(len(embeddings[p])) for p in ids])

config = PrototypingConfig()
state = init_refinement_state(X_train)
```

**One round** is three cells. Repeat them until `start_round` returns an empty list.

```python
# cell 1: cluster the residual and render one probe image per cluster
probe_paths = start_round(
    state, X_train, participant_ids, local_indices, config, frame_source, "outputs/run")
```

```python
# cell 2: open the widget and label every image
clicks = annotate_probe_images(probe_paths, config.annotation_categories)
```

```python
# cell 3: once every image has a label, close the round
finish_round(state, collect_annotations(clicks, probe_paths), config, "your-id", "outputs/run")
```

Images are shown from the largest cluster to the smallest. The five labels and what each does
to the cluster are described in [docs/METHOD.md](docs/METHOD.md#1-prototyping-paper-31).
`save_refinement_state` and `load_refinement_state` write and reload `state` between
rounds, so the annotation can be interrupted.

**Consolidation.** The accepted centroids are grouped twice by hierarchical clustering.
After each pass you judge every group from one image stacking its members' probe images:
`valid` if they all show the same action, `ambiguous` otherwise.

```python
from recursive_prototyping.prototyping.consolidation import (
    MembershipTree, assemble_final_vocabulary, consolidate_once)
from recursive_prototyping.prototyping.meta_prototypes import (
    META_CATEGORIES, collect_validity, render_meta_probe_images)

pass1 = consolidate_once(state.accepted_centroids)
paths1, _ = render_meta_probe_images(
    pass1.assignment, state.accepted_probe_paths, "outputs/run/meta", "pass1")
clicks1 = annotate_probe_images(paths1, META_CATEGORIES)            # label, then next cell
```

```python
validity1 = collect_validity(collect_annotations(clicks1, paths1), paths1)

pass2 = consolidate_once(pass1.group_centroids)
paths2, _ = render_meta_probe_images(pass2.assignment, paths1, "outputs/run/meta", "pass2")
clicks2 = annotate_probe_images(paths2, META_CATEGORIES)            # label, then next cell
```

```python
validity2 = collect_validity(collect_annotations(clicks2, paths2), paths2)
vocabulary = assemble_final_vocabulary(
    MembershipTree(pass1.assignment, pass2.assignment), validity1, validity2, pass1, pass2,
    state.accepted_centroids, state.accepted_cluster_types)
```

**Segmentation and symbolization** then run without anyone in the loop:

```python
from recursive_prototyping.config.schema import SegmentationConfig, SymbolizationConfig
from recursive_prototyping.pipeline import pooled_prototype_thresholds, symbolize_participant
from recursive_prototyping.segmentation.penalty import calibrate_and_segment

calibration = calibrate_and_segment(embeddings, SegmentationConfig())
symb_config = SymbolizationConfig()
thresholds = pooled_prototype_thresholds(
    embeddings, vocabulary.raw_centroids, symb_config, vocabulary.raw_to_prototype)

symbols = {}
for pid, X in embeddings.items():
    window_symbols, cpts, symbols[pid] = symbolize_participant(
        X, vocabulary.raw_centroids, calibration.participants[pid].cpts, symb_config,
        thresholds, vocabulary.raw_to_prototype)
```

`vocabulary.centroids` is the final set of prototypes and `symbols[pid]` the symbolic string
of each participant, as in the quickstart.

## Citation

```bibtex
@inproceedings{perochon2026recursive,
  title     = {Recursive prototyping for computational behavioral analysis from egocentric videos},
  author    = {Perochon, Sam and Oudre, Laurent},
  booktitle = {International Conference on Pattern Recognition (ICPR)},
  year      = {2026},
}
```

MIT license.
