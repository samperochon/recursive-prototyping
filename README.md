# recursive-prototyping
Code supporting our ICPR 2026 paper submission: "Recursive Prototyping for Computational Behavior Analysis from Egocentric Videos". 

This repository provides the implementation of a framework for constructing interpretable symbolic representations of egocentric video captured during procedural tasks. 
The code will be released upon paper acceptance.

## Overview

This repository provides the implementation of a framework for constructing interpretable symbolic representations of egocentric video captured during procedural tasks. The method combines:

- **Recursive prototype discovery** via iterative latent subspace refinement with minimal human feedback
- **Vocabulary consolidation** through hierarchical clustering to reduce redundancy
- **Unsupervised temporal segmentation** using kernel-based change-point detection

The resulting symbolic representations enable fine-grained behavioral analysis and cross-subject comparison without requiring dense ground-truth annotations.

Code will be released upon paper acceptance.

## Key Features

- Human-in-the-loop prototype refinement for semantic interpretability
- Generalizable across participants in single-environment prescribed tasks
- Robust to annotator subjectivity
- Compatible with pretrained video encoders (VideoMAE-v2)

## Dataset

The framework is evaluated on a clinical dataset of 162 participants performing a standardized cooking task, captured via head-mounted eye-tracker cameras.
