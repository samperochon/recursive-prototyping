# Method, as implemented

This page states what the code does and, in the last section, every place where it fills a
gap the paper leaves open or departs from it. Section numbers refer to the ICPR 2026 paper.

It is a reference, best read after the [README](../README.md) and the demo notebook. File
paths are relative to `src/recursive_prototyping/`.

Notation: `X_train` is the pooled set of window embeddings, cosine distance is
`d(u, v) = 1 - u·v / (‖u‖‖v‖)`, and change points are stored as `[0, t_1, ..., T]`.

## 1. Prototyping (paper §3.1)

**Centroid accumulation** — `prototyping/refinement.py`. Starting from `R_1 = X_train`, each
round `p`:

1. Cosine k-means (Faiss, spherical) with `C` clusters on the residual `R_p`.
2. A person labels each cluster from its visual probe image as `task-definitive`, `exo-definitive`,
   `task-ambiguous`, `exo-ambiguous` or `noise`.
3. Centroids of the two *definitive* labels are accepted.
4. `R_{p+1}` = every sample of an *ambiguous* cluster, plus the samples of *accepted*
   clusters whose distance to their centroid exceeds `d_min = 0.3`. Samples of *noise*
   clusters are dropped.

The loop stops after `n_rounds`, once more than `max_accepted_prototypes` centroids are
accepted, or if the residual has fewer than `C` samples left.

**Visual probe images** — `prototyping/vpm.py`. For each cluster, the 30 nearest and 30 furthest
members by distance to the centroid, at most one sample per participant, shown as the
central frame of each window with its normalized time in the administration (one recorded test of a participant). A cluster with fewer
than 30 participants has correspondingly fewer frames.

**Consolidation** — `prototyping/consolidation.py`. Complete-linkage HAC on the cosine
distances between accepted centroids; the number of groups maximizes the mean silhouette over
dendrogram cuts. This is applied twice: once to the accepted centroids, then to the
L2-normalized group means of the first pass. A group is kept as a prototype if it is *valid*;
each accepted centroid ends up in its second-pass group if that group is valid, otherwise in
its first-pass group if valid, otherwise it stays a prototype on its own. The result is the
prototype vocabulary and the reduction mapping from accepted centroids to prototypes.

Validity comes from one of two sources:

- a person's verdict on a composite probe of the group's members
  (`prototyping/meta_prototypes.py`): `valid` or `ambiguous`. This is the paper's
  definition.
- a headless stand-in (`compute_group_validity`): the pass produced a genuine silhouette cut
  and the group holds at least two accepted centroids.

## 2. Segmentation (paper §3.2)

`segmentation/`. Per participant, kernel change-point detection with an RBF kernel
(`ruptures.KernelCPD`), bandwidth `gamma = 1 / (2 σ²)` where `σ` is the median pairwise
Euclidean distance of that participant's embeddings.

Penalty calibration: for each participant, the exact unpenalized risk is computed for about
30 change-point counts `m` in `[0.6 M_max, M_max]`, with `M_max` = one change point per 2 s
of video. A single regression of `risk / n` on `m`, with one intercept per participant and a
shared slope `s`, gives each participant the penalty `λ = -2 n s` (`n` = that participant's
number of windows), which is then used for the final PELT segmentation.

## 3. Symbolization (paper §3.3)

`symbolization/`.

1. Each window is assigned to its nearest accepted centroid and takes that centroid's
   prototype through the reduction mapping.
2. For each prototype `g`, the mean `m_g` and standard deviation `σ_g` of the assignment
   distances are computed over the whole cohort; windows further than `m_g + σ_g` become
   background (`-1`).
3. Each segment takes the majority label of its central half.
4. Segments shorter than 4 windows whose two neighbours carry the same symbol are merged
   into them.

## 4. Choices the paper leaves open, and departures from it

"The original implementation" below is the unpublished code in which the method was first
developed.

| Topic | What the code does | Status |
|---|---|---|
| Fifth label `noise` | Offered to the annotator; such clusters are discarded, not recycled | The paper has four labels; the original annotation tool had this fifth one |
| Residual assignment | Uses the current round's k-means assignment of `R_p`, so `R_{p+1} ⊆ R_p` | The paper's supplementary material (§2, step 3) reassigns every sample against `A_p ∪ N_p` (all accepted centroids so far plus this round's ambiguous ones), under which the residual need not shrink |
| Invalid prototypes | An accepted centroid whose final group is not valid falls back to its valid pass-1 group, else stays a prototype on its own; a validity verdict is therefore also collected on pass-1 groups | The paper inspects final prototypes only and keeps the validated ones; the original implementation sends windows of rejected prototypes to background |
| End of the loop when the residual is smaller than `C` | Stops | Not specified |
| Silhouette search range | Starts at 15 groups when there are more than 15 items, and tries at most 60 evenly spaced values | The paper maximizes over all cuts; the floor follows the original implementation and avoids the trivial 2-group optimum |
| Headless validity | Genuine cut and ≥ 2 members | A stand-in; the paper defines validity as a human verdict, and a singleton is trivially valid there |
| Prototype type | Majority type of its centroids, first seen wins ties | Not specified |
| HAC linkage | Complete | Not stated in the paper; the original implementation's default |
| Kernel and bandwidth | RBF, median heuristic per participant | Not stated in the paper; as in the original implementation |
| `M_max` | Video duration / 2 s per participant, with 0.32 s per window index (stride 8 at 25 fps) | Not stated in the paper; the original implementation fixes `M_max = 900` for every participant (2 s per change point at the median 30 min duration) |
| Penalty factor | `-2 n s`, applied once | The original implementation applies the factor 2 twice |
| Central half | `trim = len // 4` on each side; segments shorter than 4 windows vote over all their windows | Rounding not specified |
| Majority-vote ties | Smallest label wins, so background wins a tie | Not specified |
| Short-segment merging | One left-to-right pass; a merge can enable the next one; first and last segments never merge | Not specified |
| Background radius | Population standard deviation, strict `>` | Strict `>` as in the paper ("exceeding"); the original implementation uses `>=`. The standard deviation's normalization is not specified; population is the original implementation's |
