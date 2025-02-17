```{eval-rst}
.. module:: phlower
```

```{eval-rst}
.. automodule:: phlower
   :noindex:
```

# API

Import phlower as:

```
import phlower
```

## tools: `tl`

```{eval-rst}
.. module:: phlower.tl
```

```{eval-rst}
.. currentmodule:: phlower
```


### difusion map

```{eval-rst}
.. autosummary::
   :toctree: generated/

   tl.diffusionGraph
   tl.diffusionGraphDM
```


### hodge laplacian decomposition to infer trajectory

```{eval-rst}
.. autosummary::
   :toctree: generated/

   tl.L1Norm_decomp
   tl.construct_delaunay
   tl.construct_delaunay_persistence
   tl.construct_trucated_delaunay
   tl.construct_circle_delaunay
   tl.count_features_at_radius
   tl.random_climb_knn
   tl.trajs_matrix
   tl.trajs_dm
   tl.trajs_clustering
   tl.harmonic_stream_tree
```


### clustering

```{eval-rst}
.. autosummary::
   :toctree: generated/

   tl.leiden
   tl.louvain
   tl.dbscan
   tl.gaussianmixture
   tl.spectralclustering
   tl.agglomerativeclustering
```




### dimensionality reduction

```{eval-rst}
.. autosummary::
   :toctree: generated/

   tl.run_pca
   tl.run_umap
   tl.run_mds
   tl.run_kernelpca
   tl.run_lda
   tl.run_isomap
   tl.run_fdl
   tl.run_palantir_fdl
   tl.run_tsne
```

### tree analysis

```{eval-rst}
.. autosummary::
   :toctree: generated/

   tl.tree_2branch_markers
   tl.tree_mbranch_markers
   tl.find_branch_start
   tl.find_branch_end
   tl.find_a_branch_all_predecessors
   tl.helping_merged_tree
   tl.helping_submerged_tree
   tl.find_samelevel_daugthers
   tl.TF_gene_correlation
   tl.branch_TF_gene_correlation
   tl.tree_branches_smooth_window
   tl.branch_heatmap_matrix
   tl.print_stream_labels
   tl.change_stream_labels
   tl.fate_tree_full_dataframe
   tl.assign_graph_node_attr_to_adata
   tl.get_tree_leaves_attr
   tl.get_all_attr_names
   tl.tree_label_dict
   tl.tree_original_dict
   tl.end_branch_dict
   tl.branch_regulator_detect
   tl.mbranch_regulator_detect
```

### adata
```{eval-rst}
.. autosummary::
   :toctree: generated/

   tl.magic_adata
```


## external: `ext`

```{eval-rst}
.. autosummary::
   :toctree: generated/

   ext.ddhodge
   ext.plot_stream_sc
   ext.plot_stream
```



## plotting: `pl`

```{eval-rst}
.. module:: phlower.pl
```

```{eval-rst}
.. currentmodule:: phlower
```

### plots

```{eval-rst}
.. autosummary::
   :toctree: generated/

    pl.nxdraw_group
    pl.nxdraw_score
    pl.nxdraw_holes
    pl.nxdraw_harmonic
    pl.plot_traj
    pl.plot_trajs_embedding
    pl.plot_triangle_density
    pl.plot_density_grid
    pl.plot_fate_tree
    pl.plot_fate_tree_embedding
    pl.plot_stream_tree_embedding
    pl.plot_trajectory_harmonic_lines
    pl.plot_trajectory_harmonic_lines_3d
    pl.plot_trajectory_harmonic_points
    pl.plot_trajectory_harmonic_points_3d
    pl.plot_eigen_line
    pl.plot_embedding
    pl.plot_pie_fate_tree
    pl.harmonic_backbone
    pl.harmonic_backbone_3d
    pl.regulator_dot_correlation
    pl.regulator_heatmap
    pl.persisitence_barcodes
    pl.persisitence_birth_death
```


### velocity like plots

```{eval-rst}
.. autosummary::
   :toctree: generated/

    pl.fate_velocity_plot_cumsum
    pl.fate_velocity_plot
```
