## About PHLOWER

Multi-modal single-cell sequencing, which captures changes in chromatin and gene expression in the same cells, is a game changer in the study of gene regulation in cellular differentiation processes. Computational trajectory analysis is a key computational task for inferring differentiation trees from this single-cell data, though current methods struggle with complex, multi-branching trees and multi-modal data. To address this, PHLOWER leverages simplicial complexes and Hodge Laplacian decomposition to infer trajectory embeddings. These natural representations of cell differentiation facilitate the estimation of their underlying differentiation trees.


### Incidence matrix

```{eval-rst}

Incidence matrix :math:`\mathbf{B_1}` records the relationship bewtween vertics and edges in a Graph :math:`G =(\mathcal{V}, \mathcal{E})`, where :math:`\mathcal{V}` is the set of vertices and :math:`\mathcal{E}` is the set of edges.
For vertex :math:`v_j \in \mathcal{V}` and edge :math:`e_i \in \mathcal{E}`, :math:`\mathbf{B_1}` is defined as:

.. math::

  \begin{equation}
  \label{eqn:incidencematrix}
  B_1[i,j] = \begin{cases}
  -1 &\text{if edge } e_j \text{ leaves  vertex }v_i \\
  1 &\text{if edge } e_j \text{ enters  vertex }v_i \\
  0 &\text{if otherwise}.
  \end{cases}
  \end{equation}




Incidence matrix :math:`\mathbf{B_2}` is a high-order of :math:`\mathbf{B_1}`, which records the relationship bewtween edges and triangles in a Graph :math:`G =(\mathcal{V}, \mathcal{E}, \mathcal{T})`, where :math:`\mathcal{T}` is the set of triangles. the definition of :math:`\mathbf{B_2}` is similar to :math:`\mathbf{B_1}`:

.. math::

  \begin{equation}
  \label{eqn:incidencematrix2}
  B_2[i,q] = \begin{cases}
  -1 &\text{if }e_i \in \bigtriangleup_{q} \text{ and } e_i \text{ has same direction as the triangle}\bigtriangleup_{q}  \\
  1 &\text{if }e_i \in \bigtriangleup_{q} \text{ and } e_i \text{ has opposite direction than the} \bigtriangleup_{q} \\
  0 &\text{if otherwise}.
  \end{cases}
  \end{equation}

```

### Hodge Laplacian

```{eval-rst}
Hodge laplacian is denfined as:

.. math::

  \begin{equation}
  \label{eqn:hodgeLaplacian}
  {L}_1 = \mathbf{B}_{1}^\top \mathbf{B}_{1} + \mathbf{B}_{2} \mathbf{B}_{2}^\top.
  \end{equation}

From the formula we can see that the Hodge Laplacian captures not only the relationship between vertices and edges but also the relationship between edges and triangles. The Hodge Laplacian matrix is a high-order graph Laplacian matrix, which can be used to infer the underlying differentiation trees.

Like the laplacian matrix, the Hodge laplacian also has the normalized version, which is defined as:

.. math::

  \begin{equation}
  \label{eqn:normL1}
  \mathcal{L}_1 = \mathbf{D}_2 \mathbf{B}_1^\top \mathbf{D}_1^{-1} \mathbf{B}_1 + \mathbf{B}_2 \mathbf{D}_3 \mathbf{B}_2^\top \mathbf{D}_2^{-1}
  \end{equation}

where :math:`\mathbf{D}_2` is the diagonal matrix of (adjusted) degrees of each edge, i.e. :math:`\mathbf{D}_2 = \max{(\text{diag}(|\mathbf{B}_2| \mathbf{1}), \mathbf{I})}`. :math:`\mathbf{D}_1` is the diagonal matrix of weighted degrees of the vertices, and :math:`\mathbf{D}_3=\frac{1}{3}\mathbf{I}`.

We construct the symmetric form of :math:`\mathcal{L}_1` as following:

.. math::

  \begin{equation}
  \mathcal{L}_1^s = \mathbf{D}_2^{-1/2} \mathcal{L}_1 \mathbf{D}_2^{1/2} = \mathbf{D}_2^{1/2} \mathbf{B}_1^\top \mathbf{D}_1^{-1} \mathbf{B}_1 \mathbf{D}_2^{1/2} + \mathbf{D}^{-1/2} \mathbf{B}_2 \mathbf{D}_3 \mathbf{B}_2^\top \mathbf{D}_2^{-1/2}.
  \end{equation}


The eigen decomposition of :math:`\mathcal{L}_1` is:

.. math::

  \begin{equation}
  \label{eqn:l1decomposition}
  \mathcal{L}_1 = \mathbf{D}_2^{1/2} \mathcal{L}_1^s \mathbf{D}_2^{-1/2} =  \mathbf{D}_2^{1/2} Q \Lambda Q^\top \mathbf{D}_2^{-1/2} = \mathbf{U} \Lambda \mathbf{U}^{-1}
  \end{equation}

where :math:`Q\in \mathbb{R}^{\|\mathcal{E}\|\times \|\mathcal{E}\|}` is the eigenvector of :math:`\mathcal{L}_1^s`, :math:`\|\mathcal{E}\|` is the number of edges in graph :math:`G`, the diagonal matrix :math:`\Lambda\in \mathbb{R}^{\|\mathcal{E}\|}` records the corresponding eigenvalues. We can make use of the right eigenvector :math:`q_R = \mathbf{D}_2^{1/2} q` as the edges spectrum.


The eigenvector with zero eigenvalue is the kernel of the Hodge Laplacian, which is the harmonic 1-forms. The harmonic 1-forms can be used to infer the underlying differentiation trees.
```



### Trajectory sampling

```{eval-rst}

To generate trajectories, we sample paths (or edge flows) in the graph by following edges with positive divergence (or increasing pseudotime). We choose a random starting point from vertices (cells) with :math:`m` lowest pseudo-time values. We choose the next vertex randomly by considering the divergence values. Only positive divergences (increase in pseudo-time) are considered. We stop when no further positive potential is available. We define a path :math:`\mathbf{f}\in\mathbb{R}^{\|\mathcal{E}\|}` on a simplicial complex as:

.. math::

  \begin{equation}
      \mathbf{f}[i,j] = \begin{cases}
          1\quad & \text{if edge }(i,j) \text{ is traversed}\\
          -1\quad & \text{if edge }(j,i) \text{ is traversed}\\
          0 \quad & \text{otherwise}
      \end{cases}
  \label{eq:edgeflow}
  \end{equation}

Random walk is repeated :math:`n` times. This provides us with a path matrix :math:`\mathbf{F}\in \mathbb{R}^{\|\mathcal{E}\| \times n}`, where :math:`\|\mathcal{E}\|` is number of edges in graph :math:`G`.


We next project these paths :math:`\mathbf{F}` onto harmonic space to estimate a trajectory embedding. Let's recapitulate the decomposition of the normalized Hodge 1-Laplacian:

.. math::

  \begin{equation}
  \mathcal{L}_1 = \mathbf{U}\Lambda \mathbf{U}^{-1},
  \end{equation}

where :math:`\mathbf{U}=(\mathbf{u}_1,\cdots, \mathbf{u}_{\|\mathcal{E}\|})` is the eigenvector matrix and :math:`\Lambda = \mathrm{diag}(\lambda_1,\cdots, \lambda_{\|\mathcal{E}\|})` are the eigenvectors. We assume the eigenvectors have been sorted by their corresponding increasing eigenvalues such that :math:`0\leq\lambda_1\leq\lambda_2\leq\cdots\leq\lambda_{\|\mathcal{E}\|}`.

Denote :math:`\mathbf{H}:=(\mathbf{u}_1,\cdots, \mathbf{u}_{h})` to be the matrix containing all the harmonic functions associated to :math:`\mathcal{L}_1`, i.e. all of the eigenvectors corresponding to the :math:`0` eigenvalues, where :math:`h` is the number with eigenvalues being equal to 0 and :math:`\mathbf{H}\in \mathbb{R}^{\|\mathcal{E}\|\times h}`. We further project edge flow matrix :math:`\mathbf{F}` onto the harmonic space by matrix multiplication  such that

.. math::

  \begin{equation}
  \mathbf{H} = \mathbf{H}^\top \mathbf{F}
  \end{equation}

where :math:`\mathbf{H}\in \mathbb{R}^{h\times n}` embed each trajectory into :math:`h` dimensions. PHLOWER next performs clustering on :math:`\mathbf{H}` with DBSCAN to group the paths into major differentiation trajectories.
```

### Cumulative trajectory embedding and tree inference

```{eval-rst}

The path representation does not keep the time step of a edge visit.  Thus we also define a traversed edge flow (transversed path) matrix :math:`\hat{\mathbf{f}}\in \mathbb{R}^{{\|\mathcal{E}\|}\times S}` to record both the time step associated with every edge visit, i.e.:

.. math::

  \begin{equation}
      \hat{\mathbf{f}}[i,j, s] = \begin{cases}
          1\quad & \text{if edge }(i,j) \text{ is traversed in step } s\\
          -1\quad & \text{if edge }(j,i) \text{ is traversed in step } s\\
          0 \quad & \text{otherwise}
      \end{cases}
  \end{equation}

where :math:`S` is the length of the flow, :math:`1\leq s\leq S` is the :math:`s\mathrm{th}` step and :math:`\|\mathcal{E}\|` is number of edges in graph :math:`G`. As we have :math:`n` trajectories, we will have :math:`n` traversed edge flow matrices :math:`\{\hat{\mathbf{f}}_1, \hat{\mathbf{f}}_2,\cdots, \hat{\mathbf{f}}_n\}`.

We make use of cumulative trajectory embedding to represent paths and for the detection of major trajectories and branching points. For a path :math:`\hat{\mathbf{f}}`, we can estimate a point associated with every step :math:`s` in this cumulative trajectory embedding space as:

.. math::

  \begin{equation}
      \mathbf{v}_s =  \sum_{i = 1}^{s} \mathbf{H}^\top \hat{\mathbf{f}}_{,i}
  \end{equation}

where :math:`\hat{\mathbf{f}}_{,i}\in \mathbb{R}^{\|\mathcal{E}\|}`, :math:`\mathbf{v}_s \in \mathbb{R}^h` is a coordinate in cumulative trajectory embedding space associated to the :math:`s\mathrm{th}` step in trajectory :math:`\hat{\mathbf{f}}` and :math:`h` is the number of harmonic functions. This is repeated for all step sizes, which defines a vector :math:`\mathbf{v} = \{\mathbf{v}_1, \cdots,\mathbf{v}_S\}` for every path. These vectors are low dimensional representations of paths in the cumulative trajectory embedding. By coloring paths from distinct groups with distinct colors, we can recognize branching point events, branches shared by trajectory groups and terminal branches. Note also that if we consider only the final entry for every path, we obtain the same result as in the previously described trajectory embedding.

Since we have performed the DBSCAN clustering method to cluster the :math:`n` paths into :math:`m` groups :math:`\{g_1, g_2, \cdots g_m\}` on the trajectory embedding. PHLOWER next uses a procedure to find the differentiation tree structure. First, it estimates pseudo-time values for every edge, i.e. the average pseudo-time :math:`u^s` from vertices associated to the edge. It next bins all edges by considering their pseudo-time, i.e. it selects the trajectory group with the lowest pseudo-time and splits its edges in :math:`p` bins. The same range of pseuudo-time is used to bin all trajectory groups and bins are indexed in increasing pseudo-time.  After binning of edges for each group, PHLOWER next finds the branching points for all group pairs by comparing the distance of edges within the bin vs. the distance of edges between the bins for a given bin index :math:`i`.

More formally, for groups :math:`g_i` and :math:`g_j` and bin :math:`k`, their corresponding edges in cumulative space are defined as set :math:`\mathbf{V}^k_i`, :math:`\mathbf{V}^k_j`. We then estimate the average edge coordinate per bin to serve as backbones for every group, i.e.:

.. math::

  \begin{equation}
      \overline{b^k_i} = \frac{1}{M}\sum_{v \in \mathbf{V}^k_i } v,
  \end{equation}

where :math:`M=|\mathbf{V}^k_i|` is the number of edges in the bin. We also consider the average distance between edges in a bin to have an estimate of the compactness of edges in a bin and trajectory, i.e.:

.. math::

  \begin{equation}
      \overline{\sigma^k_i} = \frac{1}{M^2} \sum_{u \in \mathbf{V}^k_i}  \sum_{v \in \mathbf{V}^k_i} \Big\|u-v\Big\|_2 \qquad.
  \end{equation}

Finally, we calculate the distance between two groups :math:`g_i` and :math:`g_j` in time bin :math:`k`, such that:

.. math::

  \begin{equation}
      \mathrm{d}(i,j,k) = \Bigg\| \frac{ \overline{b^k_i} }{ \overline{\sigma^k_i}} - \frac{ \overline{b^k_j} }{ \overline{\sigma^k_j}} \Bigg\|_2.
  \end{equation}

For every pair of groups, PHLOWER finds a unique branching point by transversing bins in decreasing order and finding the first bin such that :math:`\mathrm{d}(i,j,k) < \sigma` (as default :math:`1`). This is repeated for all pairs of groups. The tree is finally built in a bottom up manner. PHLOWER first considers branching points with highest index and builds a sub-tree by merging the two trajectory groups at hand. This is repeated until all branching points are considered.

```



### References

A preprint of the paper is available at: [https://doi.org/10.1101/2024.10.01.613179](https://doi.org/10.1101/2024.10.01.613179)


