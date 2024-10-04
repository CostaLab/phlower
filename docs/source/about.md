## About PHLOWER

Multi-modal single-cell sequencing, which captures changes in chromatin and gene expression in the same cells, is a game changer in the study of gene regulation in cellular differentiation processes. Computational trajectory analysis is a key computational task for inferring differentiation trees from this single-cell data, though current methods struggle with complex, multi-branching trees and multi-modal data. To address this, PHLOWER leverages simplicial complexes and Hodge Laplacian decomposition to infer trajectory embeddings. These natural representations of cell differentiation facilitate the estimation of their underlying differentiation trees.


### Incidence matrix

```{eval-rst}

Incidence matrix :math:`\mathbf{B_1}` records the relationship bewtween vertics and edges in a Graph :math:`G =(\mathcal{V}, \mathcal{E})`, where :math:`\mathcal{V}` is the set of vertices and :math:`\mathcal{E}` is the set of edges.
For vertex :math:`v_j \in \mathcal{V}` and edge :math:`e_i \in \mathcal{E}`, :math:`\mathbf{B_1}` is defined as:

.. math::
   :nowrap:
  \begin{equation}
  \label{eqn:incidencematrix}
  B_1[i,j] = \begin{cases}
  -1 &\text{if edge } e_j \text{ leaves  vertex }v_i \\
  1 &\text{if edge } e_j \text{ enters  vertex }v_i \\
  0 &\text{if otherwise}.
  \end{cases}
  \end{equation}

```


Incidence matrix $\mathbf{B_2}$ is a high-order of $\mathbf{B_1}$, which records the relationship bewtween edges and triangles in a Graph $G =(\mathcal{V}, \mathcal{E}, \mathcal{T})$, where $\mathcal{T}$ is the set of triangles. the definition of $\mathbf{B_2}$ is similar to $\mathbf{B_1}$:
  \begin{equation}
  \label{eqn:incidencematrix2}
  B_2[i,q] = \begin{cases}
  -1 &\text{if }e_i \in \bigtriangleup_{q} \text{ and } e_i \text{ has same direction as the triangle}\bigtriangleup_{q}  \\
  1 &\text{if }e_i \in \bigtriangleup_{q} \text{ and } e_i \text{ has opposite direction than the} \bigtriangleup_{q} \\
  0 &\text{if otherwise}.
  \end{cases}
  \end{equation}

### Hodge Laplacian
Hodge laplacian is denfined as:

\begin{equation}
\label{eqn:hodgeLaplacian}
{L}_1 = \mathbf{B}_{1}^\top \mathbf{B}_{1} + \mathbf{B}_{2} \mathbf{B}_{2}^\top.
\end{equation}

From the formula we can not only capture the relationship between vertices and edges, but also the relationship between edges and triangles. The Hodge Laplacian matrix is a high-order Laplacian matrix, which can be used to infer the underlying differentiation trees.

Like the laoplcian matrix, the Hodge laplacian also has the normalized version, which is defined as:
\begin{equation}
\label{eqn:normL1}
\mathcal{L}_1 = \mathbf{D}_2 \mathbf{B}_1^\top \mathbf{D}_1^{-1} \mathbf{B}_1 + \mathbf{B}_2 \mathbf{D}_3 \mathbf{B}_2^\top \mathbf{D}_2^{-1}
\end{equation}

where $\mathbf{D}_2$ is the diagonal matrix of (adjusted) degrees of each edge, i.e. $\mathbf{D}_2 = \max{(\text{diag}(|\mathbf{B}_2| \mathbf{1}), \mathbf{I})}$. $\mathbf{D}_1$ is the diagonal matrix of weighted degrees of the vertices, and $\mathbf{D}_3=\frac{1}{3}\mathbf{I}$.


We construct the symmetric form of $\mathcal{L}_1$ as following:
\begin{equation}
\label{eqn:normL1sym}
\mathcal{L}_1^s = \mathbf{D}_2^{-1/2} \mathcal{L}_1 \mathbf{D}_2^{1/2} = \mathbf{D}_2^{1/2} \mathbf{B}_1^\top \mathbf{D}_1^{-1} \mathbf{B}_1 \mathbf{D}_2^{1/2} + \mathbf{D}^{-1/2} \mathbf{B}_2 \mathbf{D}_3 \mathbf{B}_2^\top \mathbf{D}_2^{-1/2}.
\end{equation}

The eigen decomposition of $\mathcal{L}_1$ is:

\begin{equation}
\label{eqn:l1decomposition}
\mathcal{L}_1 = \mathbf{D}_2^{1/2} \mathcal{L}_1^s \mathbf{D}_2^{-1/2} =  \mathbf{D}_2^{1/2} Q \Lambda Q^\top \mathbf{D}_2^{-1/2} = \mathbf{U} \Lambda \mathbf{U}^{-1}
\end{equation}

where $Q\in \mathbb{R}^{\|\mathcal{E}^{(t)}\|\times \|\mathcal{E}^{(t)}\|}$ is the eigenvector of $\mathcal{L}_1^s$, $\|\mathcal{E}^{(t)}\|$ is the number of edges in graph $G^{(t)}$, the diagonal matrix $\Lambda\in \mathbb{R}^{\|\mathcal{E}^{(t)}\|}$ records the corresponding eigenvalues. We can make use of the right eigenvector $q_R = \mathbf{D}_2^{1/2} q$ as the edges spectrum.


The eigenvector with zero eigenvalue is the kernel of the Hodge Laplacian, which is the harmonic 1-forms. The harmonic 1-forms can be used to infer the underlying differentiation trees.
