"""Analysis tools for interpreting superposition experiments.

Provides methods for identifying polysemantic neurons, measuring feature
interference, and visualizing linguistic structure in compressed representations.
"""

from superposition.analysis.max_activations import get_max_activating_examples
from superposition.analysis.interference import compute_interference_heatmap
from superposition.analysis.embeddings import plot_embeddings_by_pos
