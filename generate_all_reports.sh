#!/bin/bash
set -e

CMD="python -m superposition analyze"
CKPT="checkpoints"

echo "=== Interference Heatmaps ==="

# Toy
echo "[1/14] toy_small"
$CMD --analysis interference --model toy --checkpoint $CKPT/toy_small.pt

# Translation
echo "[2/14] translation"
$CMD --analysis interference --model translation --checkpoint $CKPT/translation.pt


echo "[3/14] computation_abs"
$CMD --analysis interference --model computation --checkpoint $CKPT/computation_abs.pt

# Continuous thought (default + bottleneck variants)
echo "[4/14] continuous_thought"
$CMD --analysis interference --model continuous_thought --checkpoint $CKPT/continuous_thought.pt
for d in 32 64 128 256; do
    echo "[*/14] continuous_thought_d${d}"
    $CMD --analysis interference --model continuous_thought --preset continuous_thought_d${d} --checkpoint $CKPT/continuous_thought_d${d}.pt
done

# Coconut (default + no_bottleneck + bottleneck variants)
echo "[9/14] coconut"
$CMD --analysis interference --model coconut --checkpoint $CKPT/coconut.pt

echo "[10/14] coconut_no_bottleneck"
$CMD --analysis interference --model coconut --preset coconut_no_bottleneck --checkpoint $CKPT/coconut_no_bottleneck.pt
for d in 32 64 128 256; do
    echo "[*/14] coconut_d${d}"
    $CMD --analysis interference --model coconut --preset coconut_d${d} --checkpoint $CKPT/coconut_d${d}.pt
done

echo ""
echo "=== Embeddings & Activations (translation) ==="

echo "embeddings (t-SNE)"
$CMD --analysis embeddings --model translation --checkpoint $CKPT/translation.pt --method tsne

echo "embeddings (PCA)"
$CMD --analysis embeddings --model translation --checkpoint $CKPT/translation.pt --method pca

echo "max activations"
$CMD --analysis activations --model translation --checkpoint $CKPT/translation.pt

echo ""
echo "=== Geometric Structure ==="

echo "geometry: toy_small"
$CMD --analysis geometry --model toy --checkpoint $CKPT/toy_small.pt

# echo "geometry: computation_abs"
# $CMD --analysis geometry --model computation --checkpoint $CKPT/computation_abs.pt

echo ""
echo "=== Phase Diagram (trains from scratch) ==="
echo "phase_diagram"
$CMD --analysis phase_diagram --model toy --sparsity-steps 20 --importance-steps 20

echo ""
echo "All reports saved to images/"
