# Input configuration.
TRAIN_IMAGE_SIZE=384
# Anchor configuration.
ALL_ANCHOR_SCALES=\
    [(30.,),
     (30.,),
     (90.,),
     (150.,),
     (210.,),
     (270.,)]
ALL_EXTRA_SCALES=\
    [(42.43,),
     (51.96,),
     (116.19,),
     (177.48,),
     (238.12,),
     (298.5,)]
ALL_ANCHOR_RATIOS=[
    (1., 2., .5, 3., 1./3, 4., 1./4, 5., 1./5),
    (1., 2., .5, 3., 1./3, 4., 1./4, 5., 1./5),
    (1., 2., .5, 3., 1./3, 4., 1./4, 5., 1./5),
    (1., 2., .5, 3., 1./3, 4., 1./4, 5., 1./5),
    (1., 2., .5, 3., 1./3, 4., 1./4, 5., 1./5),
    (1., 2., .5, 3., 1./3, 4., 1./4, 5., 1./5)]
ALL_LAYER_SHAPES=\
    [(48, 48),
     (24, 24),
     (12, 12),
     (6, 6),
     (4, 4),
     (2, 2)]
ALL_LAYER_STRIDES=[8, 16, 32, 64, 96, 192] # based on feat_shapes
NUM_FEATURE_LAYERS=len(ALL_LAYER_SHAPES)
ANCHOR_OFFSETS=[0.5] * NUM_FEATURE_LAYERS
VERTICAL_OFFSETS=[0.5] * NUM_FEATURE_LAYERS
PRIOR_SCALING=[0.1, 0.1, 0.2, 0.2]
# Network architecture.
# Output configuration.
NUM_CLASSES=2
NUM_OFFSETS=12
