# Input configuration.
TRAIN_IMAGE_SIZE=384
# Anchor configuration.
ALL_ANCHOR_SCALES=\
    [(30.,),
     (38.4,),
     (89.6,),
     (140.8,),
     (192.,),
     (243.2,),
     (294.4,)]
ALL_EXTRA_SCALES=\
    [(33.94,),
     (58.66,),
     (112.32,),
     (164.42,),
     (216.09,),
     (267.58,),
     (318.97,)]
ALL_ANCHOR_RATIOS=[
    (1., 2., .5, 3., 1./3, 4., 1./4, 5., 1./5), # 10
    (1., 2., .5, 3., 1./3, 4., 1./4, 5., 1./5), # 10
    (1., 2., .5, 3., 1./3, 4., 1./4, 5., 1./5), # 10
    (1., 2., .5, 3., 1./3, 4., 1./4, 5., 1./5), # 10
    (1., 2., .5, 3., 1./3, 4., 1./4), # 8
    (1., 2., .5, 3., 1./3), # 6
    (1., 2., .5)] # 4
ALL_LAYER_SHAPES=\
    [(48, 48),
     (24, 24),
     (12, 12),
     (6, 6),
     (4, 4),
     (2, 2),
     (1, 1)]
ALL_LAYER_STRIDES=[8, 16, 32, 64, 96, 192, 384] # based on feat_shapes
NUM_FEATURE_LAYERS=len(ALL_LAYER_SHAPES)
ANCHOR_OFFSETS=[0.5] * NUM_FEATURE_LAYERS
VERTICAL_OFFSETS=[0.5] * NUM_FEATURE_LAYERS
PRIOR_SCALING=[0.1, 0.1, 0.2, 0.2]
# Network architecture.
# Output configuration.
NUM_CLASSES=2
NUM_OFFSETS=12
