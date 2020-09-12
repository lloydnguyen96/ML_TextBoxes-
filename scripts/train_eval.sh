# - model_dir: first location to look for model, no preprocessing required
# - checkpoint_path: second location to look for model, preprocessing required
#   (see function get_init_fn_for_scaffold. E.g., Take trained weights from
#   VGG16 to finetune another model in which VGG16 is the base network)
# two common scenarios:
# - training from scratch or finetuning our model from its checkpoint file:
# + model_dir=dir to write model (training from scratch) or dir to load
#   checkpoint file if checkpoint file exists and write model (finetuning our
#   models from its checkpoint file)
# + checkpoint_path=path to load model if checkpoint file doesn't exist in
# model_dir
# + model_scope=checkpoint_model_scope
# + checkpoint_exclude_scopes=None
# + ignore_missing_vars=False
# - training (finetuning) from other models
# + model_dir=directory to write finetuned model
# + checkpoint_path=path to load checkpoint file
# + model_scope=model name in graph we build in source code
# + checkpoint_model_scope=model name in checkpoint file
# + checkpoint_exclude_scopes=graph's scopes that didn't exist in checkpoint
#   file
# + ignore_missing_vars=True to train from scratch for missing vars
export TF_FORCE_GPU_ALLOW_GROWTH=true
# Value of CUDA_VISIBLE_DEVICES environment variable is shown in nvidia-smi as
# GPU column
export CUDA_VISIBLE_DEVICES=0
##########################################################################
################################ TRAINING ################################
python3 textboxes_plusplus.py\
    --mode='training'\
    --dataset_pattern='train-*'\
    --data_dir='./dataset/rctw/RCTW_p3/tfrecords'\
    --model_dir='./logs'\
    --checkpoint_path=None\
    --model_scope='textboxes_plusplus'\
    --checkpoint_model_scope='textboxes_plusplus'\
    --checkpoint_exclude_scopes=None\
    --ignore_missing_vars=False\
    --multi_gpu=True\
    --num_classes=2\
    --save_checkpoints_steps=500\
    --log_every_n_steps=10\
    --save_summary_steps=500\
    --keep_checkpoint_max=1\
    --max_number_of_steps=340000\
    --batch_size=16\
    --data_format='channels_first'\
    --learning_rate=1e-3\
    --decay_boundaries='3000, 200000, 240000, 270000, 290000'\
    --lr_decay_factors='0.1, 1, 0.5, 0.1, 0.05, 0.01'
##########################################################################
##########################################################################

##########################################################################
################################ EVALUATION ##############################
# python3 textboxes_plusplus.py\
#     --mode='evaluation'\
#     --dataset_pattern='val-*'\
#     --data_dir='./dataset/ctwd/tfrecords'\
#     --model_dir='./models/textboxes_plusplus_trained_ctwd_new'\
#     --checkpoint_path=None\
#     --model_scope='textboxes_plusplus'\
#     --checkpoint_model_scope='textboxes_plusplus'\
#     --checkpoint_exclude_scopes=None\
#     --ignore_missing_vars=False\
#     --multi_gpu=True\
#     --num_classes=2\
#     --log_every_n_steps=50\
#     --keep_checkpoint_max=1\
#     --train_epochs=1\
#     --batch_size=16\
#     --data_format='channels_first'\
#     --match_threshold=0.5\
#     --neg_threshold=0.5
##########################################################################
##########################################################################
