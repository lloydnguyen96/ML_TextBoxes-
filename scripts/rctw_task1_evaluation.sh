export TF_FORCE_GPU_ALLOW_GROWTH=true
# Value of CUDA_VISIBLE_DEVICES environment variable is shown in nvidia-smi as
# GPU column
export CUDA_VISIBLE_DEVICES=0
##########################################################################
################################ EVALUATION ##############################
python3 rctw_task1_evaluation.py\
    --num_classes=2\
    --image_size=384\
    --data_format='channels_last'\
    --select_threshold=0.5\
    --min_size=4.\
    --nms_threshold=0.45\
    --nms_topk=20\
    --keep_topk=200\
    --checkpoint_path='./logs'\
    --model_scope='textboxes_plusplus'\
    --input_image_root='/home/loinguyenvan/Projects/OneDriveHUST/Datasets/RCTW'\
    --input_image_stem_patterns='icdar2017rctw_test/*.jpg'\
    --output_directory='evaluation/rctw_task1_evaluation/textboxes_plusplus_trained_rctw_p3_train_ms_nr16'
##########################################################################
##########################################################################
