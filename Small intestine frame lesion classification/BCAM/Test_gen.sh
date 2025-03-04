eval_type="prob"
#eval_type="mask"
###ResNet
CUDA_VISIBLE_DEVICES=1 python test_gen_cams_boxes.py --dataset_name Crohn15to23 \
               --data_root "/mnt/minio/node77/liuzheng/BCAM/wsol_gencam_imgs" \
               --metadata_root "/mnt/minio/node77/liuzheng/BCAM/wsol_gencam_imgs/metadate" \
               --architecture resnet34 \
               --pretrained False \
               --wsol_method "bcam" \
               --experiment_name resnet34_bas_gen_lz \
               --num_val_sample_per_class 0 \
               --large_feature_map TRUE \
               --batch_size 128 \
               --override_cache FALSE \
               --workers 16 \
               --box_v2_metric True \
               --iou_threshold_list 10 20 30 40 50 \
               --cam_curve_interval 0.001 \
               --eval_checkpoint_type last \
               --save_dir 'test_gen_Crohn15to23_'$eval_type \
               --check_path "/mnt/minio/node77/liuzheng/BCAM/BCAM-main/SingleObjectLocalization/train_log_prob/resnet34_bas/11_checkpoint.pth.tar" \
               --mode "test" \
               --seed 4 \
               --target_layer "layer4" \
               --eval_type $eval_type \
               --is_vis TRUE


###ResNet_MFF
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_name Crohn15to23 \
#                --data_root "/mnt/minio/node77/caiyinqi/crohn_data/wsol_crohn_frames" \
#                --metadata_root "/mnt/minio/node77/caiyinqi/crohn_data/wsol_crohn_frames/metadata" \
#                --architecture resnet34 \
#                --pretrained False \
#                --wsol_method "bcam" \
#                --experiment_name resnet34MFF_0dot1 \
#                --num_val_sample_per_class 0 \
#                --large_feature_map TRUE \
#                --batch_size 128 \
#                --override_cache FALSE \
#                --workers 16 \
#                --box_v2_metric True \
#                --iou_threshold_list 10 20 30 40 50 \
#                --cam_curve_interval 0.001 \
#                --eval_checkpoint_type last \
#                --save_dir 'test_log_Crohn15to23_'$eval_type \
#                --check_path "/mnt/minio/node77/caiyinqi/BCAM-main/SingleObjectLocalization/train_log_prob/resnet34_MFF/14_checkpoint.pth.tar" \
#                --mode "test" \
#                --seed 4 \
#                --target_layer "layer4" \
#                --eval_type $eval_type \
#                --is_vis TRUE


# CUDA_VISIBLE_DEVICES=4 python main.py --dataset_name CUB \
#                --architecture resnet50 \
#                --wsol_method "bcam" \
#                --experiment_name resnet50 \
#                --num_val_sample_per_class 0 \
#                --large_feature_map TRUE \
#                --batch_size 32 \
#                --override_cache FALSE \
#                --workers 16 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type last \
#                --save_dir 'test_log_CUB_'$eval_type \
#                --check_path "checkpoints/resnet_cub.pth.tar" \
#                --mode "test" \
#                --seed 4 \
#                --target_layer "layer4" \
#                --eval_type $eval_type \
#                --is_vis FALSE

# ###Vgg
# CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name CUB \
#                --architecture vgg16 \
#                --wsol_method "bcam" \
#                --experiment_name vgg \
#                --pretrained TRUE \
#                --num_val_sample_per_class 0 \
#                --large_feature_map FALSE \
#                --batch_size 32 \
#                --override_cache FALSE \
#                --workers 16 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type last \
#                --save_dir 'test_log_CUB_'$eval_type \
#                --check_path "checkpoints/vgg_cub.pth.tar" \
#                --mode "test" \
#                --seed 4 \
#                --target_layer "conv6" \
#                --eval_type $eval_type

# ###Inception
# CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name CUB \
#                --architecture inception_v3 \
#                --wsol_method "bcam" \
#                --experiment_name inception \
#                --pretrained TRUE \
#                --num_val_sample_per_class 0 \
#                --large_feature_map TRUE \
#                --batch_size 32 \
#                --override_cache FALSE \
#                --workers 16 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type last \
#                --save_dir 'test_log_CUB_'$eval_type \
#                --check_path "checkpoints/inception_cub.pth.tar" \
#                --mode "test" \
#                --seed 4 \
#                --target_layer "SPG_A3_1b" \
#                --eval_type $eval_type
