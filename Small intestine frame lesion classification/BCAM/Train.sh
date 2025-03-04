#eval_type="mask"
eval_type="prob"

CUDA_VISIBLE_DEVICES=4 python main.py --dataset_name Crohn15to23 \
               --data_root "/mnt/minio/node77/caiyinqi/crohn_data/wsol_crohn_frames" \
               --metadata_root "/mnt/minio/node77/caiyinqi/crohn_data/wsol_crohn_frames/metadata" \
               --architecture resnet34 \
               --wsol_method "bcam" \
               --experiment_name resnet34_bas \
               --pretrained TRUE \
               --pretrained_path "/mnt/minio/node77/caiyinqi/vit_train/weights/best_resnet34_colorjitter.pth" \
               --num_val_sample_per_class 0 \
               --large_feature_map TRUE \
               --batch_size 128 \
               --epochs 20 \
               --lr 1e-3 \
               --lr_decay_frequency 15 \
               --weight_decay 1.00E-02 \
               --override_cache FALSE \
               --workers 16 \
               --box_v2_metric True \
               --iou_threshold_list 10 20 30 40 50 \
               --eval_checkpoint_type last \
               --rate_ff 1 \
               --rate_fb 1 \
               --rate_bf 0.5 \
               --rate_bb 0.5 \
               --save_dir 'train_log_prob' \
               --seed 4 \
               --target_layer "layer4" \
               --eval_type $eval_type \
               --eval_frequency 1 \
               --is_vis True


#####ResNet34
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name Crohn15to23 \
#                --data_root "/mnt/minio/node77/caiyinqi/crohn_data/wsol_crohn_frames" \
#                --metadata_root "/mnt/minio/node77/caiyinqi/crohn_data/wsol_crohn_frames/metadata" \
#                --architecture resnet34 \
#                --wsol_method "bcam" \
#                --experiment_name resnet34 \
#                --pretrained TRUE \
#                --pretrained_path "/mnt/minio/node77/caiyinqi/vit_train/weights/best_resnet34.pth" \
#                --num_val_sample_per_class 0 \
#                --large_feature_map TRUE \
#                --batch_size 128 \
#                --epochs 20 \
#                --lr 1e-3 \
#                --lr_decay_frequency 15 \
#                --weight_decay 1.00E-02 \
#                --override_cache FALSE \
#                --workers 16 \
#                --box_v2_metric True \
#                --iou_threshold_list 10 20 30 40 50 \
#                --eval_checkpoint_type last \
#                --rate_ff 1 \
#                --rate_fb 1 \
#                --rate_bf 0.5 \
#                --rate_bb 0.5 \
#                --save_dir 'train_log_prob' \
#                --seed 4 \
#                --target_layer "layer4" \
#                --eval_type $eval_type \
#                --eval_frequency 1 \
#                --is_vis True


# CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name Crohn15to23 \
#                --data_root "/mnt/minio/node77/caiyinqi/crohn_data/wsol_crohn_frames" \
#                --metadata_root "/mnt/minio/node77/caiyinqi/crohn_data/wsol_crohn_frames/metadata" \
#                --architecture resnet34 \
#                --wsol_method "bcam" \
#                --experiment_name resnet34_acol \
#                --pretrained TRUE \
#                --pretrained_path "/mnt/minio/node77/caiyinqi/vit_train/weights/best_resnet34_colorjitter.pth" \
#                --num_val_sample_per_class 0 \
#                --large_feature_map TRUE \
#                --batch_size 128 \
#                --epochs 20 \
#                --lr 1e-3 \
#                --lr_decay_frequency 15 \
#                --weight_decay 1.00E-02 \
#                --override_cache FALSE \
#                --workers 16 \
#                --box_v2_metric True \
#                --iou_threshold_list 10 20 30 40 50 \
#                --eval_checkpoint_type last \
#                --rate_ff 1 \
#                --rate_fb 1 \
#                --rate_bf 0.5 \
#                --rate_bb 0.5 \
#                --save_dir 'train_log_prob' \
#                --seed 4 \
#                --target_layer "layer4" \
#                --eval_type $eval_type \
#                --eval_frequency 1 \
#                --is_vis True


####resnet34+MFFmodel
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name Crohn15to23 \
#                --data_root "/mnt/minio/node77/caiyinqi/crohn_data/wsol_crohn_frames" \
#                --metadata_root "/mnt/minio/node77/caiyinqi/crohn_data/wsol_crohn_frames/metadata" \
#                --architecture resnet34 \
#                --wsol_method "bcam" \
#                --experiment_name resnet34_MFF \
#                --pretrained TRUE \
#                --pretrained_path "/mnt/minio/node77/caiyinqi/vit_train/weights/best_resnet34.pth" \
#                --num_val_sample_per_class 0 \
#                --large_feature_map TRUE \
#                --batch_size 128 \
#                --epochs 20 \
#                --lr 5e-4 \
#                --lr_decay_frequency 15 \
#                --weight_decay 1.00E-02 \
#                --override_cache FALSE \
#                --workers 16 \
#                --box_v2_metric True \
#                --iou_threshold_list 10 20 30 40 50 \
#                --eval_checkpoint_type last \
#                --rate_ff 1 \
#                --rate_fb 1 \
#                --rate_bf 0.5 \
#                --rate_bb 0.5 \
#                --save_dir 'train_log_prob' \
#                --seed 4 \
#                --target_layer "layer4" \
#                --eval_type $eval_type \
#                --eval_frequency 1 \
#                --is_vis True


####resnet50+MFFmodel
# CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name Crohn15to23 \
#                --data_root "/mnt/minio/node77/caiyinqi/crohn_data/wsol_crohn_frames" \
#                --metadata_root "/mnt/minio/node77/caiyinqi/crohn_data/wsol_crohn_frames/metadata" \
#                --architecture resnet50 \
#                --wsol_method "bcam" \
#                --experiment_name resnet50_MFF \
#                --pretrained TRUE \
#                --pretrained_path "/mnt/minio/node77/caiyinqi/vit_train/weights/best_resnet50.pth" \
#                --num_val_sample_per_class 0 \
#                --large_feature_map TRUE \
#                --batch_size 96 \
#                --epochs 20 \
#                --lr 5e-4 \
#                --lr_decay_frequency 15 \
#                --weight_decay 1.00E-02 \
#                --override_cache FALSE \
#                --workers 16 \
#                --box_v2_metric True \
#                --iou_threshold_list 10 20 30 40 50 \
#                --eval_checkpoint_type last \
#                --rate_ff 1 \
#                --rate_fb 1 \
#                --rate_bf 0.5 \
#                --rate_bb 0.5 \
#                --save_dir 'train_log_prob' \
#                --seed 4 \
#                --target_layer "layer4" \
#                --eval_type $eval_type \
#                --eval_frequency 1 \
#                --is_vis True


# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name CUB \
#                --architecture resnet50 \
#                --wsol_method "bcam" \
#                --experiment_name resnet \
#                --pretrained TRUE \
#                --num_val_sample_per_class 0 \
#                --large_feature_map TRUE \
#                --batch_size 32 \
#                --epochs 20 \
#                --lr 1.70E-4 \
#                --lr_decay_frequency 15 \
#                --weight_decay 1.00E-04 \
#                --override_cache FALSE \
#                --workers 16 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type last \
#                --rate_ff 1 \
#                --rate_fb 1 \
#                --rate_bf 1 \
#                --rate_bb 1 \
#                --save_dir 'train_log_prob' \
#                --seed 4 \
#                --target_layer "layer4" \
#                --eval_type $eval_type \
#                --eval_frequency 20

# #######VGG
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name CUB \
#                --architecture vgg16 \
#                --wsol_method "bcam" \
#                --experiment_name vgg \
#                --pretrained TRUE \
#                --num_val_sample_per_class 0 \
#                --large_feature_map FALSE \
#                --batch_size 32 \
#                --epochs 20 \
#                --lr 1.7e-5 \
#                --lr_decay_frequency 15 \
#                --weight_decay 5.00E-04 \
#                --override_cache FALSE \
#                --workers 16 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type last \
#                --rate_ff 1 \
#                --save_dir 'train_log_prob' \
#                --seed 10 \
#                --rate_fb 0.4 \
#                --rate_bf 0.4 \
#                --rate_bb 0.2 \
#                --num_head 100 \
#                --target_layer "conv6" \
#                --eval_type $eval_type \
#                --eval_frequency 20

# ######Inception
# CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name CUB \
#                --architecture inception_v3 \
#                --wsol_method "bcam" \
#                --experiment_name inception \
#                --pretrained TRUE \
#                --num_val_sample_per_class 0 \
#                --large_feature_map TRUE \
#                --batch_size 32 \
#                --epochs 50 \
#                --lr 1.7E-3 \
#                --lr_decay_frequency 15 \
#                --lr_bias_ratio 2 \
#                --weight_decay 5.00E-04 \
#                --override_cache FALSE \
#                --workers 16 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type last \
#                --rate_ff 1 \
#                --rate_fb 0.4 \
#                --rate_bf 0.4 \
#                --rate_bb 0.2 \
#                --num_head 100 \
#                --save_dir 'train_log_prob' \
#                --seed 25 \
#                --target_layer "SPG_A3_1b" \
#                --eval_frequency 50 \
#                --eval_type $eval_type


