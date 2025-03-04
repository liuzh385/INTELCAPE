#### Please perform the following steps:

**Step 1**: Clone the project

```
git clone https://github.com/liuzh385/INTELCAPE.git
cd INTELCAPE
pip install -r requirements.txt
```

------

**Step 2**: Run the small intestine segmentation code

##### The "Capsule Endoscopy Segmentation" folder contains the code for the first step of small intestine segmentation:

- `expconfigs_new/exp02_f0_pretrain_th_gau.yaml` is used for pretraining the encoder for a three-class classification. Run the training with:`CUDA_VISIBLE_DEVICES=0 python main.py --config expconfigs_new/exp02_f0_pretrain_th_gau.yaml`
- `expconfigs_new/exp03_f0_resTFE_gau.yaml` is used to train CNN+Transformer for three-class classification. Run the training with:`CUDA_VISIBLE_DEVICES=0 python main.py --config expconfigs_new/exp03_f0_resTFE_gau.yaml`
- `expconfigs_new/test_exp03_f0_resTFE_gau.yaml` is used for binary classification to search for the start and end points of the small intestine. Run the test with:`CUDA_VISIBLE_DEVICES=0 python main.py --config expconfigs_new/test_exp03_f0_resTFE_gau.yaml --test`

For video file format, store the video in the folder (the format should be `.avi`). The naming format is `[name's last character corresponding to Unicode 20xx_xx_xx]`. Use the script `prepare/find_frame.py` to find the start frames of the stomach, small intestine, and colon in the video, based on the time recorded at the top right of the video frame (since frames may drop during video transmission, we can only find the corresponding frame based on the time recorded). The CSV file must match the format of `Data/data_example.csv`.

------

**Step 3**: Run the lesion Classification code

##### The "Small Intestine Frame Lesion Classification" folder contains the code for the second step of lesion identification:

- Run binary classification training with:
  `CUDA_VISIBLE_DEVICES=0 python train.py`
- Run binary classification testing with:
  `CUDA_VISIBLE_DEVICES=0 python infer.py`

Ensure that the image files are saved in the same format as in the data folder.

##### The "SingleObjectLocalization" folder contains the weakly supervised recognition code. The `Train.sh` script is used for training, and the `Test.sh` script is used for testing, generating heatmaps and bounding boxes, and obtaining test metrics.

------

**Step 4**: Run the Crohn's disease diagnosis code

##### The "Lesion Small Intestine Frame Crohn's Diagnosis" folder contains the code for the third step of Crohn's disease recognition:

- `expconfigs_new/test01_get_framsnpy.yaml` extracts the features (as `.npy` files) for each patient's small intestine frames and the predicted lesion probability for each frame. The feature extraction uses the EfficientNet weights from the previous step. Run the extraction with:`CUDA_VISIBLE_DEVICES=0 python main.py --config expconfigs_new/test01_get_framsnpy.yaml --test`
- `expconfigs_new/test02_gettop2000.yaml` divides the small intestine segments into 4 parts, and selects the 500 frames with the highest predicted lesion probability from each segment. Run the segmentation with:`CUDA_VISIBLE_DEVICES=0 python main.py --config expconfigs_new/test02_gettop2000.yaml --test`
- `expconfigs_new/exp02_focalloss.yaml` trains a TF2 model for Crohn's disease binary classification. Run the training with:`CUDA_VISIBLE_DEVICES=0 python main.py --config expconfigs_new/exp02_focalloss.yaml`
- `expconfigs_new/test03_focalloss.yaml` tests the Crohn's disease classification performance. Run the test with:`CUDA_VISIBLE_DEVICES=0 python main.py --config expconfigs_new/test03_focalloss.yaml --test`
