# Download the dataset from https://nihcc.app.box.com/v/ChestXray-NIHCC

TRAIN_VAL="data/upstream/chestx-ray14/train_val_list.txt"
TEST="data/upstream/chestx-ray14/test_list.txt"
LABELS="data/upstream/chestx-ray14/Data_Entry_2017_v2020.csv"
VAL_RATIO=0.2

python script/upstream/chestx-ray14/0_preprocess.py \
  --train_val_split_txt "${TRAIN_VAL}" \
  --test_split_txt "${TEST}" \
  --labels_csv "${LABELS}" \
  --val_ratio "${VAL_RATIO}"
