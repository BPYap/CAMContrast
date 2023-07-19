# Download the dataset from https://github.com/nkicsl/OIA-ODIR

TRAIN_IMAGES="data/download/OIA-ODIR/Training Set/Images"
TRAIN_LABELS="data/download/OIA-ODIR/Training Set/Annotation/training annotation (English).xlsx"
DEV_IMAGES="data/download/OIA-ODIR/Off-site Test Set/Images"
DEV_LABELS="data/download/OIA-ODIR/Off-site Test Set/Annotation/off-site test annotation (English).xlsx"
TEST_IMAGES="data/download/OIA-ODIR/On-site Test Set/Images"
TEST_LABELS="data/download/OIA-ODIR/On-site Test Set/Annotation/on-site test annotation (English).xlsx"

OUTPUT_FOLDER="data/upstream/oia-odir"

python script/upstream/oia-odir/0_preprocess.py \
  --train_images "${TRAIN_IMAGES}" \
  --train_labels "${TRAIN_LABELS}" \
  --dev_images "${DEV_IMAGES}" \
  --dev_labels "${DEV_LABELS}" \
  --test_images "${TEST_IMAGES}" \
  --test_labels "${TEST_LABELS}" \
  --output_folder "${OUTPUT_FOLDER}"
