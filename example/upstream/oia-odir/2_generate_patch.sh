MODEL_PATH="model/upstream/oia-odir/heatmap_generator/pytorch_model.bin"
BACKBONE="u-net-encoder"

TRAIN_IMAGES="data/upstream/oia-odir/train"
TRAIN_LABELS="data/upstream/oia-odir/train.csv"
DEV_IMAGES="data/upstream/oia-odir/dev"
DEV_LABELS="data/upstream/oia-odir/dev.csv"

OUTPUT_FOLDER="data/upstream/oia-odir-patch"
CROP_SIZE=224

python script/upstream/oia-odir/2_generate_patch.py \
  --model_path $MODEL_PATH \
  --encoder_backbone $BACKBONE \
  --train_images $TRAIN_IMAGES \
  --train_labels $TRAIN_LABELS \
  --dev_images $DEV_IMAGES \
  --dev_labels $DEV_LABELS \
  --output_folder $OUTPUT_FOLDER \
  --crop_size $CROP_SIZE
