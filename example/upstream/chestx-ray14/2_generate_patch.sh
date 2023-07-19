MODEL_PATH="model/upstream/chestx-ray14/heatmap_generator/pytorch_model.bin"
BACKBONE="u-net-encoder"

IMAGES="data/upstream/chestx-ray14/images"
TRAIN_LABELS="data/upstream/chestx-ray14/processed/train.csv"
DEV_LABELS="data/upstream/chestx-ray14/processed/val.csv"

OUTPUT_FOLDER="data/upstream/chestx-ray14-patch"
CROP_SIZE=224

python script/upstream/chestx-ray14/2_generate_patch.py \
  --model_path $MODEL_PATH \
  --encoder_backbone $BACKBONE \
	--images $IMAGES \
	--train_labels $TRAIN_LABELS \
	--dev_labels $DEV_LABELS \
  --output_folder $OUTPUT_FOLDER \
  --crop_size $CROP_SIZE
