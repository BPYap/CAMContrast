MODEL_DIR="model/downstream/refuge-seg/camcontrast"

PRETRAIN_PATH="model/upstream/oia-odir/camcontrast-patch/pytorch_model.bin"
TRAIN_IMAGES="data/downstream/REFUGE/train/images (514)"
TRAIN_LABELS="data/downstream/REFUGE/train/segmentations (514)"
DEV_IMAGES="data/downstream/REFUGE/val/images (514)"
DEV_LABELS="data/downstream/REFUGE/val/segmentations (514)"
TEST_IMAGES="data/downstream/REFUGE/test/images (514)"
TEST_LABELS="data/downstream/REFUGE/test/segmentations"

BATCH_SIZE=5
EPOCHS=90
LEARNING_RATE=0.01
WEIGHT_DECAY=1e-5

# train
python script/downstream/refuge-seg.py \
  --model_dir ${MODEL_DIR} \
  --gpu_ids 0 \
  --num_workers 5 \
  --encoder_backbone "u-net-encoder" \
  --segmentation_architecture "u-net" \
  --seed 42 \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs $EPOCHS \
  --learning_rate $LEARNING_RATE \
  --lr_scheduler "cosine" \
  --optimizer sgd \
  --momentum 0.9 \
  --nesterov \
  --weight_decay $WEIGHT_DECAY \
  --logging_steps 100 \
  --train_image_dir "${TRAIN_IMAGES}" \
  --train_mask_dir "${TRAIN_LABELS}" \
  --eval_image_dir "${DEV_IMAGES}" \
  --eval_mask_dir "${DEV_LABELS}" \
  --merge_dataset \
  --do_train \
  --pretrain_model_path $PRETRAIN_PATH

# test
python script/downstream/refuge-seg.py \
  --model_dir ${MODEL_DIR} \
  --eval_image_dir "${TEST_IMAGES}" \
  --eval_mask_dir "${TEST_LABELS}" \
  --gpu_ids 0 \
  --num_workers 5 \
  --per_device_eval_batch_size $BATCH_SIZE \
  --do_eval \
  --segmentation_architecture "u-net" \
  --encoder_backbone "u-net-encoder"
