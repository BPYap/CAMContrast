# Fine-tuning on 50% labeled data
MODEL_DIR="model/downstream/siim-acr-seg/camcontrast/labeled=0.5"

PRETRAIN_PATH="model/upstream/chestx-ray14/camcontrast-patch/pytorch_model.bin"
TRAIN_MASKS="data/downstream/SIIM-ACR/processed/masks (514)"
TEST_MASKS="data/downstream/SIIM-ACR/processed/masks"
TRAIN_IMAGES="data/downstream/SIIM-ACR/train_png (514)"
TRAIN_LABELS="data/downstream/SIIM-ACR/processed/train.txt"
DEV_IMAGES="data/downstream/SIIM-ACR/train_png (514)"
DEV_LABELS="data/downstream/SIIM-ACR/processed/val.txt"
TEST_IMAGES="data/downstream/SIIM-ACR/test_png (514)"
TEST_LABELS="data/downstream/SIIM-ACR/processed/test.txt"

BATCH_SIZE=32
EPOCHS=60
LEARNING_RATE=0.01
WEIGHT_DECAY=1e-5

# train
python script/downstream/siim-acr-seg.py \
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
  --logging_steps 200 \
  --mask_dir "${TRAIN_MASKS}" \
  --train_image_dir "${TRAIN_IMAGES}" \
  --train_split "${TRAIN_LABELS}" \
  --eval_image_dir "${DEV_IMAGES}" \
  --eval_split "${DEV_LABELS}" \
  --merge_dataset \
  --do_train \
  --labeled_ratio 0.5 \
  --pretrain_model_path $PRETRAIN_PATH

# test
python script/downstream/siim-acr-seg.py \
  --model_dir ${MODEL_DIR} \
  --mask_dir "${TEST_MASKS}" \
  --eval_image_dir "${TEST_IMAGES}" \
  --eval_split "${TEST_LABELS}" \
  --gpu_ids 0 \
  --num_workers 5 \
  --per_device_eval_batch_size $BATCH_SIZE \
  --do_eval \
  --segmentation_architecture "u-net" \
  --encoder_backbone "u-net-encoder"

