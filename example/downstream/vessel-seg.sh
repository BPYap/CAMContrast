MODEL_DIR="model/downstream/vessel-seg/camcontrast"

PRETRAIN_PATH="model/upstream/oia-odir/camcontrast-patch/pytorch_model.bin"
CHASEDB1_TRAIN_IMAGES="data/downstream/CHASE_DB1/train/images"
CHASEDB1_TRAIN_LABELS="data/downstream/CHASE_DB1/train/segmentations"
DRIVE_TRAIN_IMAGES="data/downstream/DRIVE/train/images"
DRIVE_TRAIN_LABELS="data/downstream/DRIVE/train/segmentations"
STARE_TRAIN_IMAGES="data/downstream/STARE/train/images"
STARE_TRAIN_LABELS="data/downstream/STARE/train/segmentations"
CHASEDB1_EVAL_IMAGES="data/downstream/CHASE_DB1/test/images"
CHASEDB1_EVAL_LABELS="data/downstream/CHASE_DB1/test/segmentations"
DRIVE_EVAL_IMAGES="data/downstream/DRIVE/test/images"
DRIVE_EVAL_LABELS="data/downstream/DRIVE/test/segmentations"
STARE_EVAL_IMAGES="data/downstream/STARE/test/images"
STARE_EVAL_LABELS="data/downstream/STARE/test/segmentations"

BATCH_SIZE=5
EPOCHS=90
LEARNING_RATE=0.1
WEIGHT_DECAY=1e-5

# train
python script/downstream/vessel-seg.py \
  --model_dir ${MODEL_DIR} \
  --gpu_ids 0 \
  --num_workers 5 \
  --encoder_backbone "u-net-encoder" \
  --segmentation_architecture "u-net" \
  --seed 42 \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size 1 \
  --num_train_epochs $EPOCHS \
  --warmup_epochs 30 \
  --learning_rate $LEARNING_RATE \
  --lr_scheduler "cosine" \
  --optimizer sgd \
  --momentum 0.9 \
  --nesterov \
  --weight_decay $WEIGHT_DECAY \
  --logging_steps 100 \
  --chasedb1_train_image_dir "${CHASEDB1_TRAIN_IMAGES}" \
  --chasedb1_train_mask_dir "${CHASEDB1_TRAIN_LABELS}" \
  --drive_train_image_dir "${DRIVE_TRAIN_IMAGES}" \
  --drive_train_mask_dir "${DRIVE_TRAIN_LABELS}" \
  --stare_train_image_dir "${STARE_TRAIN_IMAGES}" \
  --stare_train_mask_dir "${STARE_TRAIN_LABELS}" \
  --do_train \
  --pretrain_model_path $PRETRAIN_PATH

# test
python script/downstream/vessel-seg.py \
  --model_dir ${MODEL_DIR} \
  --chasedb1_eval_image_dir "${CHASEDB1_EVAL_IMAGES}" \
  --chasedb1_eval_mask_dir "${CHASEDB1_EVAL_LABELS}" \
  --drive_eval_image_dir "${DRIVE_EVAL_IMAGES}" \
  --drive_eval_mask_dir "${DRIVE_EVAL_LABELS}" \
  --stare_eval_image_dir "${STARE_EVAL_IMAGES}" \
  --stare_eval_mask_dir "${STARE_EVAL_LABELS}" \
  --gpu_ids 0 \
  --num_workers 5 \
  --per_device_eval_batch_size 1 \
  --do_eval \
  --segmentation_architecture "u-net" \
  --encoder_backbone "u-net-encoder"
