MODEL_DIR="model/upstream/oia-odir/supce-patch"

TRAIN_IMAGES="data/upstream/oia-odir-patch/patches"
TRAIN_LABELS="data/upstream/oia-odir-patch/labels.csv"

BACKBONE="u-net-encoder"
BATCH_SIZE=120
EPOCHS=60
WARMUP_EPOCHS=20
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-4

python script/upstream/oia-odir/supce-patch.py \
  --model_dir $MODEL_DIR \
  --gpu_ids "0" \
  --num_workers 5 \
  --encoder_backbone $BACKBONE \
  --seed 42 \
  --per_device_train_batch_size $BATCH_SIZE \
  --num_train_epochs $EPOCHS \
  --warmup_epochs $WARMUP_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --optimizer sgd \
  --momentum 0.95 \
  --weight_decay $WEIGHT_DECAY \
  --logging_steps 100 \
  --do_train \
  --train_images $TRAIN_IMAGES \
  --train_labels $TRAIN_LABELS
