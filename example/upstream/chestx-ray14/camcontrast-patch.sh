MODEL_DIR="model/upstream/chestx-ray14/camcontrast-patch"

IMAGES="data/upstream/chestx-ray14-patch/patches"
HEATMAPS="data/upstream/chestx-ray14-patch/heatmaps"
LABELS="data/upstream/chestx-ray14-patch/labels.csv"

BACKBONE="u-net-encoder"
BATCH_SIZE=120
EPOCHS=20
WARMUP_EPOCHS=5
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-4
TEMPERATURE=0.05

python script/upstream/chestx-ray14/camcontrast-patch.py \
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
  --images $IMAGES \
  --heatmaps $HEATMAPS \
  --labels $LABELS \
  --temperature $TEMPERATURE
