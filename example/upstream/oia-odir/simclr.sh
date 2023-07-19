MODEL_DIR="model/upstream/oia-odir/simclr"

TRAIN_IMAGES="data/upstream/oia-odir/train"
TRAIN_LABELS="data/upstream/oia-odir/train.csv"
DEV_IMAGES="data/upstream/oia-odir/dev"
DEV_LABELS="data/upstream/oia-odir/dev.csv"

BACKBONE="u-net-encoder"
BATCH_SIZE=60
EPOCHS=300
WARMUP_EPOCHS=100
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-4
TEMPERATURE=0.05
DIST_FILE="/home/dist-simclr"

rm $DIST_FILE

python script/upstream/oia-odir/simclr.py \
  --model_dir $MODEL_DIR \
  --gpu_ids "0,1" \
  --world_size 2 \
  --rank 0 \
  --local_rank 0 \
  --shared_file_path "${DIST_FILE}" \
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
  --train_labels $TRAIN_LABELS \
  --dev_images $DEV_IMAGES \
  --dev_labels $DEV_LABELS \
  --temperature $TEMPERATURE &

python script/upstream/oia-odir/simclr.py \
  --model_dir $MODEL_DIR \
  --gpu_ids "0,1" \
  --world_size 2 \
  --rank 1 \
  --local_rank 1 \
  --shared_file_path "${DIST_FILE}" \
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
  --train_labels $TRAIN_LABELS \
  --dev_images $DEV_IMAGES \
  --dev_labels $DEV_LABELS \
  --temperature $TEMPERATURE
