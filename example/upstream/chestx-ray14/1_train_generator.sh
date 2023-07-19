MODEL_DIR="model/upstream/chestx-ray14/heatmap_generator"

IMAGES="data/upstream/chestx-ray14/images"
TRAIN_LABELS="data/upstream/chestx-ray14/processed/train.csv"
DEV_LABELS="data/upstream/chestx-ray14/processed/val.csv"

BACKBONE="u-net-encoder"
BATCH_SIZE=20
EPOCHS=60
WARMUP_EPOCHS=20
LEARNING_RATE=0.01
WEIGHT_DECAY=0.0
ATTN_LOSS_WEIGHT=0.01
DIST_FILE="/home/dist-generator"

rm $DIST_FILE

python script/upstream/chestx-ray14/1_train_generator.py \
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
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs $EPOCHS \
  --warmup_epochs $WARMUP_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --lr_scheduler "cosine" \
  --optimizer sgd \
  --momentum 0.95 \
  --nesterov \
  --weight_decay $WEIGHT_DECAY \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --logging_steps 1000 \
	--images $IMAGES \
	--train_labels $TRAIN_LABELS \
	--dev_labels $DEV_LABELS \
  --attention_map_loss_weight $ATTN_LOSS_WEIGHT &

python script/upstream/chestx-ray14/1_train_generator.py \
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
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs $EPOCHS \
  --warmup_epochs $WARMUP_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --lr_scheduler "cosine" \
  --optimizer sgd \
  --momentum 0.95 \
  --nesterov \
  --weight_decay $WEIGHT_DECAY \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --logging_steps 1000 \
	--images $IMAGES \
	--train_labels $TRAIN_LABELS \
	--dev_labels $DEV_LABELS \
  --attention_map_loss_weight $ATTN_LOSS_WEIGHT
