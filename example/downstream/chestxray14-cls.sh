# Linear evaluation on 50% labeled data
MODEL_DIR="model/downstream/chestx-ray14-cls/camcontrast/labeled=0.5"

PRETRAIN_PATH="model/upstream/chestx-ray14/camcontrast-patch/pytorch_model.bin"
IMAGES="data/upstream/chestx-ray14/images"
TRAIN_LABELS="data/upstream/chestx-ray14/processed/train.csv"
DEV_LABELS="data/upstream/chestx-ray14/processed/val.csv"
TEST_LABELS="data/upstream/chestx-ray14/processed/test.csv"

BACKBONE="u-net-encoder"
BATCH_SIZE=40
EPOCHS=60
LEARNING_RATE=0.1
WEIGHT_DECAY=0.0

# train
python script/downstream/chestxray14-cls.py \
  --model_dir ${MODEL_DIR} \
  --gpu_ids 0 \
  --num_workers 3 \
  --encoder_backbone $BACKBONE \
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
  --logging_steps 50 \
  --image_dir "${IMAGES}" \
  --train_label_path "${TRAIN_LABELS}" \
  --eval_label_path "${DEV_LABELS}" \
  --merge_dataset \
  --do_train \
  --freeze_weights \
  --labeled_ratio 0.5 \
  --pretrain_model_path $PRETRAIN_PATH

# test
python script/downstream/chestxray14-cls.py \
  --model_dir ${MODEL_DIR} \
  --image_dir "${IMAGES}" \
  --eval_label_path "${TEST_LABELS}" \
  --gpu_ids 0 \
  --num_workers 3 \
  --per_device_eval_batch_size $BATCH_SIZE \
  --do_eval \
  --encoder_backbone $BACKBONE
