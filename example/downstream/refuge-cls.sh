MODEL_DIR="model/downstream/refuge-cls/camcontrast"

PRETRAIN_PATH="model/upstream/oia-odir/camcontrast-patch/pytorch_model.bin"
TRAIN_IMAGES="data/downstream/REFUGE/train/images (350)"
TRAIN_LABELS="data/downstream/REFUGE/train/labels.csv"
DEV_IMAGES="data/downstream/REFUGE/val/images (350)"
DEV_LABELS="data/downstream/REFUGE/val/labels.csv"
TEST_IMAGES="data/downstream/REFUGE/test/images (350)"
TEST_LABELS="data/downstream/REFUGE/test/labels.csv"

BACKBONE="u-net-encoder"
BATCH_SIZE=40
EPOCHS=150
LEARNING_RATE=0.0003
WEIGHT_DECAY=0.033333

# train
python script/downstream/refuge-cls.py \
  --model_dir ${MODEL_DIR} \
  --gpu_ids 0 \
  --num_workers 5 \
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
  --logging_steps 100 \
  --train_image_dir "${TRAIN_IMAGES}" \
  --train_label_path "${TRAIN_LABELS}" \
  --eval_image_dir "${DEV_IMAGES}" \
  --eval_label_path "${DEV_LABELS}" \
  --merge_dataset \
  --do_train \
  --pretrain_model_path $PRETRAIN_PATH

# test
python script/downstream/refuge-cls.py \
  --model_dir ${MODEL_DIR} \
  --eval_image_dir "${TEST_IMAGES}" \
  --eval_label_path "${TEST_LABELS}" \
  --gpu_ids 0 \
  --num_workers 5 \
  --per_device_eval_batch_size $BATCH_SIZE \
  --do_eval \
  --encoder_backbone $BACKBONE
