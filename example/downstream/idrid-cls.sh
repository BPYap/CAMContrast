MODEL_DIR="model/downstream/idrid-cls/camcontrast"

PRETRAIN_PATH="model/upstream/oia-odir/camcontrast-patch/pytorch_model.bin"
TRAIN_IMAGES="data/downstream/IDRiD/B. Disease Grading/1. Original Images/a. Training Set (350)"
TRAIN_LABELS="data/downstream/IDRiD/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
TEST_IMAGES="data/downstream/IDRiD/B. Disease Grading/1. Original Images/b. Testing Set (350)"
TEST_LABELS="data/downstream/IDRiD/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"

BACKBONE="u-net-encoder"
BATCH_SIZE=40
EPOCHS=300
LEARNING_RATE=0.0003
WEIGHT_DECAY=3.333333

# train
python script/downstream/idrid-cls.py \
  --model_dir ${MODEL_DIR} \
  --gpu_ids 0 \
  --num_workers 5 \
  --encoder_backbone $BACKBONE \
  --seed 42 \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs $EPOCHS \
  --warmup_epochs 100 \
  --learning_rate $LEARNING_RATE \
  --lr_scheduler "cosine" \
  --optimizer adam \
  --weight_decay $WEIGHT_DECAY \
  --logging_steps 50 \
  --train_image_dir "${TRAIN_IMAGES}" \
  --train_label_path "${TRAIN_LABELS}" \
  --do_train \
  --pretrain_model_path $PRETRAIN_PATH

# test
python script/downstream/idrid-cls.py \
  --model_dir ${MODEL_DIR} \
  --eval_image_dir "${TEST_IMAGES}" \
  --eval_label_path "${TEST_LABELS}" \
  --gpu_ids 0 \
  --num_workers 5 \
  --per_device_eval_batch_size $BATCH_SIZE \
  --do_eval \
  --encoder_backbone $BACKBONE

