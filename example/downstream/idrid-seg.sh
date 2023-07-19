PRETRAIN_PATH="model/upstream/oia-odir/camcontrast-patch/pytorch_model.bin"

BATCH_SIZE=5
EPOCHS=300
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-5


for t in "MA" "HE" "EX" "SE"
do
    MODEL_DIR="model/downstream/idrid-seg/camcontrast/$t"
    TRAIN_IMAGES="data/downstream/IDRiD/A. Segmentation/1. Original Images/a. Training Set (514)"
    TRAIN_LABELS="data/downstream/IDRiD/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/${t} (514)"
    TEST_IMAGES="data/downstream/IDRiD/A. Segmentation/1. Original Images/b. Testing Set (514)"
    TEST_LABELS="data/downstream/IDRiD/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/${t}"

    # train
    python script/downstream/idrid-seg.py \
      --model_dir ${MODEL_DIR} \
      --gpu_ids 0 \
      --num_workers 5 \
      --encoder_backbone "u-net-encoder" \
      --segmentation_architecture "u-net" \
      --seed 42 \
      --per_device_train_batch_size $BATCH_SIZE \
      --per_device_eval_batch_size $BATCH_SIZE \
      --num_train_epochs $EPOCHS \
      --warmup_epochs 100 \
      --learning_rate $LEARNING_RATE \
      --lr_scheduler "cosine" \
      --optimizer adam \
      --weight_decay $WEIGHT_DECAY \
      --logging_steps 100 \
      --train_image_dir "${TRAIN_IMAGES}" \
      --train_mask_dir "${TRAIN_LABELS}" \
      --task_id $t \
      --do_train \
      --pretrain_model_path $PRETRAIN_PATH

    # test
    python script/downstream/idrid-seg.py \
        --model_dir ${MODEL_DIR} \
        --eval_image_dir "${TEST_IMAGES}" \
        --eval_mask_dir "${TEST_LABELS}" \
        --task_id $t \
        --gpu_ids 0 \
        --num_workers 5 \
        --per_device_eval_batch_size $BATCH_SIZE \
        --do_eval \
        --segmentation_architecture "u-net" \
        --encoder_backbone "u-net-encoder"
done
