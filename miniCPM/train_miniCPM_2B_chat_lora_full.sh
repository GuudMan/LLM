num_gpus=1

deepspeed --num_gpus $num_gpus miniCPM_2B_chat_Lora_full_train.py \
    --deepspeed ./ds_config_miniCPM.json \
    --output_dir="./output/MiniCPM" \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --logging_steps=10 \
    --num_train_epochs=3 \
    --save_steps=500 \
    --learning_rate=1e-4 \
    --save_on_each_node=True \