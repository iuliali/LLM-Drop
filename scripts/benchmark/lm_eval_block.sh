#!/usr/bin/bash

port="21804"
GPUs="0,1,2,3,4,5,6,7"

model_names=("RoLlama2-7b-Instruct") # The model to be compressed.
drop_modules=("block") # "attn" ) #"block") # The modules to be dropped.
drop_nums=("4") # The number of dropped modules.

# tasks=("boolq") # "rte" "openbookqa" "piqa" "mmlu" "winogrande" "gsm8k" "hellaswag" "arc_challenge")
tasks=("xquad_ro")
num_fewshots=("0") # "0" "0" "0" "5" "5" "5" "10" "25")
# tasks=("mmlu")
# num_fewshots=("5")


for model_name in "${model_names[@]}"
do
    # Model should be in LLM-DROP

    for drop_module in "${drop_modules[@]}"
    do
        for drop_num in "${drop_nums[@]}"
        do
            cfg_path=../results_prune/"$model_name"-"$drop_module"_drop-discrete-drop"$drop_num"/checkpoint/config.json # PATH to the corresponding config.json file.
            cp -f "$cfg_path" ./"$model_name"/config.json # Replace the original config.json file.
            cp ../results_prune/"$model_name"-"$drop_module"_drop-discrete-drop"$drop_num"/checkpoint/*.py ./"$model_name"/ # Build the configuration and modeling files for remote code.
            echo "Eval the config of:"
            echo $cfg_path

            num_tasks=${#tasks[@]}
            for ((i=0; i<$num_tasks; i++)); do
                CUDA_VISIBLE_DEVICES=$GPUs accelerate launch --main_process_port $port  -m lm_eval \
                    --model hf \
                    --model_args pretrained=./${model_name},trust_remote_code=True,dtype="bfloat16" \
                    --tasks ${tasks[$i]} \
                    --batch_size 2 \
                    --verbosity DEBUG \
                    --device mps \
                    --write_out \
                    --log_samples \
                    --show_config \
                    --limit 100 \
                    --output_path ./100-${num_fewshots[$i]}shot_${tasks[$i]}_"$model_name"_drop"$drop_num"_"$drop_module".json >> 100-output_"$model_name"_drop"$drop_num"_"$drop_module".out
            done
        done
    done
done