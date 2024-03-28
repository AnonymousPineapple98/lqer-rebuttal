#!/bin/bash

source ~/.bashrc

function check_retval() {
    retval=$1
    msg=$2
    if [ $retval -ne 0 ]; then
        echo "❌ Failed: $msg"
        exit $retval
    fi
}

function group() {
    model=$1
    model_awq=$2
    batch_size=$3
    max_seq_len=$4
    num_samples=$5
    rank=$6
    num_hidden_layers=$7

    conda run -n te --no-capture-output python profile_llama.py $model "llm-int4" \
        --batch-size $batch_size \
        --max-seq-len $max_seq_len \
        --num-samples $num_samples \
        --num-hidden-layers $num_hidden_layers
    check_retval $? "llm-int4 $model"

    conda run -n te --no-capture-output python profile_llama.py $model "fp16" \
        --batch-size $batch_size \
        --max-seq-len $max_seq_len \
        --num-samples $num_samples \
        --num-hidden-layers $num_hidden_layers
    check_retval $? "fp16 $model"

    conda run -n te --no-capture-output python profile_llama.py $model "lqer-fp8fp16" \
        --batch-size $batch_size \
        --max-seq-len $max_seq_len \
        --num-samples $num_samples \
        --rank $rank \
        --num-hidden-layers $num_hidden_layers
    check_retval $? "lqer-fp8fp16 $model"

    conda run -n te --no-capture-output python profile_llama.py $model_awq "awq" \
        --batch-size $batch_size \
        --max-seq-len $max_seq_len \
        --num-samples $num_samples \
        --num-hidden-layers $num_hidden_layers
    check_retval $? "awq $model_awq"
}

# llama-30b
model=huggyllama/llama-30b
model_awq=TheBloke/LLaMA-30b-AWQ
declare -a batch_sizes=(1 2 3 4)
max_seq_len=1048
num_samples=48
rank=32
num_hidden_layers=-1

for batch_size in "${batch_sizes[@]}"; do
    group $model $model_awq $batch_size $max_seq_len $num_samples $rank $num_hidden_layers
    check_retval $? "group $model/$model_awq, bs=$batch_size, seq_len=$max_seq_len"
done
