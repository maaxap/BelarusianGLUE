#!/bin/bash

init=mdeberta-v3-base-continue
max_seq_length=512
cache_dir=./cache

data_dir=$cache_dir/spm_$max_seq_length
mkdir -p $data_dir

if [[ ! -e  $cache_dir/spm.model ]]; then
	wget -q https://huggingface.co/microsoft/mdeberta-v3-base/resolve/main/spm.model -O $cache_dir/spm.model
fi
if [[ ! -e  $data_dir/train.txt ]]; then
	python prepare_data_lowmem.py -i texts_hplt_be_uniq -o $data_dir/train.txt --max_seq_length $max_seq_length
fi
# This should not be required, since only --do_train is specified
# if [[ ! -e  $data_dir/valid.txt ]]; then
# 	python prepare_data.py -i texts.par.valid -o $data_dir/valid.txt --max_seq_length $max_seq_length
# fi
# if [[ ! -e  $data_dir/test.txt ]]; then
# 	python prepare_data.py -i texts.par.test -o $data_dir/test.txt --max_seq_length $max_seq_length
# fi

if [[ ! -e  pytorch_model.generator.bin ]]; then
	wget -q https://huggingface.co/microsoft/mdeberta-v3-base/resolve/main/pytorch_model.generator.bin
fi
if [[ ! -e  pytorch_model.bin ]]; then
	wget -q https://huggingface.co/microsoft/mdeberta-v3-base/resolve/main/pytorch_model.bin
fi

python -m DeBERTa.apps.run \
	--tag $init \
	--do_train \
	--max_seq_len $max_seq_length \
	--dump 10000 \
	--task_name RTD \
	--data_dir $data_dir \
	--vocab_path $cache_dir/spm.model \
	--vocab_type spm \
	--output_dir ./results/$init/RTD \
	--num_train_epochs 1 \
	--model_config rtd_base.json \
	--warmup 10000 \
	--learning_rate 2e-5 \
	--train_batch_size 4 \
	--init_generator ./pytorch_model.generator.bin \
	--init_discriminator ./pytorch_model.bin \
	--decoupled_training True \
	--fp16 True
