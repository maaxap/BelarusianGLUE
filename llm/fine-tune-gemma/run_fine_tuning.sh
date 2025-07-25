fine_tune () {
  python trl/examples/scripts/sft.py --model_name ./models/$1 --dataset_name ./llm_fine_tuning/prompts_be/$2 --dataset_text_field text --load_in_4bit --use_peft --learning_rate 2e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 64 --num_train_epochs 10 --logging_strategy epoch --evaluation_strategy epoch --save_strategy epoch --do_eval --dataset_test_split validation --save_total_limit 1 --metric_for_best_model eval_loss --output_dir ./llm_fine_tuning/output/$1/$2;
  python merge_lora.py ./models/$1 ./llm_fine_tuning/output/$1/$2;
  cp -Lr ./models/$1/ ./models/$1_$2;
  rm ./models/$1_$2/model*.safetensors;
  mv ./llm_fine_tuning/output/$1/$2/merged/model*.safetensors ./models/$1_$2;
}
fine_tune_all () {
  fine_tune $1 besls;
  fine_tune $1 belacola_in_domain;
  fine_tune $1 bewic;
  fine_tune $1 bewsc;
  fine_tune $1 bertewd;
}

fine_tune_all gemma-2-9b-it
