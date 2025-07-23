# lm-evaluation-harness

This is a stripped-down version of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) that illustrates our approach to LLM evaluation with BelarusianGLUE (section 4.3 of the paper). The new configurations are stored in `lm-eval/tasks/bel_glue`. Use the `default` version to evaluate local LLMs via log probabilities of the next tokens, and the `generative` version to evaluate commercial LLMs available through their APIs.

Invocation example:
```
lm_eval --model hf --model_args pretrained=<model ID> --tasks bel_glue --device cuda:0 --batch_size 1 --log_samples --output_path <path> --trust_remote_code
```

# fine-tune-gemma

These are the scripts related to fine-tuning [Gemma 2 9B](https://huggingface.co/google/gemma-2-9b-it) on the training sets of BelarusianGLUE:
- Generate fine-tuning data with prompts in Belarusian or English using `make_train_sets_fine_tuning_{be,en}.py`.
- Download and unpack [trl](https://github.com/huggingface/trl).
- Obtain Gemma 2 9B and put in `./models`.
- Edit `run_fine_tuning.sh` to specify prompting language (`prompts_be` or `prompts_en`) and run to produce fine-tuned versions of the model, one per dataset.
