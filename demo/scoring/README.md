# Model scoring

This folder contains code to test UniDisc under various configurations/checkpoints. Specifically, we generate a set of images and captions including masks for each, and then call the server under various configurations. We use a set of reward models from `model_eval.py` to score each output. 

Here is an example workflow:

```bash
uv run demo/scoring/generate_input.py input/v1 --num_pairs 500 --mask_txt --mask_img

uv run demo/scoring/call_model.py --input_dir input/v1 --output_dir generated/v1 --num_pairs 200 --iterate_over_modes

uv run accelerate launch --main_process_port $RANDOM demo/scoring/generate_rewards.py --input_dir generated/v1 --output_file rewards_v1.json --batch_size 32

uv run demo/scoring/analyze_rewards.py rewards_v1.json --save_image
```