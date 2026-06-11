# Gaze Heads
###  [Project Website](https://gaze.baulab.info) | [Paper](https://gaze.baulab.info) | [Dataset](https://huggingface.co/datasets/baulab/openai-comic-strips) <br>
Official code implementation of "Gaze Heads: How VLMs Look at What They Describe" for the Qwen3-VL family.

***Find the attention heads that look at whatever the model is describing. Then point them somewhere else and the model describes that instead!*** <br>

## Setup
To set up your python environment:
```
conda create -n gazeheads python=3.10
conda activate gazeheads

cd gaze-heads
pip install -r requirements.txt
```
The steering evaluations use Claude as a judge, so you need to export your `ANTHROPIC_API_KEY` (ideally to your bashrc). Discovery and the trajectory plots need no API key.

## Data
To download our 500-strip [OpenAI comics dataset](https://huggingface.co/datasets/baulab/openai-comic-strips) from Hugging Face (500 six-panel strips generated with gpt-image-1, with per-panel captions):
```
python download_data.py
```
This exports the comics to `data/comics/`, one folder per comic:
```
comics/
  comic1/
    p1.png ... p6.png
  comic2/
    ...
```
You can use your own comics too: anything in this layout works. Point any script at your comics with `--comics-root` or set `GAZE_COMICS_ROOT` once in your environment. Panels are stitched into a single horizontal strip at load time.

## Discovering Gaze Heads
To discover the gaze heads of a model, run:
```
python 01_discover_gaze_heads.py --device cuda:0 --n-samples 500
```
You can discover gaze heads in under *10 mins* on a single A6000-class GPU! No training, no labels, just one forward pass per panel query. The ranking is saved to `logs/gaze_discovery/gaze_head_ranking.json` and every other script picks it up from there. Score plots and example head matrices land in `figures/gaze_discovery/`.

The paper protocol discovers heads on the COMICS corpus ([Iyyer et al. 2017](https://github.com/miyyer/comics)) and evaluates steering on the disjoint OpenAI strips. Download the raw panel images from the COMICS authors (heads up: ~65 GB):
```
wget https://obj.umiacs.umd.edu/comics/raw_panel_images.tar.gz
tar -xzf raw_panel_images.tar.gz
```
Then add `--use-raw` and strips are sampled as random consecutive 6-panel windows (seeded, reproducible):
```
python 01_discover_gaze_heads.py --use-raw --comics-root raw_panel_images --n-samples 500
```

## Watching Gaze Heads Track Narration
Gaze heads follow the story in real time: while the model narrates panel 1 they look at panel 1, and they jump to panel 2 right as the text moves on. To plot these attention staircases (gaze heads vs. random control heads):
```
python 02_narration_trajectory.py --device cuda:0 --max-comics 5
```

## Steering What the Model Describes
Steering adds a single pre-softmax bias on the gaze heads' attention: boost the target panel's image tokens, suppress the rest. Nothing is retrained, and fewer than 9% of the heads carry the lever.

### VQA steering
Ask one question about the whole strip and redirect the answer to any panel you choose:
```
python 03_steer_vqa.py --device cuda:0
```
This runs the full judged protocol over every (strip, target panel) pair with three conditions: the top-100 gaze heads, random non-gaze heads, and all heads. Steering the gaze heads redirects the answer to the chosen panel at ~80% (chance is 16.7%); non-gaze heads do nothing and all heads destroy generation. Results with bootstrap CIs are saved to `logs/steer_vqa/aggregate_results.json`. Use `--max-comics 10` for a quick run, and `--start-comic-idx` to shard across GPUs.

### Static narration steering
Ask the deliberately ambiguous question "What is happening in this panel of the comic strip?" without saying which panel. Unsteered, the model picks the first panel or summarizes the strip; with the gaze heads held on a target panel, it describes that panel:
```
python 04_steer_static_narration.py --device cuda:0
```
Each ~100-token answer gets a forced 1-of-6 panel match from the judge, and answers identical to the baseline count as misses. Use `--targets-per-strip 1` for a quick run.

### Dynamic narration steering
Switch the target panel every 50 tokens mid-generation and watch the model wrap up one panel and move to the next:
```
python 05_steer_dynamic_narration.py --device cuda:0
```
Each strip gets a random derangement schedule (never the default left-to-right order), and the script reports per-segment accuracy plus the Spearman correlation between your schedule and what the model actually described.

## Interactive Steering Notebook
For hands-on steering, open `interactive_steering.ipynb` from the repo root:
- pick a comic and a target panel from a dropdown: one question, six different answers
- type a schedule like `4,2,1,6,5,3` and watch the trajectory heatmap follow it
- load *any* image, drag a box over a region, and steer the description to it

Have fun!

## Citing our work
```bibtex
@article{gandikota2026gazeheads,
  title={Gaze Heads: How VLMs Look at What They Describe},
  author={Gandikota, Rohit and Bau, David},
  year={2026}
}
```
