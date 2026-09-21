# Pingwen RT-DETRv2 Experiments

`tools/run_experiment.py` fine-tunes a model from a full pretrained checkpoint,
automatically evaluates its `best.pth` checkpoint on the held-out test split,
and keeps all artifacts in one output directory.

```bash
conda activate rtdetr
cd /home/tkz/code/github/RT-DETR/rtdetrv2_pytorch

python tools/run_experiment.py \
  configs/rtdetrv2/rtdetrv2_r18vd_pingwen.yml \
  --pretrained pretrained/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
  --output-dir output/rtdetrv2_r18vd_pingwen_seed0 \
  --device 1 \
  --seed 0
```

The output directory contains the training checkpoints and logs, the automatic
test log (`test_metrics.log`), the evaluated checkpoint path
(`test_checkpoint.txt`), and the supplied random seed (`seed.txt`).
After a successful test, the runner retains `best.pth` and deletes `last.pth`
and `checkpoint*.pth`; removed filenames are recorded in
`deleted_checkpoints.log`.

For multiple seeds, change both `--output-dir` and `--seed` on each run. The
runner is intended for a single-GPU process; use the native distributed launch
workflow for multi-GPU training.
