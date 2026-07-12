export TFDS_DATA_DIR=./datasets/

python main.py \
    --config configs/ve/imagenet32_ncsnpp_continuous.py \
    --mode eval \
    --workdir experiments/imagenet32_ve \
    --eval_folder eval \
    --config.eval.enable_sampling=False \
    --config.eval.enable_bpd=True \
    --config.eval.bpd_dataset=test \
    --config.eval.batch_size=500 \
    --config.eval.num_repeats=1

