python eval_forecasting_helper.py \
    --forecast test_result/03-15-16-58/forecasted_trajectories.pkl \
    --gt       test_result/03-15-16-58/gt_trajectories.pkl \
    --metrics --viz --save_dir ./vis_output \
    --data_root /root/autodl-tmp/dataset/interm_data --split val