#!/bin/bash

for video_no in {1..15}
do
    for task_no in {1..4}
    do
        echo "Executing for video_no: $video_no, task_no: $task_no"
        start_time=$(date +%s)
        python -m sp_eyegan.train_ehtask_event_model -video=$video_no -task=$task_no
        python -m sp_eyegan.train_ehtask_event_model -video=$video_no -task=$task_no -event_type=saccade
        end_time=$(date +%s)
        execution_time=$((end_time - start_time))
        echo "Execution time: $execution_time seconds"
        echo "-----------------------------"
    done
done
