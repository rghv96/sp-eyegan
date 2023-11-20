import subprocess
import time

def main():
    for video_no in range(1, 16):
        for task_no in range(1, 5):
            print(f"Executing for video_no: {video_no}, task_no: {task_no}")
            start_time = time.time()

            # Use subprocess to run the Python command
            # subprocess.run([
            #     "python", "-m", "sp_eyegan.create_event_data_from_ehtask",
            #     "-video", str(video_no),
            #     "-task", str(task_no)
            # ])

            # subprocess.run([
            #     "python", "-m", "sp_eyegan.create_scanpath_from_ehtask",
            #     "-video", str(video_no),
            #     "-task", str(task_no),
            #     "-type", "random"
            # ])

            # subprocess.run([
            #     "python", "-m", "sp_eyegan.train_ehtask_event_model",
            #     "-video", str(video_no),
            #     "-task", str(task_no),
            # ])
            # subprocess.run([
            #     "python", "-m", "sp_eyegan.train_ehtask_event_model",
            #     "-video", str(video_no),
            #     "-task", str(task_no),
            #     "-event_type", "saccade"
            # ])
            subprocess.run([
                "python", "-m", "sp_eyegan.create_synthetic_data_ehtask",
                "-video", str(video_no),
                "-task", str(task_no)
            ])

            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time} seconds")
            print("-----------------------------")

if __name__ == '__main__':
    # execute only if run as a script
    raise SystemExit(main())