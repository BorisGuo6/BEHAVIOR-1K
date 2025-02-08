import json
import os
import signal
import subprocess
import sys
from dask.distributed import Client, as_completed
import fs.copy
from fs.multifs import MultiFS
import fs.path
from fs.tempfs import TempFS
import tqdm

from b1k_pipeline.utils import ParallelZipFS, PipelineFS, TMP_DIR, launch_cluster

WORKER_COUNT = 2
MAX_TIME_PER_PROCESS = 20 * 60  # 20 minutes

def run_on_scene(dataset_path, scene):
    python_cmd = ["python", "-m", "b1k_pipeline.usd_conversion.usdify_scenes_process", dataset_path, scene]
    cmd = ["micromamba", "run", "-n", "omnigibson", "/bin/bash", "-c", "source /isaac-sim/setup_conda_env.sh && rm -rf /root/.cache/ov/texturecache && " + " ".join(python_cmd)]
    with open(f"/scr/ig_pipeline/logs/{scene}.log", "w") as f, open(f"/scr/ig_pipeline/logs/{scene}.err", "w") as ferr:
        try:
            p = subprocess.Popen(cmd, stdout=f, stderr=ferr, cwd="/scr/ig_pipeline", start_new_session=True)
            return p.wait(timeout=MAX_TIME_PER_PROCESS)
        except subprocess.TimeoutExpired:
            print(f'Timeout for {scene} ({MAX_TIME_PER_PROCESS}s) expired. Killing', file=sys.stderr)
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            return p.wait()

def main():
    with PipelineFS() as pipeline_fs, \
         ParallelZipFS("objects_usd.zip") as objects_fs, \
         ParallelZipFS("metadata.zip") as metadata_fs, \
         ParallelZipFS("scenes.zip") as scenes_fs, \
         TempFS(temp_dir=str(TMP_DIR)) as dataset_fs:
        with ParallelZipFS("scenes_json.zip", write=True) as out_fs:
            # Copy everything over to the dataset FS
            print("Copying input to dataset fs...")
            multi_fs = MultiFS()
            multi_fs.add_fs("metadata", metadata_fs, priority=1)
            multi_fs.add_fs("objects", objects_fs, priority=1)
            multi_fs.add_fs("scenes", scenes_fs, priority=1)

            # Copy all the files to the output zip filesystem.
            total_files = sum(1 for f in multi_fs.walk.files())
            with tqdm.tqdm(total=total_files) as pbar:
                fs.copy.copy_fs(multi_fs, dataset_fs, on_copy=lambda *args: pbar.update(1))

            print("Launching cluster...")
            dask_client = launch_cluster(WORKER_COUNT)

            # Start the batched run
            scenes = [x for x in dataset_fs.listdir("scenes")]
            print("Queueing scenes.")
            print("Total count: ", len(scenes))
            futures = {}
            for scene in scenes:
                worker_future = dask_client.submit(
                    run_on_scene,
                    dataset_fs.getsyspath("/"),
                    scene,
                    retries=2,
                    pure=False)
                futures[worker_future] = scene

            # Wait for all the workers to finish
            print("Queued all scenes. Waiting for them to finish...")
            logs = {}
            for future in tqdm.tqdm(as_completed(futures.keys()), total=len(futures)):
                try:
                    out = future.result()
                    # logs[futures[future]] = out.stdout.decode("utf-8")
                except subprocess.CalledProcessError as e:
                    print("Error in worker")
                    # print(e.stdout.decode("utf-8"))

            # Move the USDs to the output FS
            print("Copying scene JSONs to output FS...")
            usd_glob = sorted(
                {x.path for x in dataset_fs.glob("scenes/*/json/")} |
                {x.path for x in dataset_fs.glob("scenes/*/layout/")})
            for item in tqdm.tqdm(usd_glob):
                fs.copy.copy_fs(dataset_fs.opendir(item), out_fs.makedirs(item))

            print("Done processing. Archiving things now.")

        # Save the logs
        with pipeline_fs.pipeline_output().open("usdify_scenes.json", "w") as f:
            json.dump(logs, f)

        # At this point, out_temp_fs's contents will be zipped. Save the success file.
        pipeline_fs.pipeline_output().touch("usdify_scenes.success")

if __name__ == "__main__":
    main()
