import glob
import hydra
import numpy as np
from unified_video_action.env_runner.base_image_runner import BaseImageRunner


def load_env_runner(cfg, output_dir):
    if "libero" in cfg.task.name:
        hdf5_files = glob.glob(cfg.task.dataset.dataset_path + "/*hdf5")

        env_runners = []
        for file in hdf5_files:
            # configure env
            env_runner: BaseImageRunner
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner, task_dir=file, output_dir=output_dir
            )
            assert isinstance(env_runner, BaseImageRunner)
            env_runners.append(env_runner)

            if cfg.training.debug:
                break
        return env_runners

    else:
        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
        assert isinstance(env_runner, BaseImageRunner)
        return env_runner


def env_rollout(cfg, env_runners, policy):
    step_log = {}
    if "libero" in cfg.task.name:
        for env_runner in env_runners:
            runner_log = env_runner.run(policy)
            step_log.update(runner_log)

        if cfg.checkpoint.topk.monitor_key == "test_mean_score":
            assert "test_mean_score" not in step_log
            all_test_mean_score = {
                k: v for k, v in step_log.items() if "test/" in k and "_mean_score" in k
            }
            step_log["test_mean_score"] = np.mean(list(all_test_mean_score.values()))

            all_train_mean_score = {
                k: v
                for k, v in step_log.items()
                if "train/" in k and "_mean_score" in k
            }
            step_log["train_mean_score"] = np.mean(list(all_train_mean_score.values()))
    else:
        env_runner = env_runners
        runner_log = env_runner.run(policy)
        step_log.update(runner_log)

        step_log["train_mean_score"] = runner_log["train/mean_score"]
        step_log["test_mean_score"] = runner_log["test/mean_score"]
    return step_log
