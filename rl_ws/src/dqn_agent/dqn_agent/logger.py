from pathlib import Path

import wandb
import torch.utils.tensorboard as tensorboard


class LOGGER:
    def __init__(self, project, project_name, config, project_dir, enable=True, use_wandb=True):
        self.enable = enable

        self.writer = None

        if self.enable:
            logs_dir = project_dir / "logs"
            logs_dir.mkdir(exist_ok=True)

            if use_wandb:
                wandb.init(
                    project=project,
                    name=project_name,
                    dir=project_dir,
                    sync_tensorboard=True,
                    config=config,
                )

            self.writer = tensorboard.SummaryWriter(logs_dir)
            self.writer.add_text("Hyper Parameters", "|Param|Value|\n|-|-|\n%s" % ("\n".join([f"|{param}|{value}|" for param, value in config.items()])))

    def Log(self, episode, log_data):
        if not self.enable:
            return

        for key, value in log_data.items():
            if value is not None:
                self.writer.add_scalar(key, value, episode)

        self.writer.flush()
