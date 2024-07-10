from omegaconf import OmegaConf


def save_config(cfg, save_path):
    OmegaConf.save(config=cfg, f=save_path)


def load_config(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    return cfg


def generate_innovation_script(cfg, save_dir):
    # Get input configuration formatted.
    cmd_str = config_to_terminal_command(cfg)

    # Create bash script to run.
    breakpoint
    pass

    # Save invocation script.


def config_to_terminal_command(cfg):
    breakpoint()
    pass
