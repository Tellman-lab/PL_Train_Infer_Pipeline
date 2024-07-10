import shutil
import argparse
from glob import glob

from tqdm import tqdm


def delete_failed_experiments(base_exp_dir):
    # Find all experiments
    exp_dirs = glob(base_exp_dir + '/*/*/', recursive=True)

    # Figure out if experiment failed early.
    # Experiments only have a .hydra folder, a fit.log file, and tensorboard_logs.
    early_fail_del_count = 0
    for exp_dir in tqdm(exp_dirs,
                        desc='Removing early failure experiments',
                        colour='green'):
        exp_file_paths = glob(exp_dir + '/*')

        # Note: /.hydra/ will not be found because of the '.' before hydra
        if len(exp_file_paths) <= 2:
            early_fail_del_count += 1
            shutil.rmtree(exp_dir)

    print(
        f'Number of early failure experiments found and deleted: [{early_fail_del_count}/{len(exp_dirs)}]'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_exp_dir', type=str)
    args = parser.parse_args()

    delete_failed_experiments(args.base_exp_dir)
