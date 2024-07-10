import os
import json
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf

from st_water_seg.fit import fit_model

ALL_REGION_NAMES = {
    'thp': {
        'S1': ['BEI', 'CMO', 'CTO', 'DKA', 'GIL', 'HTX', 'KTM', 'NSW', 'SPS'],
        'S2': ['CMO', 'DKA', 'GIL', 'HTX', 'KTM', 'NSW', 'PNE', 'SPS'],
        'PS':
        ['BEI', 'CMO', 'CTO', 'DKA', 'GIL', 'HTX', 'KTM', 'NSW', 'PNE', 'SPS']
    },
    'csdap': {
        'S2': [
            'Bangladesh', 'Bolivia', 'Colombia', 'Florence', 'Ghana', 'Harvey',
            'LittleRock', 'Mekong', 'Nebraska', 'Nepal', 'Nigeria', 'NorthAL',
            'Paraguay', 'RedRiverNorth', 'Somalia', 'Spain', 'Tulsa', 'USA',
            'Uzbekistan'
        ],
        'S1': [
            'Bangladesh', 'Bolivia', 'Colombia', 'Florence', 'Ghana', 'Harvey',
            'LittleRock', 'Mekong', 'Nebraska', 'Nepal', 'Nigeria', 'NorthAL',
            'Paraguay', 'RedRiverNorth', 'Somalia', 'Spain', 'Tulsa', 'USA',
            'Uzbekistan'
        ],
        'PS': [
            'Bangladesh', 'Bolivia', 'Colombia', 'Florence', 'Ghana', 'Harvey',
            'LittleRock', 'Mekong', 'Nebraska', 'Nepal', 'Nigeria', 'NorthAL',
            'Paraguay', 'RedRiverNorth', 'Somalia', 'Spain', 'Tulsa', 'USA',
            'Uzbekistan'
        ],
    },
    'combined': {
        # 'PS': [
        #     'Bangladesh', 'Bolivia', 'Colombia', 'Florence', 'Ghana', 'Harvey',
        #     'LittleRock', 'Mekong', 'Nebraska', 'Nepal', 'Nigeria', 'NorthAL',
        #     'Paraguay', 'RedRiverNorth', 'Somalia', 'Spain', 'Tulsa', 'USA',
        #     'Uzbekistan', 'BEI', 'CMO', 'CTO', 'DKA', 'GIL', 'HTX', 'KTM',
        #     'NSW', 'PNE', 'SPS'
        # ],
        'PS': [
            'Bangladesh', 'Bolivia', 'Colombia', 'Florence', 'Ghana', 'Harvey',
            'LittleRock', 'Mekong', 'Nebraska', 'Nepal', 'Nigeria', 'NorthAL',
            'Paraguay', 'RedRiverNorth', 'Somalia', 'Spain', 'Tulsa', 'USA',
            'Uzbekistan'
        ],
        'S1': [
            'Bolivia', 'Ghana', 'India', 'Mekong', 'Nigeria', 
            'Pakistan', 'Paraguay', 'Somalia', 'Spain', 'Sri-Lanka', 'USA'
        ],
        'S2': [
            'Bangladesh', 'Bolivia', 'Colombia', 'Florence', 'Ghana', 'Harvey',
            'LittleRock', 'Mekong', 'Nebraska', 'Nepal', 'Nigeria', 'NorthAL',
            'Paraguay', 'RedRiverNorth', 'Somalia', 'Spain', 'Tulsa', 'USA',
            'Uzbekistan'
        ],
        'L8': [
            'Bangladesh', 'Bolivia', 'Colombia', 'Ghana', 'Harvey',
            'LittleRock', 'Mekong', 'Nebraska', 'Nepal', 'Nigeria', 'NorthAL',
            'Paraguay', 'RedRiverNorth', 'Somalia', 'Spain', 'Tulsa', 'USA',
            'Uzbekistan'
        ]
    },
    's1f11': {
        # 'S1': [
        #     'Nigeria', 'Somalia', 'Mekong', 'Spain', 'Sri-Lanka', 'Bolivia',
        #     'India', 'Pakistan', 'Paraguay', 'USA', 'Ghana'
        # ],
        'S1': [
            'Bangladesh', 'Florence', 'Nebraska', 'NorthAL', 'RedRiverNorth'
        ],
        # 'S1': [
        #     'Florence'
        # ],
        'S2': [
            'Nigeria', 'Somalia', 'Mekong', 'Spain', 'Sri-Lanka', 'Bolivia',
            'India', 'Pakistan', 'Paraguay', 'USA', 'Ghana'
        ]
    }
}


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def cross_validate(cfg):
    dataset_name = cfg.dataset.name
    sensor_name = cfg.dataset.sensor
    region_names = ALL_REGION_NAMES[dataset_name][sensor_name]

    region_scores = {}
    for region_name in tqdm(region_names,
                            desc='Cross validate experiment',
                            colour='green'):
        

        # Overwrite the eval_region setting.
        cfg.eval_region = region_name

        # Update experiment directory.
        exp_dir = os.path.join(os.getcwd(), region_name)
        os.makedirs(exp_dir, exist_ok=True)
        print(f'Experiment directory: {exp_dir}')
        print(f'Experiment directory: {exp_dir}')
        print("---------------------")
        print('impact S1 original 300 b8')
        print("---------------------")
        print("---------------------")
       

        # Save updated config file.
        exp_cfg_path = os.path.join(exp_dir, 'config.yaml')
        OmegaConf.save(config=cfg, f=exp_cfg_path)

        # Train model.
        best_model_path = fit_model(cfg=cfg, overwrite_exp_dir=exp_dir)

        # Get the best validation score.
        best_validation_score = float(best_model_path.split('=')[-1][:-5])

        # Record score for this region.
        region_scores[region_name] = best_validation_score

    # Save results.
    # TODO: Come up with better place to save results.
    base_dir = '/'.join(
        os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
    save_name = f'./cross_val_{cfg.dataset.name}_{cfg.dataset.sensor}.json'
    save_path = os.path.join(base_dir, save_name)
    json.dump(region_scores, open(save_path, 'w'), indent=2)


if __name__ == '__main__':
    cross_validate()
    
