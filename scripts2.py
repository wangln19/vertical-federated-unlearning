import os
import sys

# os.system('python main_pipeline.py --configs test1')
# os.system('python main_pipeline.py --wth_DP True --configs test1')
# os.system('python main_pipeline.py --regular_h 1e-8 --configs test1')
for random_remove_features_percentage in [0.04, 0.08, 0.12]:
# [0.04, 0.08, 0.12, 0.16, 0.20]
    for times in range(1):
        # os.system('python main_pipeline.py --configs test1 --random_remove_features_percentage ' + str(random_remove_features_percentage))
        # os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --regular_h 1e-8 --configs test_unlearning1')
        # os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --apply_R2S True --configs test_unlearning1')
        os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --wth_asynchronous_unlearning True --configs test_unlearning1 --seed 2042')
        # os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --configs test_unlearning1')
        # os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --wth_asynchronous_unlearning True --wth_DP True --configs test_unlearning1 --seed 2042')
        # os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --wth_DP True --configs test_unlearning1')
        os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --wth_asynchronous_unlearning True --apply_gradient_ascent True --configs test_unlearning1 --seed 2042')
        # os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --apply_gradient_ascent True --configs test_unlearning1')
        # os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --wth_asynchronous_unlearning True --wth_DP True --apply_gradient_ascent True --configs test_unlearning1 --seed 2042')
        # os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --wth_DP True --apply_gradient_ascent True --configs test_unlearning1')
        
        
# for times in range(1):
#     os.system('python main_pipeline.py --remove_specific_clients True')
#     # os.system(f'python unlearning_pipeline.py --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/ --regular_h 1e-8')
#     os.system(f'python unlearning_pipeline.py --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/ --apply_R2S True')
#     os.system(f'python unlearning_pipeline.py --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/ --wth_asynchronous_unlearning True')
#     os.system(f'python unlearning_pipeline.py --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/')
#     os.system(f'python unlearning_pipeline.py --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/ --wth_asynchronous_unlearning True --wth_DP True')
#     os.system(f'python unlearning_pipeline.py --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/ --wth_DP True')
#     os.system(f'python unlearning_pipeline.py --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/ --wth_asynchronous_unlearning True --apply_gradient_ascent True')
#     os.system(f'python unlearning_pipeline.py --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/ --apply_gradient_ascent True')
#     os.system(f'python unlearning_pipeline.py --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/ --wth_asynchronous_unlearning True --wth_DP True --apply_gradient_ascent True')
#     os.system(f'python unlearning_pipeline.py --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/ --wth_DP True --apply_gradient_ascent True')

