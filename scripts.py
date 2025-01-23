import os
import sys

os.system('python main_pipeline.py --trained True')
os.system('python main_pipeline.py --wth_DP True --trained True')
# os.system('python main_pipeline.py --regular_h 1e-8 --trained True')
# for random_remove_features_percentage in [0.04, 0.08, 0.12, 0.16, 0.20]:
# # [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
#     for times in range(1):
#         os.system('python main_pipeline.py --random_remove_features_percentage ' + str(random_remove_features_percentage))
#         # os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --regular_h 1e-8')
#         os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --apply_R2S True')
#         os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --wth_asynchronous_unlearning True')
#         os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/')
#         os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --wth_asynchronous_unlearning True --wth_DP True')
#         os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --wth_DP True')
#         os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --wth_asynchronous_unlearning True --apply_gradient_ascent True')
#         os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --apply_gradient_ascent True')
#         os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --wth_asynchronous_unlearning True --wth_DP True --apply_gradient_ascent True')
#         os.system(f'python unlearning_pipeline.py --unlearning_features True --src_file_dir trained_models/random_remove_{random_remove_features_percentage}_features_retrain/ --wth_DP True --apply_gradient_ascent True')
        
        
for times in range(1):
    os.system('python main_pipeline.py --trained True --remove_specific_clients True')
    # os.system(f'python unlearning_pipeline.py --trained True --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/ --regular_h 1e-8')
    os.system(f'python unlearning_pipeline.py --trained True --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/ --apply_R2S True')
    # os.system(f'python unlearning_pipeline.py --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/ --wth_asynchronous_unlearning True')
    os.system(f'python unlearning_pipeline.py --trained True --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/')
    # os.system(f'python unlearning_pipeline.py --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/ --wth_asynchronous_unlearning True --wth_DP True')
    os.system(f'python unlearning_pipeline.py --trained True --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/ --wth_DP True')
    # os.system(f'python unlearning_pipeline.py --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/ --wth_asynchronous_unlearning True --apply_gradient_ascent True')
    os.system(f'python unlearning_pipeline.py --trained True --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/ --apply_gradient_ascent True')
    # os.system(f'python unlearning_pipeline.py --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/ --wth_asynchronous_unlearning True --wth_DP True --apply_gradient_ascent True')
    os.system(f'python unlearning_pipeline.py --trained True --unlearning_clients True --src_file_dir trained_models/remove_specific_clients_retrain/ --wth_DP True --apply_gradient_ascent True')

