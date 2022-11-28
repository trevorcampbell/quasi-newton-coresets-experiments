#!/bin/bash

python3 main.py --data_num 10000 plot coreset_size mu_err --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Relative Mean Error"
python3 main.py --data_num 10000 plot coreset_size Sig_err --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Relative Log-Variance Error"
#python3 main.py --data_num 10000 plot coreset_size imq_stein --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "IMQ Stein Discr."
#python3 main.py --data_num 10000 plot coreset_size gauss_stein --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Gauss Stein Discr."
#python3 main.py --data_num 10000 plot coreset_size imq_mmd --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "IMQ MMD Discr."
#python3 main.py --data_num 10000 plot coreset_size gauss_mmd --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Gauss MMD Discr."
python3 main.py --data_num 10000 plot coreset_size fklw --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Forward KL"
python3 main.py --data_num 10000 plot coreset_size rklw --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Reverse KL"
python3 main.py --data_num 10000 plot coreset_size t_build --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Build Time (s)"
python3 main.py --data_num 10000 plot coreset_size t_per_sample --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Per Sample Time (s)"






