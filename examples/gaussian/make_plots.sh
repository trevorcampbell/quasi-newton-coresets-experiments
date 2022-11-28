#!/bin/bash

python3 main.py --samples_inference 1000 --dataset synth_gauss_large plot coreset_size mu_err --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_y_type linear --plot_x_label "Coreset Size" --plot_y_label "Relative Mean Error"
python3 main.py --samples_inference 1000 --dataset synth_gauss_large plot coreset_size cwise_mu_err --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_y_type linear --plot_x_label "Coreset Size" --plot_y_label "Component-wise Relative Mean Error"
python3 main.py --samples_inference 1000 --dataset synth_gauss_large plot coreset_size logsig_diag_err --plot_toolbar  --groupby coreset_size --plot_type line --plot_legend alg --plot_y_type linear --plot_x_label "Coreset Size" --plot_y_label "Relative Log-Variance Error"
python3 main.py --samples_inference 1000 --dataset synth_gauss_large plot coreset_size cwise_logsig_diag_err --plot_toolbar  --groupby coreset_size --plot_type line --plot_legend alg --plot_y_type linear --plot_x_label "Coreset Size" --plot_y_label "Component-wise Relative Log-Variance Error"
python3 main.py --samples_inference 1000 --dataset synth_gauss_large plot coreset_size Sig_err --plot_toolbar  --groupby coreset_size --plot_type line --plot_legend alg --plot_y_type linear --plot_x_label "Coreset Size" --plot_y_label "Relative Cov Cholesky Error"
#python3 main.py --samples_inference 1000 --dataset synth_gauss plot coreset_size imq_stein --plot_toolbar  --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "IMQ Stein Discr."
#python3 main.py --samples_inference 1000 --dataset synth_gauss plot coreset_size gauss_stein --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Gauss Stein Discr."
#python3 main.py --samples_inference 1000 --dataset synth_gauss plot coreset_size imq_mmd --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "IMQ MMD Discr."
#python3 main.py --samples_inference 1000 --dataset synth_gauss plot coreset_size gauss_mmd --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Gauss MMD Discr."
python3 main.py --samples_inference 1000 --dataset synth_gauss_large plot coreset_size fklw --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Forward KL"
python3 main.py --samples_inference 1000 --dataset synth_gauss_large plot coreset_size rklw --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Reverse KL"
python3 main.py --samples_inference 1000 --dataset synth_gauss_large plot coreset_size t_build --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Build Time (s)"
python3 main.py --samples_inference 1000 --dataset synth_gauss_large plot coreset_size t_per_sample --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Per Sample Time (s)"

#python3 main.py plot Ms fklw --groupby Ms --summarize trial --plot_type line --plot_x_type linear --plot_legend alg --plot_x_label "Iterations" --plot_y_label "Forward KL"
#python3 main.py plot csizes fklw --groupby Ms --summarize trial --plot_type line --plot_x_type linear --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Forward KL"
#python3 main.py plot cputs fklw --groupby Ms --summarize trial --plot_type line --plot_x_type linear --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "Forward KL"
#
#python3 main.py plot Ms mu_errs --summarize trial --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label "Relative Mean Error"
#python3 main.py plot csizes mu_errs --summarize trial --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Relative Mean Error"
#python3 main.py plot cputs mu_errs --summarize trial --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "Relative Mean Error"
#
#python3 main.py plot Ms Sig_errs --summarize trial --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label "Relative Covariance Error"
#python3 main.py plot csizes Sig_errs --summarize trial --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Relative Covariance Error"
#python3 main.py plot cputs Sig_errs --summarize trial --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "Relative Covariance Error"
#






