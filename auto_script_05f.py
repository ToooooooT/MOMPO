import os

eps_time = .05
ratio_set = [0.5, 1.0, 1.1, 1.2]
tolerance = [0.7, 0.7, 8.2, 8.2]
alpha = .01

for ratio, tol in zip(ratio_set, tolerance):
    print(f"Running setting ratio={ratio}, tol={tol} ...")
    eps_treasure = ratio * eps_time
    os.system(f"python run_cate_mompo.py --epsilons \"{eps_treasure:.4f},{eps_time:.2f}\" --tolerance {tol} --alpha {alpha}")
