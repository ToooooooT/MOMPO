import os

eps_time = .01
ratio_set = [0.5, 1.0, 1.1, 1.2, 1.3,  1.4,  1.5]
tolerance = [0.7, 0.7, 8.2, 8.2, 11.5, 14.0, 22.4]
alpha = .01

for ratio, tol in zip(ratio_set, tolerance):
    print(f"Running setting ratio={ratio}, tol={tol} ...")
    eps_treasure = ratio * eps_time
    os.system(f"python run_cate_mompo.py --epsilons \"{eps_treasure},{eps_time}\" --tolerance {tol} --alpha {alpha}")
