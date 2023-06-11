import os

eps_time = .05
ratio_set = list(reversed([1.3,  1.4,  1.5]))
tolerance = list(reversed([11.5, 15.1, 23.7]))
alpha = .01

for ratio, tol in zip(ratio_set, tolerance):
    print(f"Running setting ratio={ratio}, tol={tol} ...")
    eps_treasure = ratio * eps_time
    os.system(f"python run_cate_mompo.py --epsilons \"{eps_treasure},{eps_time}\" --tolerance {tol} --alpha {alpha}")
