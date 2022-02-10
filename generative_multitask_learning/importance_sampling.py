from attributes_of_people_data import *
from scipy.stats import weightedtau

def make_importance_weights(y, iid_prior, ood_prior):
    importance_weights = []
    for y_elem in y:
        marginalizing_idxs = []
        for task_idx, class_idx in enumerate(y_elem):
            if class_idx == Y_MISSING_VALUE:
                marginalizing_idxs.append(slice(iid_prior.shape[task_idx]))
            else:
                marginalizing_idxs.append(class_idx.item())
        log_iid_prior = np.log(iid_prior[tuple(marginalizing_idxs)].sum())
        log_ood_prior = np.log(ood_prior[tuple(marginalizing_idxs)].sum())
        importance_weights.append(np.exp(log_ood_prior - log_iid_prior))
    importance_weights = np.array(importance_weights)
    importance_weights = importance_weights / importance_weights.sum()
    return importance_weights

def evaluate(y, pred, main_task_idx, importance_weights):
    task_y = y[:, main_task_idx]
    task_pred = pred[:, main_task_idx]
    valid_idxs = np.flatnonzero(task_y != Y_MISSING_VALUE)
    task_y, task_pred, task_importance_weights = task_y[valid_idxs], task_pred[valid_idxs], importance_weights[
        valid_idxs]
    return accuracy_score(task_y, task_pred, sample_weight=task_importance_weights)

def main(args):
    rng = np.random.RandomState(0)
    y_pred = {"y": [], "pred_0.0": []}
    for alpha in args.alphas:
        y_pred[f"pred_{alpha}"] = []
    for seed in range(args.n_seeds):
        y_seed, discrim_pred_seed = load_file(os.path.join(args.y_pred_dpath, f"s={seed}", "y_pred_None.pkl"))
        y_pred["y"].append(y_seed)
        y_pred["pred_0.0"].append(discrim_pred_seed)
        for alpha in args.alphas:
            _, generative_pred_seed = load_file(os.path.join(args.y_pred_dpath, f"s={seed}", f"y_pred_{alpha}.pkl"))
            y_pred[f"pred_{alpha}"].append(generative_pred_seed)
    for k, v in y_pred.items():
        y_pred[k] = np.concatenate(v)
    iid_log_prior = torch.load(args.log_prior_fpath).numpy()
    iid_prior = softmax(iid_log_prior).reshape(args.n_classes_list)

    results = {"distance": [], "acc_0.0": []}
    for alpha in args.alphas:
        results[f"acc_{alpha}"] = []
    for _ in range(args.n_reps):
        ood_log_prior = iid_log_prior.copy()
        # Shuffle
        shuffle_prob = rng.uniform(0, args.shuffle_prob_ub)
        shuffle_idxs = rng.choice(len(iid_log_prior), int(2 * shuffle_prob * len(iid_log_prior)), replace=False)
        if len(shuffle_idxs) > 0:
            if len(shuffle_idxs) % 2 > 0:
                shuffle_idxs = shuffle_idxs[:-1]
            lhs_idxs, rhs_idxs = np.split(shuffle_idxs, 2)
            lhs_values = ood_log_prior[lhs_idxs]
            ood_log_prior[lhs_idxs] = ood_log_prior[rhs_idxs]
            ood_log_prior[rhs_idxs] = lhs_values
        # Perturb
        sd = rng.uniform(EPSILON, args.sd_ub)
        ood_log_prior += rng.normal(0, sd, len(ood_log_prior))
        results["distance"].append(weightedtau(iid_log_prior, ood_log_prior)[0])
        ood_prior = softmax(ood_log_prior).reshape(args.n_classes_list)
        importance_weights = make_importance_weights(y_pred["y"], iid_prior, ood_prior)
        results["acc_0.0"].append(evaluate(y_pred["y"], y_pred["pred_0.0"], args.main_task_idx, importance_weights))
        for alpha in args.alphas:
            results[f"acc_{alpha}"].append(
                evaluate(y_pred["y"], y_pred[f"pred_{alpha}"], args.main_task_idx, importance_weights))
    for k, v in results.items():
        results[k] = np.array(v)
    save_file(results, os.path.join(args.y_pred_dpath, f"importance_sampling_mt={args.main_task_idx}.pkl"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--main_task_idx", type=int)
    parser.add_argument("--y_pred_dpath", type=str)
    parser.add_argument("--log_prior_fpath", type=str)
    parser.add_argument("--n_classes_list", nargs="+", type=int)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--shuffle_prob_ub", type=float, default=0.5)
    parser.add_argument("--sd_ub", type=float, default=5)
    parser.add_argument("--n_reps", type=int, default=1000)
    args = parser.parse_args()
    main(args)