from utils import *

N_COORDS = 4
GENDER_COL_IDX = 0

def cooccurrence_mat(train_ratio):
    task_idxs = [0, 1, 2, 3, 4, 5, 6, 7]
    trainval_df = make_df(task_idxs, "trainval")
    train_df = trainval_df.sample(frac=train_ratio, random_state=np.random.RandomState(0))
    train_df.replace(Y_MISSING_VALUE, np.nan, inplace=True)
    result = np.zeros((len(task_idxs), len(task_idxs)))
    for task_idx_row in task_idxs:
        for task_idx_col in task_idxs:
            if task_idx_row == task_idx_col:
                continue
            subset = train_df.iloc[:, [task_idx_row, task_idx_col]]
            result[task_idx_row, task_idx_col] = len(subset.dropna())
    return result

def make_log_prior(task_idxs, train_ratio, pseudocount):
    trainval_df = make_df(task_idxs, "trainval")
    train_df = trainval_df.sample(frac=train_ratio, random_state=np.random.RandomState(0))
    train_df.replace(Y_MISSING_VALUE, np.nan, inplace=True)
    train_df.dropna(inplace=True)
    train_df = train_df.values.astype(int)
    prior = np.full((2, 2), pseudocount)
    for train_df_elem in train_df:
        prior[tuple(train_df_elem)] += 1
    prior = prior / prior.sum()
    return torch.tensor(np.log(prior.flatten()))

def make_log_prior_indep(task_idxs, train_ratio):
    trainval_df = make_df(task_idxs, "trainval")
    train_df = trainval_df.sample(frac=train_ratio, random_state=np.random.RandomState(0))
    train_df.replace(Y_MISSING_VALUE, np.nan, inplace=True)
    p0 = np.nanmean(train_df.iloc[:, 0])
    p1 = np.nanmean(train_df.iloc[:, 1])
    p00 = (1 - p0) * (1 - p1)
    p01 = (1 - p0) * p1
    p10 = p0 * (1 - p1)
    p11 = p0 * p1
    prior = np.array(
        [[p00, p01],
         [p10, p11]])
    return torch.tensor(np.log(prior.flatten()))

def make_df(task_idxs, stage):
    fpath = os.path.join(os.environ["DATA_DPATH"], "attributes_dataset", stage + "_labels.txt")
    coord_col_names = [f"coord_{i}" for i in range(N_COORDS)]
    task_col_names = [
        "is_male",
        "has_long_hair",    # 0
        "has_glasses",      # 1
        "has_hat",          # 2
        "has_t-shirt",      # 3
        "has_long_sleeves", # 4
        "has_shorts",       # 5
        "has_jeans",        # 6
        "has_long_pants"]   # 7
    df = pd.read_table(fpath, sep=" ", header=None, index_col=0, names=coord_col_names + task_col_names)
    df.drop(coord_col_names + ["is_male"], axis=1, inplace=True)
    df.replace(0, np.nan, inplace=True)
    df.replace(-1, 0, inplace=True)
    df.replace(np.nan, Y_MISSING_VALUE, inplace=True)
    df = df.iloc[:, task_idxs]
    df = df.astype("long")
    fnames = list(df.index)
    fpaths = [os.path.join(os.environ["DATA_DPATH"], "attributes_dataset", stage, fname) for fname in fnames]
    df.index = fpaths
    return df

def make_attributes_of_people_data(task_idxs, train_ratio, batch_size, n_workers):
    trainval_df = make_df(task_idxs, "trainval")
    train_df = trainval_df.sample(frac=train_ratio, random_state=np.random.RandomState(0))
    val_df = trainval_df.drop(train_df.index)
    test_df = make_df(task_idxs, "test")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])
    train_data = DataLoader(DataFrameDataset(train_df, train_transform), shuffle=True, batch_size=batch_size,
        num_workers=n_workers, pin_memory=True)
    val_data = DataLoader(DataFrameDataset(val_df, eval_transform), batch_size=batch_size, num_workers=n_workers,
        pin_memory=True)
    test_data = DataLoader(DataFrameDataset(test_df, eval_transform), batch_size=batch_size, num_workers=n_workers,
        pin_memory=True)
    return train_data, val_data, test_data