from utils import *

def make_object_class_idxs():
    with open(os.path.join(os.environ["DATA_DPATH"], "taskonomy", "imagenet1000_clsidx_to_labels.txt")) as f:
        idx_to_name = ast.literal_eval(f.read())
    name_to_idx = {v: k for k, v in idx_to_name.items()}

    lines = []
    with open(os.path.join(os.environ["DATA_DPATH"], "taskonomy", "imagenet_selected_classes.txt")) as f:
        for line in f:
            line = line.strip()
            line = line.split(" ", 1)[1]
            line = line.rstrip(",")
            lines.append(line)
    idxs = [name_to_idx[line] for line in lines]
    return idxs

def make_scene_class_idxs():
    df = pd.read_csv(os.path.join(os.environ["DATA_DPATH"], "taskonomy", "Scene hierarchy - Places365.csv"))
    df = df[["category", 'workplace (office building, factory, lab, etc.)', "home or hotel"]]
    df = df[(df['workplace (office building, factory, lab, etc.)'] == 1) | (df["home or hotel"] == 1)]
    idxs = np.array(df.index)
    return idxs

def make_df(task_idxs, stage):
    fpath = os.path.join(os.environ["DATA_DPATH"], "taskonomy", f"{stage}_df.pkl")
    if os.path.exists(fpath):
        df = load_file(fpath)
    else:
        object_class_idxs = make_object_class_idxs()
        scene_class_idxs = make_scene_class_idxs()
        locations = pd.read_csv(os.path.join(os.environ["DATA_DPATH"], "taskonomy", "splits_taskonomy",
            "train_val_test_medium.csv"))
        locations = locations.id[locations[stage] == 1]
        df = {"x_fpaths": [], "y_object": [], "y_scene": []}
        x_fpaths = []
        for location in locations:
            x_dpath = os.path.join(os.environ["DATA_DPATH"], "taskonomy", "taskonomy_medium", "rgb", "taskonomy", location)
            try:
                x_fpaths += [os.path.join(x_dpath, fname) for fname in os.listdir(x_dpath)]
            except FileNotFoundError:
                pass
        for x_fpath in x_fpaths:
            y_object_fpath = x_fpath.replace("rgb", "class_object").replace("png", "npy")
            y_scene_fpath = x_fpath.replace("rgb", "class_scene").replace("png", "npy")
            y_scene_fpath = y_scene_fpath.replace("class_scene.npy", "class_places.npy")
            try:
                y_object = np.argmax(np.load(y_object_fpath)[object_class_idxs])
                y_scene = np.argmax(np.load(y_scene_fpath)[scene_class_idxs])
                df["x_fpaths"].append(x_fpath)
                df["y_object"].append(y_object)
                df["y_scene"].append(y_scene)
            except: # This is not good practice
                pass
        df = pd.DataFrame(df)
        df.index = df.x_fpaths
        df.drop("x_fpaths", axis=1, inplace=True)
        df = df.astype("long")
        save_file(df, fpath)
    df = df.iloc[:, task_idxs]
    return df

def make_log_prior(pseudocount):
    train_df = make_df([0, 1], "train").values
    prior = np.full((100, 64), pseudocount)
    for train_df_elem in train_df:
        prior[tuple(train_df_elem)] += 1
    prior = prior / prior.sum()
    return torch.tensor(np.log(prior.flatten()))

def make_taskonomy_data(task_idxs, batch_size, n_workers):
    train_df = make_df(task_idxs, "train")
    val_df = make_df(task_idxs, "val")
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
        num_workers=n_workers, pin_memory=True, collate_fn=filter_none)
    val_data = DataLoader(DataFrameDataset(val_df, eval_transform), batch_size=batch_size, num_workers=n_workers,
        pin_memory=True, collate_fn=filter_none)
    test_data = DataLoader(DataFrameDataset(test_df, eval_transform), batch_size=batch_size, num_workers=n_workers,
        pin_memory=True, collate_fn=filter_none)
    return train_data, val_data, test_data