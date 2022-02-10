from utils import *
from copy import deepcopy
from attributes_of_people_data import make_attributes_of_people_data
from taskonomy_data import make_taskonomy_data
from cross_stitch_resnet import CrossStitchOptimizer, CrossStitchResnet
from nps_resnet import NPSResnet
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18, resnet34, resnet50, densenet121, ResNet

gin.external_configurable(make_attributes_of_people_data)
gin.external_configurable(make_taskonomy_data)
gin.external_configurable(CrossStitchResnet)
gin.external_configurable(NPSResnet)
gin.external_configurable(resnet18)
gin.external_configurable(resnet34)
gin.external_configurable(resnet50)
gin.external_configurable(densenet121)
gin.external_configurable(Adam)
gin.external_configurable(SGD)
gin.external_configurable(CrossStitchOptimizer)
gin.external_configurable(CosineAnnealingLR)

@gin.configurable
class Experiment:
    def __init__(self,
                 save_dpath,
                 seed,
                 stages,
                 main_task_idx,
                 make_data,
                 net_class,
                 n_classes_list,
                 optimizer,
                 lr_scheduler,
                 n_epochs,
                 n_early_stop_epochs=20,
                 log_prior_fpath=None,
                 log_prior_mult=None,
                 is_generative=False):
        self.save_dpath = save_dpath
        set_seed(seed)
        self.stages = stages
        self.main_task_idx = main_task_idx
        self.n_epochs = n_epochs
        self.n_early_stop_epochs = n_early_stop_epochs
        self.log_prior_mult = log_prior_mult
        self.is_generative = is_generative
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_data, self.val_data, self.test_data = make_data()
        self.n_tasks = len(n_classes_list)
        self.net = net_class()
        if self.n_tasks == 1:
            if isinstance(self.net, ResNet):
                self.net.fc = nn.Linear(self.net.fc.in_features, n_classes_list[0])
            self.optimizer = optimizer(self.net.parameters())
        else:
            if isinstance(self.net, CrossStitchResnet):
                self.optimizer = optimizer(self.net)
            else:
                self.optimizer = optimizer(self.net.parameters())
        self.scheduler = lr_scheduler(self.optimizer, n_epochs)
        self.optimal_val_loss = np.Inf
        self.optimal_val_epoch = 0
        self.optimal_weights = deepcopy(self.net.state_dict())
        self.net.to(self.device)
        if os.path.exists(self.save_dpath):
            self.load_checkpoint()
        else:
            os.makedirs(self.save_dpath)
            self.epoch = 0
        if log_prior_fpath is not None:
            self.log_prior = torch.load(log_prior_fpath).numpy()
            self.log_prior = self.log_prior.reshape(1, *n_classes_list)
    '''
    Train for a specified number of epochs, while intermittently saving checkpoints and keeping track of the optimal
    weights with respect to validation performance. At the end of training, load the optimal weights and perform inference
    on the test set.
    '''

    def save_checkpoint(self):
        '''
        Save the current state.
        '''
        self.net.to("cpu")
        checkpoint = {
            "random_state_np": np.random.get_state(),
            "random_state_pt": torch.get_rng_state(),
            "random_state": random.getstate(),
            "net_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "optimal_val_loss": self.optimal_val_loss,
            "optimal_val_epoch": self.optimal_val_epoch,
            "epoch": self.epoch}
        self.net.to(self.device)
        save_file(checkpoint, os.path.join(self.save_dpath, "checkpoint.pkl"))
        torch.save(self.optimal_weights, os.path.join(self.save_dpath, "optimal_weights.pt"))

    def load_checkpoint(self):
        '''
        Load a previously saved state.
        '''
        print("Loading checkpoint")
        checkpoint = load_file(os.path.join(self.save_dpath, "checkpoint.pkl"))
        np.random.set_state(checkpoint["random_state_np"])
        torch.set_rng_state(checkpoint["random_state_pt"])
        random.setstate(checkpoint["random_state"])
        self.net.to("cpu")
        self.net.load_state_dict(checkpoint["net_state"])
        self.net.to(self.device)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.optimal_weights = torch.load(os.path.join(self.save_dpath, "optimal_weights.pt"))
        self.optimal_val_loss = checkpoint["optimal_val_loss"]
        self.optimal_val_epoch = checkpoint["optimal_val_epoch"]
        self.epoch = checkpoint["epoch"]

    def train_epoch(self):
        '''
        Train for a single epoch.
        '''
        epoch_loss = RunningAverage()
        self.net.train()
        for x_batch, y_batch in self.train_data:
            # Forward prop
            x_batch = x_batch.to(self.device)
            loss = 0
            unnorm_log_prob = self.net(x_batch)
            if not isinstance(unnorm_log_prob, list):
                unnorm_log_prob = [unnorm_log_prob]
            for task_idx, task_unnorm_log_prob in enumerate(unnorm_log_prob):
                task_label = y_batch[:, task_idx]
                valid_idxs = np.flatnonzero(task_label != Y_MISSING_VALUE)
                if len(valid_idxs) == 0:
                    continue
                task_unnorm_log_prob, task_label = task_unnorm_log_prob[valid_idxs], task_label[valid_idxs]
                task_label = task_label.to(self.device)
                task_loss = F.cross_entropy(task_unnorm_log_prob, task_label, reduction="none")
                epoch_loss.update(task_loss)
                loss += task_loss.mean()
            # Backprop
            if isinstance(loss, int):
                continue
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Summary
        self.log("train", epoch_loss())

    def eval_epoch(self, is_val):
        '''
        Evaluate on the validation or test set.
        '''
        epoch_loss = RunningAverage()
        eval_data = self.val_data if is_val else self.test_data
        y = [[] for _ in range(self.n_tasks)]
        marginal_log_pred = [[] for _ in range(self.n_tasks)]
        self.net.eval()
        with torch.no_grad():
            for x_batch, y_batch in eval_data:
                x_batch = x_batch.to(self.device)
                unnorm_log_pred = self.net(x_batch)
                if not isinstance(unnorm_log_pred, list):
                    unnorm_log_pred = [unnorm_log_pred]
                for task_idx, task_unnorm_log_pred in enumerate(unnorm_log_pred):
                    task_label = y_batch[:, task_idx]
                    y[task_idx].append(task_label)
                    marginal_log_pred[task_idx].append(np.log(softmax(task_unnorm_log_pred.cpu().numpy(), axis=1)))
                    valid_idxs = np.flatnonzero(task_label != Y_MISSING_VALUE)
                    task_unnorm_log_pred, task_label = task_unnorm_log_pred[valid_idxs], task_label[valid_idxs]
                    task_label = task_label.to(self.device)
                    task_loss = F.cross_entropy(task_unnorm_log_pred, task_label, reduction="none")
                    if task_idx == self.main_task_idx:
                        epoch_loss.update(task_loss)
        if is_val:
            loss = epoch_loss()
            self.log("val", loss)
            return loss
        else:
            y = np.c_[[np.concatenate(elem) for elem in y]].T
            marginal_log_pred = [np.concatenate(elem) for elem in marginal_log_pred]
            log_joint_pred = self.to_log_joint_pred(marginal_log_pred)
            if self.is_generative:
                log_joint_pred = self.to_generative_pred(log_joint_pred)
            class_pred = self.to_class_pred(log_joint_pred)
            save_file((y, class_pred), os.path.join(self.save_dpath, f"y_pred_{self.log_prior_mult}.pkl"))
            for task_idx in range(self.n_tasks):
                task_pred = class_pred[:, task_idx]
                task_label = y[:, task_idx]
                valid_idxs = np.where(task_label != Y_MISSING_VALUE)[0]
                task_pred, task_label = task_pred[valid_idxs], task_label[valid_idxs]
                if task_idx == self.main_task_idx:
                    acc = accuracy_score(task_label, task_pred)
            self.log(f"test_{self.log_prior_mult}", acc)

    def to_log_joint_pred(self, marginal_log_pred):
        log_joint_shape = [task_log_pred.shape[1] for task_log_pred in marginal_log_pred]
        n_examples = len(marginal_log_pred[0])
        log_joint_pred = np.full(log_joint_shape + [n_examples], np.nan)
        for flat_idx in range(np.prod(log_joint_shape)):
            unflat_idx = np.unravel_index(flat_idx, log_joint_shape)
            log_prob = 0
            for task_idx, class_idx in enumerate(unflat_idx):
                log_prob += marginal_log_pred[task_idx][:, class_idx]
            log_joint_pred[unflat_idx] = log_prob
        log_joint_pred = np.moveaxis(log_joint_pred, -1, 0)
        return log_joint_pred

    def to_generative_pred(self, log_joint_pred):
        return log_joint_pred - self.log_prior_mult * self.log_prior

    def to_class_pred(self, log_joint_pred):
        class_pred = []
        log_joint_shape = log_joint_pred.shape[1:]
        for pred_elem in log_joint_pred:
            class_pred.append(np.unravel_index(np.argmax(pred_elem), log_joint_shape))
        class_pred = np.array(class_pred)
        return class_pred

    def log(self, stage, value):
        summary_str = "{}, {}, {}".format(
            get_time(),
            self.epoch,
            value)
        write(os.path.join(self.save_dpath, f"{stage}_summary.txt"), summary_str)

    def run(self):
        '''
        See the constructor docstring.
        '''
        for epoch in range(self.epoch, self.n_epochs):
            if "train" in self.stages:
                self.train_epoch()
                self.scheduler.step()
            if "val" in self.stages:
                loss = self.eval_epoch(True)
                if loss < self.optimal_val_loss:
                    self.optimal_val_loss = loss
                    self.optimal_val_epoch = self.epoch
                    self.net.to("cpu")
                    self.optimal_weights = deepcopy(self.net.state_dict())
                    self.net.to(self.device)
            self.epoch += 1
            self.save_checkpoint()
            if self.epoch - self.optimal_val_epoch == self.n_early_stop_epochs:
                break
        if "test" in self.stages:
            self.net.to("cpu")
            self.net.load_state_dict(self.optimal_weights)
            self.net.to(self.device)
            self.eval_epoch(False)

if __name__ == "__main__":
    config_path = sys.argv[-1]
    gin.parse_config_file(config_path)
    Experiment().run()