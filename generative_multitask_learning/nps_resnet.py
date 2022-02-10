from utils import *

class NPSResnet(nn.Module):
    def __init__(self, dataset, seed, task_idxs, n_classes_list, net_class):
        super(NPSResnet, self).__init__()
        self.nets = nn.ModuleList()
        for task_idx, n_classes in zip(task_idxs, n_classes_list):
            net = net_class()
            net.fc = nn.Linear(net.fc.in_features, n_classes)
            net.load_state_dict(torch.load(os.path.join("results", dataset, "stl", "cross_stitch_resnet",
                f"t={task_idx}", f"s={seed}", "optimal_weights.pt")))
            self.nets.append(net)

    def forward(self, x):
        return [net(x) for net in self.nets]
