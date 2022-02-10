from utils import *

class CrossStitchOptimizer(torch.optim.Adam):
    def __init__(self, net, lr, cross_stitch_lr, **kwargs):
        params = []
        for stl_net in net.nets:
            params.append({"params": stl_net.parameters()})
        params.append({"params": net.initial_unit, "lr": cross_stitch_lr})
        params.append({"params": net.layer1_unit, "lr": cross_stitch_lr})
        params.append({"params": net.layer2_unit, "lr": cross_stitch_lr})
        params.append({"params": net.layer3_unit, "lr": cross_stitch_lr})
        params.append({"params": net.layer4_unit, "lr": cross_stitch_lr})
        super(CrossStitchOptimizer, self).__init__(params, lr=lr, **kwargs)

class CrossStitchResnet(nn.Module):
    def __init__(self, dataset, seed, task_idxs, n_classes_list, net_class, layer_dims, cross_stitch_init):
        super(CrossStitchResnet, self).__init__()
        self.nets = nn.ModuleList()
        for task_idx, n_classes in zip(task_idxs, n_classes_list):
            net = net_class()
            net.fc = nn.Linear(net.fc.in_features, n_classes)
            net.load_state_dict(torch.load(os.path.join("results", dataset, "stl", f"t={task_idx}", f"s={seed}", "optimal_weights.pt")))
            self.nets.append(net)
        self.initial_unit = self.np_to_param(self.make_cross_stitch_unit(len(self.nets), 64, cross_stitch_init))
        self.layer1_unit = self.np_to_param(self.make_cross_stitch_unit(len(self.nets), layer_dims[0], cross_stitch_init))
        self.layer2_unit = self.np_to_param(self.make_cross_stitch_unit(len(self.nets), layer_dims[1], cross_stitch_init))
        self.layer3_unit = self.np_to_param(self.make_cross_stitch_unit(len(self.nets), layer_dims[2], cross_stitch_init))
        self.layer4_unit = self.np_to_param(self.make_cross_stitch_unit(len(self.nets), layer_dims[3], cross_stitch_init))

    def np_to_param(self, x):
        return nn.Parameter(torch.tensor(x, dtype=torch.float32))

    def make_cross_stitch_unit(self, n_nets, n_channels, cross_stitch_init):
        # Broadcast over batch, width, and height indices
        x = np.full((n_nets, n_nets, 1, n_channels, 1, 1), (1 - cross_stitch_init) / (n_nets - 1))
        for net_idx in range(n_nets):
            x[net_idx, net_idx, ...] = cross_stitch_init
        return x

    def cross_stitch(self, premerge, cross_stitch_unit):
        postmerge = []
        for net_cross_stitch_unit in cross_stitch_unit:
            net_postmerge = 0
            for net_cross_stitch_unit_elem, activations in zip(net_cross_stitch_unit, premerge):
                net_postmerge += net_cross_stitch_unit_elem * activations
            postmerge.append(net_postmerge)
        return postmerge

    def forward(self, x):
        premerge = []
        for net in self.nets:
            net_output = net.conv1(x)
            net_output = net.bn1(net_output)
            net_output = net.relu(net_output)
            net_output = net.maxpool(net_output)
            premerge.append(net_output)
        postmerge = self.cross_stitch(premerge, self.initial_unit)

        premerge = []
        for net_id, net in enumerate(self.nets):
            premerge.append(net.layer1(postmerge[net_id]))
        postmerge = self.cross_stitch(premerge, self.layer1_unit)

        premerge = []
        for net_id, net in enumerate(self.nets):
            premerge.append(net.layer2(postmerge[net_id]))
        postmerge = self.cross_stitch(premerge, self.layer2_unit)

        premerge = []
        for net_id, net in enumerate(self.nets):
            premerge.append(net.layer3(postmerge[net_id]))
        postmerge = self.cross_stitch(premerge, self.layer3_unit)

        premerge = []
        for net_id, net in enumerate(self.nets):
            premerge.append(net.avgpool(net.layer4(postmerge[net_id])))
        postmerge = self.cross_stitch(premerge, self.layer4_unit)

        output = []
        for net_id, net in enumerate(self.nets):
            output.append(net.fc(torch.flatten(postmerge[net_id], 1)))
        return output