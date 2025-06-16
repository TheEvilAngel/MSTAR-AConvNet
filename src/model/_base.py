import torch
import pdb
import model.network
import time


class Model(object):
    def __init__(self, **params):
        print("Starting Model initialization...")
        total_start = time.time()

        print("Setting up device...")
        t0 = time.time()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device setup took {time.time() - t0:.2f}s")

        print("Creating network...")
        t0 = time.time()
        self.net = model.network.Network(
            classes=params.get('classes', 10),
            channels=params.get('channels', 1),
            dropout_rate=params.get('dropout_rate', 0.5),
            cfm_input_dim=params.get('cfm_input_dim', 100),
            use_cfm=params.get('use_cfm', True),
            cfm_type=params.get('cfm_type', 'hrrp_linear')
        )
        print(f"Network creation took {time.time() - t0:.2f}s")

        print("Moving network to device...")
        t0 = time.time()
        self.net.to(self.device)
        print(f"Moving to device took {time.time() - t0:.2f}s")

        print("Setting up training parameters...")
        t0 = time.time()
        self.lr = params.get('lr', 1e-3)
        self.lr_step = params.get('lr_step', [50])
        self.lr_decay = params.get('lr_decay', 0.1)
        self.lr_scheduler = None
        self.momentum = params.get('momentum', 0.9)
        self.weight_decay = params.get('weight_decay', 4e-3)
        self.criterion = torch.nn.CrossEntropyLoss()
        print(f"Parameter setup took {time.time() - t0:.2f}s")

        print("Creating optimizer...")
        t0 = time.time()
        # 使用一个优化器优化所有参数
        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        print(f"Optimizer creation took {time.time() - t0:.2f}s")

        print("Setting up learning rate scheduler...")
        t0 = time.time()
        if self.lr_decay:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=self.lr_step,
                gamma=self.lr_decay
            )
        print(f"Scheduler setup took {time.time() - t0:.2f}s")
        
        print(f"Total initialization took {time.time() - total_start:.2f}s")

    def optimize(self, x, y, cfm_input=None):
        if self.net.use_cfm:
            if cfm_input is None:
                pdb.set_trace()
                cfm_input = torch.randn(1, self.net.cfm_input_dim, device=self.device)
            p = self.net(x.to(self.device), cfm_input.to(self.device))
        else:
            p = self.net(x.to(self.device))
            
        loss = self.criterion(p, y.to(self.device))

        # 清零所有梯度
        self.optimizer.zero_grad()
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def inference(self, x, cfm_input=None):
        if self.net.use_cfm:
            if cfm_input is None:
                cfm_input = torch.randn(1, self.net.cfm_input_dim, device=self.device)
            return self.net(x.to(self.device), cfm_input.to(self.device))
        return self.net(x.to(self.device))

    def save(self, path):
        save_dict = {
            'network_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(save_dict, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.net.eval()
