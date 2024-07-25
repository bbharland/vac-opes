import numpy as np
import torch
import torch.nn as nn
from .dataset import WeightedTimelaggedDataset, WeightedTrajectoryDataset
from .util import to_torch


def sym_eig(a: torch.Tensor):
    """Solve (regularized) eigenvalue problem for symmetric torch Tensor
    ** TODO: clearer to put transpose in sym_inverse rather than here?
    ** TODO: surely pytorch can infer dtype, device?
    """
    ar = a + 1e-6 * torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    f = torch.linalg.eigh(ar)
    return f.eigenvalues, f.eigenvectors


def sym_inverse(a: torch.Tensor, return_sqrt=False):
    eigvals, eigvecs = sym_eig(a)
    if return_sqrt:
        eigvalm = torch.diag(torch.sqrt(1 / eigvals))
    else:
        eigvalm = torch.diag(1 / eigvals)
    return eigvecs @ eigvalm @ eigvecs.t()


def cov_matrices_weighted(x: torch.Tensor, xweights: torch.Tensor,
                          y: torch.Tensor, yweights: torch.Tensor):
    """Return cov(X, X) and cov(X, Y) where X, Y do not have zero mean.

    Use the biased estimator for simplicity.  The correct unbiased estimator is more involved and described at:

        https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance

    Note: PyTorch broadcasting rules makes keepdim=True unnecessary in .sum(axis=0)
    """
    def mean(w, x):
        return (w.reshape(-1, 1) * x).sum(axis=0) / w.sum()

    def cov_meanfree(w, x, y):
        # return w * x.t() @ y / (w.sum() - 1)
        return w * x.t() @ y / w.sum()

    xmean = mean(xweights, x)
    ymean = mean(yweights, y)
    x = x - xmean
    y = y - ymean

    c00 = cov_meanfree(xweights, x, x)
    c01 = cov_meanfree(xweights, x, y)
    c11 = cov_meanfree(yweights, y, y)

    mean = 0.5 * (xmean + ymean)
    c0 = 0.5 * (c00 + c11)            # average x, y variance
    c1 = 0.5 * (c01 + c01.t())        # add reverse transitions
    return mean, c0, c1


def koopman_matrix_weighted(x: torch.Tensor, xweights: torch.Tensor,
                            y: torch.Tensor, yweights: torch.Tensor):
    """Minibatch estimate of Koopman matrix during training.  The subtraction of the means of x, y vectors means that the eigenfunction corresponding to the equilibrium process has been projected out.
    """
    _, c0, c1 = cov_matrices_weighted(x, xweights, y, yweights)
    inv_sqrt_c0 = sym_inverse(c0, return_sqrt=True)
    return inv_sqrt_c0 @ c1 @ inv_sqrt_c0


def vampnet(p):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        torch.set_num_threads(12)
    print(f'{device = }, {torch.backends.cudnn.enabled = }')
    print(f'{torch.cuda.is_available() = }, {torch.__version__ = }')

    net = nn.Sequential(
        nn.BatchNorm1d(p.num_features),
        nn.Linear(p.num_features, 100), nn.ELU(),
        nn.Linear(100, 100), nn.ELU(),
        nn.Linear(100, 30), nn.ELU(),
        nn.Linear(30, p.num_eigvecs), nn.Tanh()
    )
    return VAMPNet(net, device, p.learning_rate, p.loss_method)


class VAMPNet:
    """Optimize objective function of Koopman matrix, K
    vamp1: loss = -(1 + tr K), where the 1 is the Perron eigenvalue
    vamp2: loss = -(1 + tr KK')

    Pytorch module notes:
    ---------------------
    1. Check device network is on:
        next(net.parameters()).device

    1. Train, evaluation modes
        net.train() -> net.training = True
        net.eval() -> net.training = False
    Evaluation mode: ignores dropouts, batchnorm taken from saved statistics and not computed on the fly.

    2. Disabling autograd: when you don't need/want to track gradients on parameters
        context manager:    with torch.no_grad():
        set to inference:   for p in net.parameters():
                                p.requires_grad = False
    """
    def __init__(self, net, device, learning_rate, loss_method):
        assert loss_method in ('vamp1', 'vamp2')
        self.net = net.to(device=device).float()
        self.device = device
        self.optim = torch.optim.Adam(params=self.net.parameters(),
                                      lr=learning_rate)
        self.loss_method = loss_method

        self._train_scores = []
        self._test_scores = []

    @property
    def train_scores(self):
        return np.array(self._train_scores)

    @property
    def test_scores(self):
        return np.array(self._test_scores)

    def fit(self, data_loader_train, data_loader_test, num_epochs=1, progress=None):
        for epoch in progress(range(num_epochs), desc="VAMPnet epoch",
                              total=num_epochs, leave=False):
            # training
            self.net.train()
            for x, wx, y, wy in data_loader_train:
                self.optim.zero_grad()
                loss = self.loss(self.net(x.to(device=self.device)),
                                 wx.to(device=self.device),
                                 self.net(y.to(device=self.device)),
                                 wy.to(device=self.device))
                loss.backward()
                self.optim.step()
                self._train_scores.append([epoch + 1, (-loss).item()])

            # validation
            self.net.eval()
            for x, wx, y, wy in data_loader_test:
                loss = self.loss(self.net(x.to(device=self.device)),
                                 wx.to(device=self.device),
                                 self.net(y.to(device=self.device)),
                                 wy.to(device=self.device))
                self._test_scores.append([epoch + 1, (-loss).item()])

    def loss(self, x: torch.Tensor, xweights: torch.Tensor,
             y: torch.Tensor, yweights: torch.Tensor):
        koopman = koopman_matrix_weighted(x, xweights, y, yweights)
        if self.loss_method == 'vamp1':
            vamp_score = torch.linalg.norm(koopman, ord='nuc')
        else:
            vamp_score = torch.square(torch.linalg.norm(koopman, ord='fro'))
        return -(1 + vamp_score)


class SRV:
    """Network:
    - references are vampnet.net, self.net, returned reference in srv_net()
    - changed to inference mode in __init__()
    - moved to CPU in fit()

    Eigenfunctions:
    SRV.__call__(x) returns ndarray, shape (num_eigvecs,)
    srv_net(x) returns ndarray, shape (num_cvs,)

    Pytorch module notes:
    ---------------------
    1. Check device network is on:
        next(net.parameters()).device

    1. Train, evaluation modes
        net.train() -> net.training = True
        net.eval() -> net.training = False
    Evaluation mode: ignores dropouts, batchnorm taken from saved statistics and not computed on the fly.

    2. Disabling autograd: when you don't need/want to track gradients on parameters
        context manager:    with torch.no_grad():
        set to inference:   for p in net.parameters():
                                p.requires_grad = False

    TODO, May 26, 2024:
    -------------------
    Fix strange behavior when running cell in test-vampnet-srv.ipynb.  Running the cell a second time causes no error(!?)

            from src.vampnet import SRV

            fit_srv = True
            if fit_srv:
                srv = SRV(vn.net, sd.lagtime)
                srv.fit(dataset)
                save_npz(sd.files['srv'], srv)
                torch.save(srv.srv_net(), sd.files['srv_net'])
            else:
                srv = load_npz(sd.files['srv'])

            sd.save_eigen_data(srv)
            sd.save_dihedral_grid_data()

    RuntimeError => sd.save_eigen_data(srv)
                 => psi = srv(self.features)
                 => z = self.net(to_torch(features, device=self.device)).cpu().numpy()

    RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument weight in method wrapper__native_batch_norm)
    """
    def __init__(self, net, lagtime):
        """Paramters
        ---------
        net : torch.nn.modules.container.Sequential
            From vampnet.net
        lagtime : float
            Lag time (tau) in ps
        """
        from .util import get_module_device

        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

        self.lagtime = lagtime
        self.num_eigvecs = self.net[-2].out_features
        self.device = get_module_device(self.net)
        self.mean = None
        self.transform_matrix = None
        self.eigvals = None

    def timescales(self):
        return -self.lagtime / np.log(self.eigvals)

    def fit(self, dataset):
        from scipy.linalg import inv, sqrtm

        def eigh_sorted(a):
            w, v = np.linalg.eigh(a)
            idx = w.argsort()[::-1]
            return np.real(w[idx]), np.real(v[:, idx])

        def to_numpy(a):
            return a.cpu().numpy().astype('float64')

        # adding WeightedTrajectoryDatasets recasts as WeightedTimelaggedDataset
        assert type(dataset) in (
            WeightedTimelaggedDataset, WeightedTrajectoryDataset
            ), "For now, only do WeightedTimelaggedDataset"

        # transform on 'device' with torch
        x = self.net(to_torch(dataset.x, device=self.device))
        xweights = to_torch(dataset.xweights, device=self.device)
        y = self.net(to_torch(dataset.y, device=self.device))
        yweights = to_torch(dataset.yweights, device=self.device)
        mean, c0, c1 = cov_matrices_weighted(x, xweights, y, yweights)

        # final eigenvalue problem (transform matrix) on CPU with numpy
        self.mean, c0, c1 = to_numpy(mean), to_numpy(c0), to_numpy(c1)
        inv_sqrt_c0 = inv(sqrtm(c0))
        koopman = inv_sqrt_c0 @ c1 @ inv_sqrt_c0
        self.eigvals, eigvecs = eigh_sorted(koopman)
        self.transform_matrix = inv_sqrt_c0 @ eigvecs

    def srv_net(self, num_cvs=2):
        """Return network with transformation added as a linear layer.
        The layers are set to inference mode, and moved to CPU.
        """
        W = to_torch(self.transform_matrix[:, :num_cvs])
        b = -to_torch((self.mean @ self.transform_matrix)[:num_cvs])

        # torch convention for layer: y = xW' + b
        eig_layer = nn.Linear(self.num_eigvecs, num_cvs)
        eig_layer.weight = nn.Parameter(W.t())
        eig_layer.bias = nn.Parameter(b)

        net = nn.Sequential(*self.net, eig_layer)
        for p in net.parameters():
            p.requires_grad = False
        return net.to(device=torch.device('cpu')).float()

    def __call__(self, features):
        z = self.net(to_torch(features, device=self.device)).cpu().numpy()
        return (z - self.mean) @ self.transform_matrix
