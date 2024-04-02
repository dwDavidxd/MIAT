from utils_attack.wasserstein_attack import WassersteinAttack
from utils_attack.projection import dual_projection, dual_capacity_constrained_projection

import torch


def str2bool(x):
    return bool(int(x))


def str_or_none(x):
    return None if x == "None" else str(x)


def int_or_none(x):
    return None if x == "None" else int(x)


def float_or_none(x):
    return None if x == "None" else float(x)

def tensor_norm(tensor, p=2):
    """
    Return the norm for a batch of samples
    Args:
        tensor: tensor of size (batch, channel, img_size, last_dim)
        p: 1, 2 or inf

        if p is inf, the size of tensor can also be (batch, channel, img_size)
    Return:
        tensor of size (batch, )
    """
    assert tensor.layout == torch.strided

    if p == 1:
        return tensor.abs().sum(dim=(1, 2, 3))
    elif p == 2:
        return torch.sqrt(torch.sum(tensor * tensor, dim=(1, 2, 3)))
    elif p == 'inf':
        return tensor.abs().view(tensor.size(0), -1).max(dim=-1)[0]
    else:
        assert 0


def check_hypercube(adv_example, tol=None, verbose=True):
    if verbose:
        print("----------------------------------------------")
        print("num of pixels that exceed exceed one {:d}  ".format((adv_example > 1.).sum(dim=(1, 2, 3)).max().item()))
        print("maximum pixel value                  {:f}".format(adv_example.max().item()))
        print("total pixel value that exceed one    {:f}".format((adv_example - 1.).clamp(min=0.).sum(dim=(1, 2, 3)).max().item()))
        print("% of pixel value that exceed one     {:f}%".format(
            100 * ((adv_example - 1.).clamp(min=0.).sum(dim=(1, 2, 3)) / adv_example.sum(dim=(1, 2, 3))).max().item()))
        print("----------------------------------------------")

    if tol is not None:
        assert(((adv_example - 1.).clamp(min=0.).sum(dim=(1, 2, 3)) / adv_example.sum(dim=(1, 2, 3))).max().item() < tol)


class LinfFWA(WassersteinAttack):

    def __init__(self, predict, loss_fn, eps, kernel_size, lr, nb_iter, dual_max_iter, grad_tol, int_tol, device="cuda",
                 postprocess=False, verbose=True,):
        super().__init__(predict=predict, loss_fn=loss_fn, eps=eps, kernel_size=kernel_size, device=device,
                         postprocess=postprocess, verbose=verbose,)

        self.lr = lr
        self.nb_iter = nb_iter
        self.dual_max_iter = dual_max_iter
        self.grad_tol = grad_tol
        self.int_tol = int_tol

        self.inf = 1000000

        # self.capacity_proj_mod = capacity_proj_mod

    def perturb(self, X, y):
        batch_size, c, h, w = X.size()

        self.initialize_cost(X, inf=self.inf)
        pi = self.initialize_coupling(X).clone().detach().requires_grad_(True)
        normalization = X.sum(dim=(1, 2, 3), keepdim=True)

        for t in range(self.nb_iter):
            adv_example = self.coupling2adversarial(pi, X)
            scores = self.predict(adv_example.clamp(min=self.clip_min, max=self.clip_max))

            loss = self.loss_fn(scores, y)
            loss.backward()

            with torch.no_grad():
                self.lst_loss.append(loss.item())
                self.lst_acc.append((scores.max(dim=1)[1] == y).sum().item())

                """Add a small constant to enhance numerical stability"""
                # print(tensor_norm(pi.grad, p='inf').min())
                pi.grad /= (tensor_norm(pi.grad, p='inf').view(batch_size, 1, 1, 1) + 1e-35)
                assert (pi.grad == pi.grad).all() and (pi.grad != float('inf')).all() and (pi.grad != float('-inf')).all()

                pi += self.lr * pi.grad

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()

                # if self.capacity_proj_mod == -1:
                pi, num_iter = dual_projection(pi,
                                               X,
                                               cost=self.cost,
                                               eps=self.eps * normalization.squeeze(),
                                               dual_max_iter=self.dual_max_iter,
                                               grad_tol=self.grad_tol,
                                               int_tol=self.int_tol)

                # elif (t + 1) % self.capacity_proj_mod == 0:
                #     pi = dual_capacity_constrained_projection(pi,
                #                                               X,
                #                                               self.cost,
                #                                               eps=self.eps * normalization.squeeze(),
                #                                               transpose_idx=self.forward_idx,
                #                                               detranspose_idx=self.backward_idx,
                #                                               coupling2adversarial=self.coupling2adversarial)
                #     num_iter = 3000

                end.record()

                torch.cuda.synchronize()

                self.run_time += start.elapsed_time(end)
                self.num_iter += num_iter
                self.func_calls += 1

                if self.verbose and (t + 1) % 10 == 0:
                    print("num of iters : {:4d}, ".format(t + 1),
                          "loss : {:9.3f}, ".format(loss.item()),
                          "acc : {:5.2f}%, ".format((scores.max(dim=1)[1] == y).sum().item() / batch_size * 100),
                          "dual iter : {:4d}, ".format(num_iter),
                          "per iter time : {:7.3f}ms".format(start.elapsed_time(end) / num_iter))

                self.check_nonnegativity(pi / normalization, tol=1e-6, verbose=False)
                self.check_marginal_constraint(pi / normalization, X / normalization, tol=1e-6, verbose=False)
                self.check_transport_cost(pi / normalization, tol=1e-3, verbose=False)

            pi = pi.clone().detach().requires_grad_(True)

        with torch.no_grad():
            adv_example = self.coupling2adversarial(pi, X)
            check_hypercube(adv_example, verbose=self.verbose)
            self.check_nonnegativity(pi / normalization, tol=1e-5, verbose=self.verbose)
            self.check_marginal_constraint(pi / normalization, X / normalization, tol=1e-5, verbose=self.verbose)
            self.check_transport_cost(pi / normalization, tol=self.eps * 1e-3, verbose=self.verbose)

            if self.postprocess is True:
                if self.verbose:
                    print("==========> post-processing projection........")

                pi = dual_capacity_constrained_projection(pi,
                                                          X,
                                                          self.cost,
                                                          eps=self.eps * normalization.squeeze(),
                                                          transpose_idx=self.forward_idx,
                                                          detranspose_idx=self.backward_idx,
                                                          coupling2adversarial=self.coupling2adversarial)

                adv_example = self.coupling2adversarial(pi, X)
                check_hypercube(adv_example, tol=self.eps * 1e-1, verbose=self.verbose)
                self.check_nonnegativity(pi / normalization, tol=1e-6, verbose=self.verbose)
                self.check_marginal_constraint(pi / normalization, X / normalization, tol=1e-6, verbose=self.verbose)
                self.check_transport_cost(pi / normalization, tol=self.eps * 1e-3, verbose=self.verbose)

        """Do not clip the adversarial examples to preserve pixel mass"""
        return adv_example