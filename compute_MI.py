import torch
from functions.dim_losses import donsker_varadhan_loss, infonce_loss, fenchel_dual_loss


def compute_loss(args, former_input, latter_input, encoder, dim_local, dim_global, v_out=False, with_latent=False,
                 fake_relu=False, no_relu=False):

    if no_relu and (not with_latent):
        print("WARNING: 'no_relu' has no visible effect if 'with_latent is False.")
    if no_relu and fake_relu:
        raise ValueError("Options 'no_relu' and 'fake_relu' are exclusive")

    output = compute_out(args=args, former_input=former_input, latter_input=latter_input, encoder=encoder,
                         dim_local=dim_local, dim_global=dim_global, v_out=v_out)

    return output


def compute_out(args, former_input, latter_input, encoder, dim_local, dim_global, v_out):
    '''
            Compute dim loss or classificaiton loss
            :param former_input: x
            :param latter_input: x' for h(x')
            :param loss_type : 'dim' (mi estimation) or 'cla' (classification)
            :param detach:
            :param enc_in_eval:
            :return:
            '''

    rep_out = encoder(latter_input, args.is_internal, args.is_internal_last)

    out_local, out_global = extract(former_input, rep_out, dim_local, dim_global)

    va_fd_measure = args.va_fd_measure
    va_mode = args.va_mode
    loss_encoder_dim = cal_dim(out_local, out_global, va_fd_measure, va_mode, scale=1.0, v_out=v_out)

    return loss_encoder_dim


def sample_locations(enc, n_samples):
    '''Randomly samples locations from localized features.

    Used for saving memory.

    Args:
        enc: Features.
        n_samples: Number of samples to draw.

    Returns:
        torch.Tensor

    '''
    n_locs = enc.size(2)
    batch_size = enc.size(0)
    weights = torch.tensor([1. / n_locs] * n_locs, dtype=torch.float)
    idx = torch.multinomial(weights, n_samples * batch_size, replacement=True) \
        .view(batch_size, n_samples)
    enc = enc.transpose(1, 2)
    adx = torch.arange(0, batch_size).long()
    enc = enc[adx[:, None], idx].transpose(1, 2)

    return enc


def extract(input, outs, local_net=None, global_net=None, local_samples=None,
            global_samples=None):
    '''Wrapper function to be put in encoder forward for speed.

    Args:
        outs (list): List of activations
        local_net (nn.Module): Network to encode local activations.
        global_net (nn.Module): Network to encode global activations.

    Returns:
        tuple: local, global outputs

    '''
    L = input
    G = outs
    # All globals are reshaped as 1x1 feature maps.
    global_size = G.size()[1:]
    if len(global_size) == 1:
        G = G[:, :, None, None]
    L = L.detach()
    L = local_net(L)
    G = global_net(G)

    N, local_units = L.size()[:2]
    L = L.view(N, local_units, -1)
    G = G.view(N, local_units, -1)

    # Sample locations for saving memory.
    if global_samples is not None:
        G = sample_locations(G, global_samples)

    if local_samples is not None:
        L = sample_locations(L, local_samples)

    return L, G


def cal_dim(L, G, measure='JSD', mode='fd', scale=1.0, act_penalty=0., v_out=False):
    '''

    Args:
        measure: Type of f-divergence. For use with mode `fd`.
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
        scale: Hyperparameter for local DIM. Called `beta` in the paper.
        act_penalty: L2 penalty on the global activations. Can improve stability.

    '''

    if mode == 'fd':
        loss = fenchel_dual_loss(L, G, measure=measure)
    elif mode == 'nce':
        loss = infonce_loss(L, G)
    elif mode == 'dv':
        loss = donsker_varadhan_loss(L, G, v_out)
    else:
        raise NotImplementedError(mode)

    if act_penalty > 0.:
        act_loss = act_penalty * (G ** 2).sum(1).mean()
    else:
        act_loss = 0.

    loss_encoder = scale * loss + act_loss

    return loss_encoder
