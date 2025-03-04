from einops import repeat
import torch
from dynamics.dynamic_sit import create_transport
from dynamics.dynamic_sit.transport import Sampler
from model_dit import DiT
from utils import uvit_sde


def test_sit():
    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
    )
    img_dim = 32
    in_channels = 3
    model = DiT(img_dim=img_dim, in_channels=in_channels, num_classes=-1, depth=4)
    _param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Param count: {_param_count / 1e6}M")

    x = torch.rand(10, in_channels, img_dim, img_dim).to("cuda")
    t = torch.rand(10).to("cuda")
    z = torch.rand_like(x)
    training_losses_fn = transport.training_losses
    transport_sampler = Sampler(transport)
    sample_fn = transport_sampler.sample_ode(num_steps=2)  # default to ode sampling

    _loss = training_losses_fn(model, x, {})
    print(_loss)

    x_sampled = sample_fn(z, model)[-1]
    print(x_sampled.shape)


def test_uvit_dynamic(schedule="linear", pred="noise_pred", num_steps=2):
    from utils.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

    noise_schedule = NoiseScheduleVP(schedule=schedule)

    def training_losses_fn(model, x, model_kwargs=None):
        sm = uvit_sde.ScoreModel(nnet=model, pred=pred, sde=uvit_sde.VPSDE())
        loss = uvit_sde.LSimple(sm, x, pred=pred, **model_kwargs)
        d = dict(loss=loss)
        return d

    loss = training_losses_fn(model, x)
    print(loss)

    ############################
    def sample_fn(z, model, model_kwargs=None):
        model_fn = model_wrapper(
            model, noise_schedule, time_input_type="0", model_kwargs=model_kwargs
        )
        dpm_solver = DPM_Solver(model_fn, noise_schedule)
        x_sampled = dpm_solver.sample(
            z,
            steps=num_steps,
            eps=1e-4,
            adaptive_step_size=False,
            fast_version=True,
        )
        x_sampled = repeat(x_sampled, "b c h w -> ss b c h w", ss=7)
        return x_sampled

    x_sampled = sample_fn(z, model)[-1]
    print(x_sampled.shape)


def test_discretefm_1d(
    mask_token_id=0, vocab_size=248, device="cuda", input_tensor_type="bt", n_steps=100
):

    from dynamics.dynamic_discretefm import (
        DiscreteFM,
        Ucoupling,
        CubicScheduler,
        SimpleSampler,
    )

    img_dim = 32
    vocab_size = 3
    from model_discrete_dit1d_flex import DiT

    model = DiT(
        num_tokens=img_dim * img_dim, vocab_size=vocab_size, num_classes=-1, depth=4
    ).to(device)
    _param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Param count: {_param_count / 1e6}M")

    t = torch.rand(10).to("cuda")

    coupling = Ucoupling(mask_token_id=mask_token_id)
    kappa = CubicScheduler()
    discretefm = DiscreteFM(
        vocab_size=vocab_size,
        coupling=coupling,
        kappa=kappa,
        device=device,
        input_tensor_type=input_tensor_type,
    )
    training_losses_fn = discretefm.training_losses

    sampler = SimpleSampler(
        mask_token_id=mask_token_id, input_tensor_type=input_tensor_type
    )  # or CorrectorSampler()

    def sample_fn(sample_size, model, **model_kwargs):
        r = sampler(sample_size, discretefm, model=model, n_steps=n_steps)
        r = torch.stack(r, dim=0).to(device)
        return r

    x = torch.randint(0, vocab_size, (4, img_dim * img_dim)).to(device)

    _loss = training_losses_fn(model, x, {})
    print(_loss)

    x_sampled = sample_fn((4, img_dim * img_dim), model)[-1]
    print(x_sampled.shape)


def test_discretefm_2d(
    mask_token_id=0,
    img_dim=32,
    vocab_size=2048,
    device="cuda",
    input_tensor_type="bwh",
    n_steps=100,
):
    from dynamics.dynamic_discretefm import (
        DiscreteFM,
        Ucoupling,
        CubicScheduler,
        SimpleSampler,
    )

    from model_discrete_ditllama2d import DiT

    model = DiT(
        patch_size=2,
        embed_dim=768,
        depth=12,
        num_heads=12,
        in_channels=1,
        vocab_size=vocab_size,
    ).to(device)
    _param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Param count: {_param_count / 1e6}M")

    t = torch.rand(10).to("cuda")

    coupling = Ucoupling(mask_token_id=mask_token_id)
    kappa = CubicScheduler()
    discretefm = DiscreteFM(
        vocab_size=vocab_size,
        coupling=coupling,
        kappa=kappa,
        device=device,
        input_tensor_type=input_tensor_type,
    )
    training_losses_fn = discretefm.training_losses

    sampler = SimpleSampler(
        mask_token_id=mask_token_id, input_tensor_type=input_tensor_type
    )  # or CorrectorSampler()

    def sample_fn(sample_size, model, **model_kwargs):
        r = sampler(sample_size, discretefm, model=model, n_steps=n_steps)
        r = torch.stack(r, dim=0).to(device)
        return r

    sample_shape = (4, img_dim, img_dim)

    x = torch.randint(0, vocab_size, sample_shape).to(device)

    _loss = training_losses_fn(model, x, {})
    print(_loss)

    x_sampled = sample_fn(sample_size=sample_shape, model=model)[-1]
    print(x_sampled.shape)


if __name__ == "__main__":
    # test_sit()
    # test_uvit_dynamic()
    test_discretefm_2d()
    # test_discretefm_1d()
