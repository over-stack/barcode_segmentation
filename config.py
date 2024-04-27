import torch


class Settings:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    gpu_available = torch.cuda.is_available()
    accelerator = 'gpu' if gpu_available else 'cpu'
    device = 'cuda' if gpu_available else 'cpu'
    num_classes = 2
    batch_size = 64
    num_workers = 4
    pin_memory = gpu_available
    height, width = 512, 512
    lr = 1e-4
    weight_decay = 1e-5
    num_epochs = 500
    log_every_n_steps = 5
    early_stopping_patience = 20
    seed = 4
    train_steps_per_epoch = 0
    model_name = 'model_v1.0.0'
    model_path = 'saved_model.ckpt'
    dataset_path = 'datasets/barcode_segmentation'
    eps = 1e-9

    delta_var = 0.5
    delta_dist = 1.5
    norm = 2
    output_downscale = 4

