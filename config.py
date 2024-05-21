import torch


class Settings:
    gpu_available = torch.cuda.is_available()
    accelerator = 'gpu' if gpu_available else 'cpu'
    device = 'cuda' if gpu_available else 'cpu'
    pin_memory = gpu_available
    log_every_n_steps = 5
    eps = 1e-9

    model_name = 'model_v1.0.0'
    model_path = 'saved_model.ckpt'
    # dataset_path = 'datasets/muenster_barcode_db'
    dataset_path = 'datasets/ZVZ-real-512'

    mean = [0.64195696, 0.62124015, 0.59351514]
    std = [0.30759264, 0.30921391, 0.32350163]

    height, width = 512, 512
    output_downscale = 4
    num_workers = 4

    in_channels = 3  # without background   (18)
    num_classes = 1
    embedding_dims = 2
    num_filters = 24

    num_epochs = 500
    batch_size = 64

    lr = 1e-2
    # add different learning rate for other loss function
    weight_decay = 1e-7
    lr_reduce_factor = 0.5
    lr_reduce_patience = 3
    early_stopping_patience = 10
    seed = 4

    min_object_area = 30

    # classification-detection loss
    weight_positive = 15
    weight_negative = 1
    weight_k_worst_negative = 5
    detection_loss_weight = 1
    classification_loss_weight = 1
    postprocessing_threshold = 0.5

    # discriminative loss
    delta_var = 0.5
    delta_dist = 3
    var_term_weight = 1
    dist_term_weight = 2
    reg_term_weight = 0.001
    norm = 2

    # total loss
    base_loss_weight = 1
    embedding_loss_weight = 0.3

    classes = {'EAN13': 1, 'PDF417': 2, 'DataMatrix': 3, 'QRCode': 4, 'RoyalMailCode': 5, 'Kix': 6,
               'Code128': 7, 'UPCA': 8, 'Aztec': 9, 'Interleaved25': 10, 'JapanPost': 11, 'Code39': 12,
               'Postnet': 13, 'UCC128': 14, 'IntelligentMail': 15, '2-Digit': 16, 'EAN8': 17, 'IATA25': 18}

    classes_reverse = {1: 'EAN13', 2: 'PDF417', 3: 'DataMatrix', 4: 'QRCode', 5: 'RoyalMailCode', 6: 'Kix',
               7: 'Code128', 8: 'UPCA', 9: 'Aztec', 10: 'Interleaved25', 11: 'JapanPost', 12: 'Code39',
               13: 'Postnet', 14: 'UCC128', 15: 'IntelligentMail', 16: '2-Digit', 17: 'EAN8', 18: 'IATA25'}

    class_weights = torch.tensor([
        0.21915584415584416, 0.3479381443298969, 0.7336956521739131, 0.45918367346938777, 0.3515625, 8.4375,
        0.6683168316831684, 6.75, 1.6071428571428572, 3.0681818181818183, 2.7, 1.35, 9.642857142857142,
        9.642857142857142, 33.75, 22.5, 33.75, 22.5
    ])

    class_colors = list([torch.tensor(color) for color in [
        (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (155, 0, 155), (155, 155, 0), (0, 155, 155), (155, 200, 155), (200, 155, 155), (155, 155, 200),
        (200, 200, 155), (155, 200, 200), (200, 155, 200), (155, 0, 200), (200, 0, 155), (0, 155, 200),
        (200, 155, 0), (155, 200, 0)
    ]])
