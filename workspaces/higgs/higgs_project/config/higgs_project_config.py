def set_config(c):
    c.input_path = "/gluster/home/ofrebato/baler/workspaces/higgs/data/13TeV_combined_cut_numpy.npz"
    c.compression_ratio = 10
    # c.number_of_columns = 24
    # c.latent_space_size = 15
    c.epochs = 100
    c.early_stopping = False
    c.early_stopping_patience = 100
    c.min_delta = 0
    c.lr_scheduler = True
    c.lr_scheduler_patience = 50
    c.model_name = "AE"
    c.model_type = "dense"
    c.custom_norm = True
    c.l1 = True
    c.reg_param = 0.001
    c.RHO = 0.05
    c.lr = 0.001
    c.batch_size = 6000
    c.test_size = 0
    c.data_dimension = 1
    c.apply_normalization = False
    c.extra_compression = False
    c.intermittent_model_saving = False
    c.intermittent_saving_patience = 100
    c.activation_extraction = False
    c.deterministic_algorithm = False
    c.compress_to_latent_space = False
    c.save_error_bounded_deltas = False
    c.error_bounded_requirement = 1
    c.convert_to_blocks = False
    c.separate_model_saving = False
