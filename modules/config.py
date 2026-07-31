from yacs.config import CfgNode as CN

_C = CN()

_C.num_modes = 32
_C.hidden_size = 768
_C.hidden_dropout_prob = 0.1
_C.num_mask_decoder_layers = 2
_C.num_mode_encoder_layers = 6
_C.loss = CN()
_C.loss.commitment_cost = 0.25