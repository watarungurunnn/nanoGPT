# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-enwik8-char_moe_2M_load_loss_12'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often
#init_from = 'resume'

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'enwik8_char'
wandb_run_name = 'nanoGPT_moe_load_loss_12'

dataset = 'enwik8_char'
batch_size = 16
block_size = 512
gradient_accumulation_steps = 1

# baby GPT model :)
n_layer = 12
n_head = 12
num_experts = 4
use_moe = True
num_experts_per_tok = 2
n_embd = 312
dropout = 0.2
min_lr = 1e-9

# max_iters = 8000000
# lr_decay_iters = 8000000 # make equal to max_iters usually
max_iters = 2000000
lr_decay_iters = 2000000 # make equal to max_iters usually
# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
