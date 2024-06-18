# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-enwik8-char_small'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'enwik8_char_small'
wandb_run_name = 'mini-gpt'

dataset = 'enwik8_char'
batch_size = 16
block_size = 512
gradient_accumulation_steps = 1

# baby GPT model :)
n_layer = 4
n_head = 4
n_embd = 384
dropout = 0.2

max_iters = 8000000
lr_decay_iters = 8000000 # make equal to max_iters usually
# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
