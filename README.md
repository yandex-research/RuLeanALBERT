# RuLeanALBERT

RuLeanALBERT is a pretrained masked language model for the Russian language using a memory-efficient architecture.

## Using the model
You can download the pretrained weights, the tokenizer and the config file used for pretraining from the Hugging Face Hub: [huggingface.co/yandex/RuLeanALBERT](https://huggingface.co/yandex/RuLeanALBERT).
Download them directly by running the following code:
```
wget https://huggingface.co/yandex/RuLeanALBERT/resolve/main/state.pth state.pth
wget https://huggingface.co/yandex/RuLeanALBERT/raw/main/config.json config.json
mkdir tokenizer
wget https://huggingface.co/yandex/RuLeanALBERT/raw/main/tokenizer/config.json tokenizer/config.json
wget https://huggingface.co/yandex/RuLeanALBERT/raw/main/tokenizer/special_tokens_map.json tokenizer/special_tokens_map.json
wget https://huggingface.co/yandex/RuLeanALBERT/raw/main/tokenizer/tokenizer.json tokenizer/tokenizer.json
```

As the model itself is using custom code (see [`src/models`](./src/models)), right now the simplest solution is to clone the repository and to use the relevant classes (`LeanAlbertForPreTraining` and its dependencies) in your code.

Loading the model is as simple as
```
tokenizer = AlbertTokenizerFast.from_pretrained('tokenizer')
config = LeanAlbertConfig.from_pretrained('config.json')
model = LeanAlbertForPreTraining(config)
model.resize_token_embeddings(len(tokenizer))
checkpoint = torch.load(checkpoint_path, map_location='cpu')['model']
model.load_state_dict(checkpoint)
```

## Fine-tuning guide

Once you have downloaded the model, you can use [`finetune_mlm.py`](./finetune_mlm.py) to evaluate it on [RussianSuperGLUE](https://russiansuperglue.com/) and [RuCoLA](https://rucola-benchmark.com/).

To do this, you can run the command as follows:
```
python finetune_mlm.py -t TASK_NAME \
 --checkpoint state.pth \
 --data-dir . \
 --batch-size 32 \
 --grad-acc-steps 1 \
 --dropout 0.1 \
 --weight-decay 0.01 \
 --num-seeds 1
```

Most datasets will be loaded from the Hugging Face Hub. However, you need to download [RuCoLA](https://github.com/RussianNLP/RuCoLA) and [RuCoS](https://russiansuperglue.com/tasks/task_info/RuCoS) from their respective sources and place them in `RuCoLA` and `RuCoS` directories respectively.

For reference, finetuning with a batch size of 32 on RuCoLA should take approximately 10 GB of GPU memory. If you exceed the GPU memory limits for a specific task, you can reduce the batch size by reducing `--batch-size` and increasing `--grad-acc-steps` accordingly.

If you want to finetune an existing masked language model from the Hugging Face Hub, you can do it with the same code. The script directly supports fine-tuning RuRoBERTa-large: simply change `--checkpoint` to `--model-name ruroberta-large`.

For LiDiRus, you should use the model trained on the TERRa dataset with `predict_on_lidirus.py`. 
For RWSD, all ML-based solutions perform worse than the most-frequent class baseline: in our experiments, we found that the majority of runs with RuLeanALBERT or RuRoBERTa converge to a similar solution.

## Pretraining a new model

Here you can find the best practices that we learned from running the experiment. These may help you set up your own collaborative experiment.

If your training run is not confidential, feel free to ask for help on the [Hivemind discord server](https://discord.gg/vRNN9ua2).

<details>
  <summary><b>1. Choose and verify your training configuration</b></summary>  
  
  Depending on you use case, you may want to change
   - Dataset and preprocessing ([`data.py`](tasks/mlm/data.py));
   - Tokenizer ([`arguments.py`](arguments.py));
   - Model config

  In particular, you need to specify the datasets in the `make_training_dataset` function from [`data.py`](tasks/mlm/data.py).
  One solution is to use the [`datasets`](https://github.com/huggingface/datasets) library and stream one of the existing large datasets for your target domain. **A working example** of `data.py` can be found in the [NCAI-research/CALM](https://github.com/NCAI-Research/CALM/blob/main/tasks/mlm/data.py) project.
  
  
  When transitioning to a new language or new dataset, it is important to check that the tokenizer/collator works as intended **before** you begin training.
  The best way to do that is to manually look at training minibatches:
  ```python
  from tasks.mlm.data import make_training_dataset
  from tasks.mlm.whole_word_mask import DataCollatorForWholeWordMask
  
  tokenizer = create_tokenizer_here(...)
  dataset = make_training_dataset(tokenizer, max_sequence_length=...)  # see arguments.py
  collator = DataCollatorForWholeWordMask(tokenizer, pad_to_multiple_of=...)  # see arguments.py
  data_loader = torch.utils.data.DataLoader(dataset, collate_fn=collator, batch_size=4)

  # generate a few batches
  rows = []
  with tqdm(enumerate(data_loader)) as progress:
      for i, row in progress:
          rows.append(row)
          if i > 10:
              break
  
  # look into the training data
  row_ix, sample_ix = 0, 1
  sources = [tokenizer.decode([i]) for i in rows[row_ix]['input_ids'][sample_ix].data.numpy()]
  print("MASK RATE:", (rows[row_ix]['input_ids'][sample_ix] == 4).data.numpy().sum() / (rows[row_ix]['input_ids'][sample_ix] != 0).data.numpy().sum())

  for i in range(len(sources)):
      if sources[i] == '[MASK]':
          pass#sources[i] = '[[' + tokenizer.decode(rows[row_ix]['labels'][sample_ix][i].item()) + ']]'

  print(' '.join(sources))
  ```
  
  If you make many changes, it also helps to train a very model using your own device to check if everything works as intended. A good initial configuration is 6 layers, 512 hidden, 2048 intermediate).
  
  If you're training with volunteers, the most convenient way is to set up a Hugging Face organization. For instructions on that, see "make your own" section of https://training-transformers-together.github.io . We use WANDB for tracking logs and training progress: we've set up a [WandB team](https://docs.wandb.ai/ref/app/features/teams) for this experiment. Alternatively, you can use hivemind standalone (and even without internet access) by setting --authorize False and WANDB_DISABLED=true -- or manually removing the corresponding options from the code.
 
</details>

<details>
  <summary> <b>2. Setting up auxiliary peers</b> </summary>

Auxiliary peers are low-end servers without GPU that will keep track of the latest model checkpoint and report metrics and assist in communication.
You will need 1-3 workers that track metrics, upload statistics, etc. These peers do not use GPU.
If you have many participants are behind firewall (in --client_mode), it helps to add more auxiliary servers, as they can serve as relays and help with all-reduce.
  
__Minimum requirements:__ 15+ GB RAM, at least 100Mbit/s download/upload speed, at least one port opened to incoming connections;

__Where to get:__ cloud providers that have cheap ingress/egress pricing. Good examples: [pebblehost](https://pebblehost.com/dedicated/) "Essential-01" and [hetzner](https://www.hetzner.com/cloud) CX41. Path of the true jedi: use your homelab or university server -- but that may require networking experience. AWS/GCP/Azure has similar offers, but they cost more due to [egress pricing](https://cloud.google.com/vpc/network-pricing).


__Setup env:__

```
sudo apt install -y git tmux
curl https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh > Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh -b -p ~/anaconda3
source ~/anaconda3/bin/activate
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

git clone https://github.com/yandex-research/RuLeanALBERT
cd RuLeanALBERT && pip install -q -r requirements.txt &> log

# re-install bitsandbytes for the actual CUDA version
pip uninstall -y bitsandbytes-cuda111
pip install bitsandbytes-cuda113==0.26.0

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```


__Run auxiliary worker:__

1. Open a tmux (or screen) session that will stay up after you logout. (`tmux new` , [about tmux](https://tmuxcheatsheet.com/))
 
2. Measure internet bandwidth and set `$BANDWIDTH` variable
```bash

# You can measure bandwidth automatically:
curl -s https://gist.githubusercontent.com/justheuristic/5467799d8f2ad59b36fa75f642cc9b87/raw/c5a4b9b66987c2115e6c54a07d97e0104dfbcd97/speedtest.py | python -  --json > speedtest.json
export BANDWIDTH=`python -c "import json; speedtest = json.load(open('speedtest.json')); print(int(max(1, min(speedtest['upload'], speedtest['download']) / 1e6)))"`
echo "Internet Bandwidth (Mb/s) = $BANDWIDTH"
  
# If that doesn't work, you can simply `export BANDWIDTH=TODOyour_bandwidth_mbits_here` using the minimum of download and upload speed.
```
  

3. Run the auxiliary peer
```bash
export MY_IP=`curl --ipv4 -s http://whatismyip.akamai.com/`
echo "MY IP (check not empty):" $MY_IP
# If empty, please set this manually: export MY_IP=...
# When training on internal infrastructure, feel free to use internal IP.
# If using IPv6, please replace /ip4/ with /ip6/ in subsequent lines

export PORT=12345   # please choose a port where you can accept incoming tcp connections (or open that port if you're on a cloud)
export LISTEN_ON=/ip4/0.0.0.0/tcp/$PORT
export ANNOUNCE_ON=/ip4/$MY_IP/tcp/$PORT
export WANDB_START_METHOD=thread
export CUDA_VISIBLE_DEVICES=  # do not use GPUs even if they are avilable
  
# organizations
export WANDB_ENTITY=YOUR_USERNAME_HERE
export HF_ORGANIZATION_NAME=YOUR_ORG_HERE

# experiment name
export EXP_NAME=my-exp
export WANDB_PROJECT=$EXP_NAME
export HF_MODEL_NAME=$EXP_NAME

export WANDB_API_KEY=TODO_get_your_wandb_key_here_wandb.ai/authorize
export HF_USER_ACCESS_TOKEN=TODO_create_user_access_token_here_with_WRITE_permissions_https://huggingface.co/settings/token
# note: you can avoid setting the two tokens above: in that case, the script will ask you to login to wandb and huggingface
  
# activate your anaconda environment
source ~/anaconda3/bin/activate


ulimit -n 16384 # this line is important, ignoring it may cause a "Too Many Open Files" error

python run_aux_peer.py --run_id $EXP_NAME --host_maddrs $LISTEN_ON --announce_maddrs $ANNOUNCE_ON --wandb_project $WANDB_PROJECT --store_checkpoints --upload_interval 43200 --repo_url $HF_ORGANIZATION_NAME/$HF_MODEL_NAME --assist_in_averaging --bandwidth $BANDWIDTH
# Optionally, add more peers to the training via `--initial_peers ONE_OR_MORE PEERS_HERE`
```

If everything went right, it will print its address as such:
![image](https://user-images.githubusercontent.com/3491902/146950956-0ea06e77-15b4-423f-aeaa-02eb6aec06db.png)

Please copy this address and use it as ``--initial_peers`` with GPU trainers and other auxiliary peers.
</details>


<details>
  <summary><b>3. Setting up a trainer</b></summary>
Trainers are peers with GPUs (or other compute accelerators) that compute gradients, average them via all-reduce and perform optimizer steps.
There are two broad types of trainers: normal (full) peers and client mode peers. Client peers rely on others to average their gradients, but otherwise behave same as full peers. You can designate your trainer as a client-only using the `--client_mode` flag.
  
__When do I need client mode?__ if a peer is unreliable (e.g. will likely be gone in 1 hour) OR sits behind a firewall that blocks incoming connections OR has very unstable internet connection, it should be a client. For instance, it is recommended to set colab / kaggle peers as clients. In turn, cloud GPUs (even spot instances!) are generally more reliable and should be full peers.

Participating as a client is easy, you can find the code for that in **this colab notebook(TODO)**. Setting up a full peer is more difficult,
### Set up environment:

This part is the same as in auxiliary peer, except we don't need LFS (that was needed to upload checkpoints).
```bash
sudo apt install -y git tmux
curl https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh > Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh -b -p ~/anaconda3
source ~/anaconda3/bin/activate
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

git clone https://github.com/yandex-research/RuLeanALBERT
cd RuLeanALBERT && pip install -q -r requirements.txt &> log

# re-install bitsandbytes for the actual CUDA version
pip uninstall -y bitsandbytes-cuda111
pip install -y bitsandbytes-cuda113==0.26.0
  
# note: we use bitsandbytes for 8-bit LAMB, and in turn, bitsandbytes needs cuda -- even if you run on a non-CUDA device.
```

```bash
export MY_IP=`curl --ipv4 -s http://whatismyip.akamai.com/`
echo "MY IP (check not empty):" $MY_IP
# If empty, please set this manually: export MY_IP=...
# When training on internal infrastructure, feel free to use internal IP.
# If using IPv6, please replace /ip4/ with /ip6/ in subsequent lines

 export PORT=31337  # same requirements as for aux peer
export LISTEN_ON=/ip4/0.0.0.0/tcp/$PORT

export CUDA_VISIBLE_DEVICES=0  # supports multiple cuda devices!

# organization & experiment name
export WANDB_ENTITY=YOUR_USERNAME_HERE
export HF_ORGANIZATION_NAME=YOUR_ORG_HERE
export EXP_NAME=my-exp
export WANDB_PROJECT=$EXP_NAME-hivemind-trainers
export HF_MODEL_NAME=$EXP_NAME

export WANDB_API_KEY=get_your_wandb_key_here_https://wandb.ai/authorize_OR_just_login_on_wandb
export HF_USER_ACCESS_TOKEN=create_user_access_token_here_with_WRITE_permissions_https://huggingface.co/settings/token
# note: you can avoid setting the two tokens above: in that case, the script will ask you to login to wandb and huggingface

export INITIAL_PEERS="/ip4/IP_ADDR/tcp/12345/p2p/PEER_ID"
# ^-- If you're runnnng an independent experiment, this must be your own initial peers. Can be either auxiliary peers or full gpu peers.


curl -s https://raw.githubusercontent.com/sivel/speedtest-cli/master/speedtest.py | python -  --json > speedtest.json
export BANDWIDTH=`python -c "import json; speedtest = json.load(open('speedtest.json')); print(int(max(1, min(speedtest['upload'], speedtest['download']) / 1e6)))"`
echo "Internet Bandwidth (Mb/s) = $BANDWIDTH"

ulimit -n 16384 # this line is important, ignoring it may cause a "Too Many Open Files" error

python run_trainer.py --run_id $EXP_NAME --host_maddrs $LISTEN_ON --announce_maddrs $ANNOUNCE_ON --initial_peers $INITIAL_PEERS --bandwidth $BANDWIDTH \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1
# you can tune per_device_train_batch_size, gradient_accumulation steps, --fp16, --gradient_checkpoints based on the device. A good rule of thumb is that the device should compute (batch size x num accumulations) gradients over 1-10 seconds. Setting very large gradient_accumulation_steps can cause your peer to miss an averaging round.

```
  
  
</details>

<details>
  <summary><b>Best (and worst) practices</b></summary>
    
  - __Hardware requirements:__ The code is meant to run with the following specs: 2-core CPU, 12gb RAM (more if you train a bigger model). Peers used as `--initial_peers` must be accessible by others, so you may need to open a network port for incoming connections. The rest depends on what role you're playing:

      - __Auxiliary peers:__  If you use `--upload_interval X --repo_url Y` must have enough disk space to store all the checkpoints. For instance, assuming that training takes 1 month and the model+optimizer state takes 1GB, you will need 30GB with `--upload_interval 86400`, 60GB if `--upload_interval 28800`, etc. If `assist_in_averaging`, ensure you have at least 100Mbit/s bandwidth, more is better.
    
      - __Trainers__ need *some* means for compute: a GPU with at least 6GB memory or a TPU - as long as you can run pytorch on that. You will need to tune `--per_device_train_batch_size X` to fit into memory. Also, you can use `--fp16` even on old GPUs to save memory. Finally, `--gradient_checkpointing` can reduce memory usage at the cost of 30-40% slower training. Non-client-mode peers must have at least 100Mbit/s network bandwidth, mode is better.
    
      
  - __Swarm composition:__ you will need 2-3 peers with public IP as `--initial_peers` for redundancy. If some participants are behind firewalls, we recommend finding at least one non-firewalled participant per 5 peers behind firewall.
  
</details>

## Acknowledgements

Many of the best practices from this guide were found in the [CALM](https://github.com/NCAI-Research/CALM) project for collaborative training by NCAI.
