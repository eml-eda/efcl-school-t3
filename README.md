# EFCL Summer School 2024 - Track 3 - Hands On

This repository contains the material for the Track 3 Hands-on Sessions.

It mainly consists of some Jupyter notebooks, plus some support scripts. The notebooks should be followed in this order:

**Hands-On #1**: `hands_on_1.ipynb`

**Hands-On #2**: `hands_on_2.ipynb`

**Hands-On #3**: `hands_on_3.ipynb`

**Hands-On #4**: `hands_on_4.ipynb`

**Hands-On #5**: `hands_on_5.ipynb`

## Prerequisites and General information

### Download the dataset

The first step to setup the hands-on is to download our pre-collected dataset from [this](https://www.dropbox.com/scl/fi/yucxd67gvdkb5vl7tw0th/dataset.zip?rlkey=gf7gje37fhet26z5t6hq0tv0z&st=8rbzte7m&dl=0) link. When requested, use the password that we will provide you during the labs.

Save the dataset under `<GIT_FOLDER>/dataset`.

### Start conda environment

To activate the pre-installed conda environment, run:
```
source /usr/itetnas04/data-scratch-01/$USER/data/conda/bin/activate
```

If there is no conda environment available, you can create your own environment:

```
./install_conda.sh
# log out and log in again
source /usr/itetnas04/data-scratch-01/$USER/data/conda/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu118 -r requirements_torch.txt
pip install -r requirements.txt
```

### Start jupyter lab

For these sessions, you are suggested to use Jupyter Lab (*). Start it with the following command (from the cloned folder):
```
python3 -m jupyterlab --no-browser --port 5901 --ip $(hostname -f)
```
or
```
jupyter lab --no-browser --port 5901 --ip $(hostname -f)
```
The port range should be [5900-5999].

(*) Note: you can also open the notebooks with the classic `jupyter notebook` command if you want, but `jupyter lab` makes it easier to navigate the sections of each notebook.


## Hands-on #1

You will run Hands-on #1 on a GPU cluster, rather than using your local machine, in order to accelerate the training processes. An interactive session lasting for 300 minutes (NAS can be demanding) on a GPU node can be started with:
```
export SLURM_CONF=/home/sladmsnow/slurm/slurm.conf
srun --time 300 --gres=gpu:1 --pty bash -i
```

You can now reactivate your conda environment and start your jupyter lab session on the GPU.

**IMPORTANT**: for sake of time, each group will generate a single Optimized DNN (rather than a full Pareto front) during the hands-on session. The instructors will tell you what value to try. Once done, you will have to upload the optimized **and finetuned** model to [this](https://www.dropbox.com/request/IRUUGGAlAZ4ShAWPr8MF) link, specifying your group name (pick the name that you want, just no vulgarity please &#128515;). 

We will then pick one lucky winner DNN, which *all groups* will use for the personalized fine-tuning in Hands-on #3. Once selected, we will put the winner in [this](https://www.dropbox.com/scl/fo/17nahcdckiig6b5wagk7d/ANIurZ2izCXPrXjapr_l4s8?rlkey=fc9erl3lyq509puspy35ikc3h&st=ge5f8ywd&dl=0) folder for you to download.


## Hands-on #2

You will run Hands-on #2 locally, using the BioGAP for data acquisition and a terminal to run the associated scripts. Make sure to activate the conda environment in the terminal, as described above. The instructions are available in the associated notebook.

## Hands-on #3

Connect to the GPU as you did in Hands-on #1, activate the conda environment, and start your jupyter lab session.

## Hands-on #4

### Initialize Apptainer

The requirements for Hands-on #4 are more complex, as you are expected to cross-compile an application for a RISC-V architecture. The SDK you will use has been developed for Ubuntu 22.04, but you are currently using a Debian 10 machine. To make this possible, we will use an [Apptainer](https://computing.ee.ethz.ch/Services/Apptainer) container. The Apptainer is already located in `/scratch/$USER`. To active it, you can run:
```
cd /scratch/$USER/
apptainer shell ubuntu
source /home/efcl_venv/bin/activate
```

### Prepare deployment environment

You can now use the apptainer as you would use your host. For Hands-on #4, there are several steps you need to take in order to prepare the deployment. First, we copy on '/scratch' all directories concerning the deployment:

```
cp -r /home/gap_sdk_private/ /scratch/$USER/
cp -r /home/match/ /scratch/$USER/
cp -r /home/match_gap9/ /scratch/$USER/

cd /home/$USER/<GIT_FOLDER>
```

You can now start the jupyer lab session in the apptainer. Make sure you are using the correct kernel by running:

```
python3 -m ipykernel install --user --name=efcl_venv
```

and selecting `efcl_venv` when starting your notebook. To do so, open the "Kernel" drop-down menu on the top right of the window and under "Change kernel", select the previously created `efcl_venv` kernel. You can also connect the GAP9 evaluation kit, containing the GAPmod, to your computer - use the switch to turn it on and make sure the LEDs turn on.

## Hands-on #5

You will run Hands-on #2 locally, using the BioGAP for data acquisition, a terminal to run the acquisition script, and another terminal to compute and display the predicted gestures. Make sure to activate the conda environment in the terminals, as described above. The instructions are available in the associated notebook.




