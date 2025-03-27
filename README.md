<p align="center">
  <img src="https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/docs/images/MONAI-logo-color.png" width="30%"/>
</p>

# MONAI Vision Language Models
The repository provides a collection of vision language models, benchmarks, and related applications, released as part of Project [MONAI](https://monai.io) (Medical Open Network for Artificial Intelligence).

## 💡 News

- [2024/12/04] The arXiv version of VILA-M3 is now available [here](https://arxiv.org/abs/2411.12915).
- [2024/10/31] We released the [VILA-M3-3B](https://huggingface.co/MONAI/Llama3-VILA-M3-3B), [VILA-M3-8B](https://huggingface.co/MONAI/Llama3-VILA-M3-8B), and [VILA-M3-13B](https://huggingface.co/MONAI/Llama3-VILA-M3-13B) checkpoints on [HuggingFace](https://huggingface.co/MONAI).
- [2024/10/24] We presented VILA-M3 and the VLM module in MONAI at MONAI Day ([slides](./m3/docs/materials/VILA-M3_MONAI-Day_2024.pdf), [recording](https://www.youtube.com/watch?v=ApPVTuEtBjc&list=PLtoSVSQ2XzyDOjOn6oDRfEMCD-m-Rm2BJ&index=16))
- [2024/10/24] Interactive [VILA-M3 Demo](https://vila-m3-demo.monai.ngc.nvidia.com/) is available online!

## VILA-M3

**VILA-M3** is a *vision language model* designed specifically for medical applications. 
It focuses on addressing the unique challenges faced by general-purpose vision-language models when applied to the medical domain and integrated with existing expert segmentation and classification models.

<p align="center">
  <img src="m3/docs/images/VILA-M3_overview_v2.png" width="95%"/>
</p>

For details, see [here](m3/README.md).

### Online Demo

Please visit the [VILA-M3 Demo](https://vila-m3-demo.monai.ngc.nvidia.com/) to try out a preview version of the model.

<p align="center">
  <img src="m3/docs/images/gradio_app_ct.png" width="70%"/>
</p>

## Local Demo

### Prerequisites

#### **Recommended: Build Docker Container**
1.  To run the demo, we recommend building a Docker container with all the requirements.
    We use a [base image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda) with cuda preinstalled.
    ```bash
    docker build --network=host --progress=plain -t monai-m3:latest -f m3/demo/Dockerfile .
    ```
2. Run the container
    ```bash
    docker run -it --rm --ipc host --gpus all --net host monai-m3:latest bash
    ```
    > Note: If you want to load your own VILA checkpoint in the demo, you need to mount a folder using `-v <your_ckpts_dir>:/data/checkpoints` in your `docker run` command.
3. Next, follow the steps to start the [Gradio Demo](./README.md#running-the-gradio-demo).

#### Alternative: Manual installation
1. **Linux Operating System**

1. **CUDA Toolkit 12.2** (with `nvcc`) for [VILA](https://github.com/NVlabs/VILA).

    To verify CUDA installation, run:
    ```bash
    nvcc --version
    ```
    If CUDA is not installed, use one of the following methods:
    - **Recommended** Use the Docker image: `nvidia/cuda:12.2.2-devel-ubuntu22.04`
        ```bash
        docker run -it --rm --ipc host --gpus all --net host nvidia/cuda:12.2.2-devel-ubuntu22.04 bash
        ```
    - **Manual Installation (not recommended)** Download the appropiate package from [NVIDIA offical page](https://developer.nvidia.com/cuda-12-2-2-download-archive)

1. **Python 3.10** **Git** **Wget** and **Unzip**:
    
    To install these, run
    ```bash
    sudo apt-get update
    sudo apt-get install -y wget python3.10 python3.10-venv python3.10-dev git unzip
    ```
    NOTE: The commands are tailored for the Docker image `nvidia/cuda:12.2.2-devel-ubuntu22.04`. If using a different setup, adjust the commands accordingly.

1. **GPU Memory**: Ensure that the GPU has sufficient memory to run the models:
    - **VILA-M3**: 8B: ~18GB, 13B: ~30GB
    - **CXR**: This expert dynamically loads various [TorchXRayVision](https://github.com/mlmed/torchxrayvision) models and performs ensemble predictions. The memory requirement is roughly 1.5GB in total.
    - **VISTA3D**: This expert model dynamically loads the [VISTA3D](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/monaitoolkit/models/monai_vista3d) model to segment a 3D-CT volume. The memory requirement is roughly 12GB, and peak memory usage can be higher, depending on the input size of the 3D volume.
    - **BRATS**: (TBD)

1. ***Setup Environment***: Clone the repository, set up the environment, and download the experts' checkpoints:
    ```bash
    git clone https://github.com/Project-MONAI/VLM --recursive
    cd VLM
    python3.10 -m venv .venv
    source .venv/bin/activate
    make demo_m3
    ```

### Running the Gradio Demo

1. Navigate to the demo directory:
    ```bash
    cd m3/demo
    ```

1. Start the Gradio demo:
    > This will automatically download the default VILA-M3 checkpoint from Hugging Face.
    ```bash
    python gradio_m3.py
    ```

1. Alternative: Start the Gradio demo with a local checkpoint, e.g.:
    ```bash
    python gradio_m3.py  \
    --source local \
    --modelpath /data/checkpoints/<8B-checkpoint-name> \
    --convmode llama_3
    ```
> For details, see the available [commmandline arguments](./m3/demo/gradio_m3.py#L855).

### Running RESTful API Demo

1. Navigate to the demo directory:
    ```bash
    cd m3/demo
    ```

1. Create .env file
    ```
    RELATIVE_DIRECTORY = images/input
    LOGFILE = log/main.log
    PORT = 8585
    ```
1. Create image temporary directory and log directory
    ```bash
    mkdir -p images/input
    mkdir log
    ```

1. Start API service
    ```bash
    python server.py
    ```

#### Adding your own expert model
- This is still a work in progress. Please refer to the [README](m3/demo/experts/README.md) for more details.

## Contributing

To lint the code, please install these packages:

```bash
pip install -r requirements-ci.txt
```

Then run the following command:

```bash
isort --check-only --diff .  # using the configuration in pyproject.toml
black . --check  # using the configuration in pyproject.toml
ruff check .  # using the configuration in ruff.toml
```

To auto-format the code, run the following command:

```bash
isort . && black . && ruff format .
```

## References & Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{nath2024vila,
  title={VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledge},
  author={Nath, Vishwesh and Li, Wenqi and Yang, Dong and Myronenko, Andriy and Zheng, Mingxin and Lu, Yao and Liu, Zhijian and Yin, Hongxu and Law, Yee Man and Tang, Yucheng and others},
  journal={arXiv preprint arXiv:2411.12915},
  year={2024}
}
```
