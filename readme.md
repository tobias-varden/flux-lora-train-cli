# Flux LORA Training CLI

This is a command-line interface (CLI) tool for creating a Flux LORA (LoRA: Low-Rank Adaptation) using [replicate.com](https://replicate.com). Flux LORA is a technique for fine-tuning a pre-trained model to learn a new concept, in this case, a specific token.

## Prerequisites

- Python 3.6 or higher
- A [replicate.com](https://replicate.com) account and API token
- A GitHub account

## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/flux-lora-train-cli.git
cd flux-lora-train-cli
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Set your replicate.com API token as an environment variable:

```bash
export REPLICATE_API_TOKEN=your-api-token
```

## Usage

```bash
python main.py owner model_name image_path token
```

- `owner`: Your GitHub username.
- `model_name`: The name for your LORA model.
- `image_path`: The path to the zip file containing training images.
- `token`: The token for the concept to be learned.

## Example

```bash
python main.py my-github-username my-lora-model path/to/images.zip my-token
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [replicate.com](https://replicate.com) for providing the infrastructure for training the LORA model.
- [Flux LORA](https://github.com/Flux-AI-Labs/flux-lora) for the LORA training technique.