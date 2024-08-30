import argparse
import os
import time
import zipfile
import replicate

def is_zip_file(file_path):
    # Check file extension
    if not file_path.lower().endswith('.zip'):
        print ("not a zip file according to path")
        return False

    # Commented away as the mimetype check on windows relies on the OS's MIME type database based on the file's extension which returns application/x-zip-compressed instead of application/zip
    # Check mime type
    # mime_type, _ = mimetypes.guess_type(file_path)
    # if mime_type != 'application/zip':
    #     print ("not a zip file according to mime type")
    #     print(mime_type)
    #     return False

    # Try to open as a zip file
    try:
        with zipfile.ZipFile(file_path, 'r'):
            return True
    except zipfile.BadZipFile as error:
        print ("not a zip file according to zipfile")
        print(error)
        return False

def create_flux_lora(owner: str, model_name: str, image_path: str, token: str = "TOK"):
    # Check if REPLICATE_API_TOKEN is set
    if "REPLICATE_API_TOKEN" not in os.environ:
        raise ValueError("REPLICATE_API_TOKEN environment variable is not set")

    # Create the model
    model = replicate.models.create(
        owner=owner,
        name=model_name,
        visibility="private",
        hardware="gpu-t4"  # will be overridden in the flux training step
    )

    print(f"Model created: {model.owner}/{model.name}")

    # Start the training
    with open(image_path, "rb") as f:
        print(f)
        training = replicate.trainings.create(
            version="ostris/flux-dev-lora-trainer:4ffd32160efd92e956d39c5338a9b8fbafca58e03f791f6d8011f3e20e8ea6fa",
            input={
                "input_images": f,
                "steps": 1000,
                "prefix": f"A photo of {token}, "
            },
            destination=f"{model.owner}/{model.name}"
        )

    print(f"Training started: {training.status}")
    print(f"Training URL: https://replicate.com/p/{training.id}")

    # Monitor training progress
    while training.status not in ["succeeded", "failed", "canceled"]:
        time.sleep(10)  # Check every 10 seconds
        training.reload()
        print(f"Training status: {training.status}")

    if training.status == "succeeded":
        print("Training completed successfully!")
        print(f"Model URL: https://replicate.com/{model.owner}/{model.name}")
        
        # Get the latest version
        model.reload()
        latest_version = model.latest_version
        
        print("To run the model, use the following code:")
        print(f"""
import replicate

output = replicate.run(
    "{model.owner}/{model.name}:{latest_version.id}",
    input={{
        "prompt": "A portrait photo of a {token}, your description here",
        "num_inference_steps": 28,
        "height": 1024,
        "width": 1024,        
        "guidance_scale": 3.5,
        "model": "dev",
    }}
)

print(f"Generated image URL: {{output}}")
        """)
    else:
        print(f"Training failed or was canceled. Status: {training.status}")
        print(f"Training logs: {training.logs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Flux LORA using replicate.com")
    parser.add_argument("owner", help="Your GitHub username")
    parser.add_argument("model_name", help="Name for the LORA model")
    parser.add_argument("image_path", help="Path to the zip file containing training images")
    parser.add_argument("token", help="Token for the concept to be learned")

    args = parser.parse_args()

    if not is_zip_file(args.image_path):
        raise ValueError("Invalid image path: must be a zip file")

    create_flux_lora(args.owner, args.model_name, args.image_path, args.token)