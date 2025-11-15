import os
import click
from huggingface_hub import snapshot_download
from openface.demo import process_image

def download_weights_from_hf(repo_id, save_path):
    """
    Downloads an entire folder from Hugging Face Model Hub.

    Args:
        repo_id (str): The Hugging Face repo ID.
        save_path (str): The local path to save the folder contents.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print("Downloading weights from Hugging Face...")
        snapshot_download(repo_id=repo_id, local_dir=save_path, repo_type="model")
        print(f"Weights downloaded to {save_path}")
    else:
        print("Weights already exist. Skipping download.")

@click.group()
def cli():
    """Command-line interface for OpenFace."""
    click.echo("CLI initialized")
    pass

@cli.command(name='download')  # Explicitly set command name
@click.option("--repo-id", default="nutPace/openface_weights", help="Hugging Face repo ID")
@click.option("--output", default="./weights", help="Path to save the weights")
def download(repo_id, output):
    """Download weights from Hugging Face."""
    click.echo(f"Starting download from {repo_id}")  # Debug line
    save_path = os.path.abspath(output)
    download_weights_from_hf(repo_id, save_path)

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='results', 
              help='Directory to save results')
@click.option('--device', '-d', default='cuda',
              type=click.Choice(['cuda', 'cpu']),
              help='Device to run inference on')
def detect(image_path, output_dir, device):
    """Process an image and save face analysis results to CSV.
    
    IMAGE_PATH: Path to the input image file
    """
    try:
        output_file = process_image(
            image_path=image_path,
            output_dir=output_dir,
            device=device
        )
        click.echo(f"Results saved to: {output_file}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

@cli.command(name='detect-video')
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='results', 
              help='Directory to save results')
@click.option('--device', '-d', default='cuda',
              type=click.Choice(['cuda', 'cpu']),
              help='Device to run inference on')
def detect_video(video_path, output_dir, device):
    """Process a video and save per-frame face analysis results to CSV.

    VIDEO_PATH: Path to the input video file
    """
    from openface.demo import process_video
    try:
        output_file = process_video(
            video_path=video_path,
            output_dir=output_dir,
            device=device
        )
        click.echo(f"Results saved to: {output_file}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    cli()
