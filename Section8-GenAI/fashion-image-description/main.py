"""
Section 8 - Lecture 8: Generate Fashion Image Descriptions with Gemini
Demonstrates Gemini's native multimodal capabilities for image understanding.

Replaces the old approach that required:
  1. Deploying a Salesforce BLIP model on Vertex AI (GPU endpoint)
  2. Using BLIP for image captioning
  3. Using PaLM 2 to expand captions into product descriptions

Now: Gemini handles everything in a single API call -- no GPU, no deployment.

Usage:
    python main.py

Prerequisites:
    pip install google-cloud-aiplatform google-cloud-storage Pillow matplotlib
    gcloud auth application-default login
"""

import io
import vertexai
from google.cloud import storage
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part, Image

# ─── Configuration ────────────────────────────────────────────────────────────
PROJECT_ID = "your-project-id"  # <-- CHANGE THIS
LOCATION = "us-central1"
GCS_BUCKET = "github-repo"
IMAGE_PREFIX = "product_img/"

# ─── Initialize ───────────────────────────────────────────────────────────────
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.0-flash")
storage_client = storage.Client()


def read_image_from_gcs(bucket_name: str, image_path: str) -> bytes:
    """Read an image from Google Cloud Storage and return raw bytes."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(image_path)
    return blob.download_as_bytes()


def generate_product_description(image_bytes: bytes, image_name: str) -> str:
    """
    Generate a product description from a fashion image using Gemini.

    Gemini processes the image directly -- no need for a separate
    captioning model like BLIP. This replaces ~50 lines of BLIP
    deployment and inference code with a single API call.
    """
    # Create an Image part from the raw bytes
    image_part = Part.from_data(data=image_bytes, mime_type="image/jpeg")

    prompt = """You are a digital marketer working for a retail fashion organization.
You are an expert in building detailed and catchy product descriptions for a fashion e-commerce website.

Look at this fashion product image and generate:
1. A short, catchy product title (one line)
2. A detailed product description (2-3 sentences) that highlights the style, material, and occasions to wear it
3. Three relevant hashtags for social media

Format your response as:
Title: ...
Description: ...
Hashtags: ...
"""

    response = model.generate_content(
        [image_part, prompt],
        generation_config=GenerationConfig(
            temperature=0.4,
            max_output_tokens=512,
            top_p=0.8,
            top_k=40,
        ),
    )
    return response.text


def main():
    """Process all fashion images from GCS and generate descriptions."""
    print("=" * 70)
    print("Fashion Product Description Generator (Gemini Multimodal)")
    print("=" * 70)

    # Optional: try to import matplotlib for image display
    try:
        from matplotlib import pyplot as plt
        from PIL import Image as PILImage
        can_display = True
    except ImportError:
        can_display = False
        print("(matplotlib/Pillow not installed -- skipping image display)\n")

    for i in range(1, 9):
        image_name = f"fashion{i}.jpeg"
        image_path = f"{IMAGE_PREFIX}{image_name}"

        print(f"\n{'─' * 70}")
        print(f"Processing: {image_name}")
        print(f"{'─' * 70}")

        try:
            # Read image from GCS
            image_bytes = read_image_from_gcs(GCS_BUCKET, image_path)

            # Display the image if matplotlib is available
            if can_display:
                pil_image = PILImage.open(io.BytesIO(image_bytes))
                plt.figure(figsize=(4, 4))
                plt.imshow(pil_image)
                plt.axis("off")
                plt.title(image_name)
                plt.show()

            # Generate product description with Gemini (single API call!)
            description = generate_product_description(image_bytes, image_name)
            print(description)

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    print(f"\n{'=' * 70}")
    print("Done! All product descriptions generated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
