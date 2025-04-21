import os
import requests
from PIL import Image
from io import BytesIO
import time
import random
from pathlib import Path

class ImageDownloader:
    def __init__(self, output_dir='data'):
        self.output_dir = output_dir
        self.bloodstain_dir = os.path.join(output_dir, 'bloodstain')
        self.no_bloodstain_dir = os.path.join(output_dir, 'no_bloodstain')
        
        # Create output directories if they don't exist
        os.makedirs(self.bloodstain_dir, exist_ok=True)
        os.makedirs(self.no_bloodstain_dir, exist_ok=True)
        
        # Unsplash API endpoint and headers
        self.api_url = "https://api.unsplash.com"
        self.headers = {
            'Authorization': 'Client-ID YOUR_ACCESS_KEY'  # You'll need to replace this
        }
        
        # Search queries for different types of images
        self.bloodstain_queries = [
            "blood stain",
            "blood spatter",
            "blood pattern",
            "blood drop"
        ]
        
        self.no_bloodstain_queries = [
            "clean surface",
            "clean floor",
            "clean wall",
            "clean fabric"
        ]
    
    def search_unsplash(self, query, per_page=30):
        """Search Unsplash for images"""
        try:
            response = requests.get(
                f"{self.api_url}/search/photos",
                headers=self.headers,
                params={
                    'query': query,
                    'per_page': per_page,
                    'orientation': 'landscape'  # Get landscape images for better quality
                }
            )
            response.raise_for_status()
            return response.json()['results']
        except Exception as e:
            print(f"Error searching Unsplash for {query}: {e}")
            return []
    
    def download_image(self, photo, output_path):
        """Download and save an image from Unsplash"""
        try:
            # Get the regular size URL (good quality but not too large)
            url = photo['urls']['regular']
            
            response = requests.get(url)
            response.raise_for_status()
            
            # Open the image
            img = Image.open(BytesIO(response.content))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to a standard size
            img = img.resize((224, 224))
            
            # Save the image
            img.save(output_path, 'JPEG')
            return True
        except Exception as e:
            print(f"Error downloading image: {e}")
            return False
    
    def download_images(self, queries, output_dir, num_images=100):
        """Download images for a set of queries"""
        downloaded = 0
        for query in queries:
            if downloaded >= num_images:
                break
                
            print(f"Searching for: {query}")
            photos = self.search_unsplash(query)
            
            for photo in photos:
                if downloaded >= num_images:
                    break
                
                # Generate a unique filename
                filename = f"{downloaded + 1}.jpg"
                output_path = os.path.join(output_dir, filename)
                
                if self.download_image(photo, output_path):
                    downloaded += 1
                    print(f"Downloaded {downloaded}/{num_images}: {photo['alt_description'] or 'Untitled'}")
                    
                    # Add a small delay to be respectful to the server
                    time.sleep(1)
    
    def run(self, num_images_per_class=100):
        """Run the download process for both classes"""
        print("Downloading bloodstain images...")
        self.download_images(self.bloodstain_queries, self.bloodstain_dir, num_images_per_class)
        
        print("\nDownloading non-bloodstain images...")
        self.download_images(self.no_bloodstain_queries, self.no_bloodstain_dir, num_images_per_class)

if __name__ == "__main__":
    downloader = ImageDownloader()
    downloader.run(num_images_per_class=100) 