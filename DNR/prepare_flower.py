import os
import tarfile
import scipy.io
import shutil
from tqdm import tqdm
from urllib.request import urlretrieve

# URLs
BASE_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
IMAGES_URL = BASE_URL + "102flowers.tgz"
LABELS_URL = BASE_URL + "imagelabels.mat"
SETID_URL = BASE_URL + "setid.mat"

# Output directory
DEST_ROOT = "flower102"
IMG_DIR = os.path.join(DEST_ROOT, "jpg")

# Download and extract helper
def download_and_extract(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading: {url}")
        urlretrieve(url, dest_path)
    if dest_path.endswith(".tgz") or dest_path.endswith(".tar"):
        with tarfile.open(dest_path) as tar:
            tar.extractall(path=os.path.dirname(dest_path))

def prepare_flower102_structure():
    print("Downloading and extracting dataset...")

    os.makedirs(DEST_ROOT, exist_ok=True)
    download_and_extract(IMAGES_URL, os.path.join(DEST_ROOT, "102flowers.tgz"))
    download_and_extract(LABELS_URL, os.path.join(DEST_ROOT, "imagelabels.mat"))
    download_and_extract(SETID_URL, os.path.join(DEST_ROOT, "setid.mat"))

    # Load splits
    setid = scipy.io.loadmat(os.path.join(DEST_ROOT, "setid.mat"))
    labels = scipy.io.loadmat(os.path.join(DEST_ROOT, "imagelabels.mat"))['labels'][0]

    def make_split(name, indices):
        split_dir = os.path.join(DEST_ROOT, "Flower102", name)
        os.makedirs(split_dir, exist_ok=True)

        for i in tqdm(indices, desc=f"Preparing {name}"):
            img_id = f"image_{i:05d}.jpg"
            label = labels[i - 1]
            class_dir = os.path.join(split_dir, f"class_{label:03d}")
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(IMG_DIR, img_id)
            dst = os.path.join(class_dir, img_id)
            shutil.copyfile(src, dst)

    # Split creation
    make_split("train", setid["trnid"][0])
    make_split("val", setid["valid"][0])
    make_split("test", setid["tstid"][0])

    print("✅ Flower102 prepared in ImageFolder format.")
    print(f"Train path: {os.path.abspath(os.path.join(DEST_ROOT, 'Flower102', 'train'))}")

if __name__ == "__main__":
    prepare_flower102_structure()
