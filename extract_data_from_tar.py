import tarfile
from io import BytesIO
from PIL import Image
from pathlib import Path
import os
import json

# Paths
# Template path with a placeholder for numbers
template_path = "/leonardo_scratch/large/usertrain/a08trb15/cc12m_wds/cc12m-{:06d}.tar"
path2save = "/leonardo_scratch/large/usertrain/a08trb15/cc12m_subset"

# Ensure save directories exist
images_save_path = Path(path2save) / "images"
captions_save_path = Path(path2save) / "caption"
partition_save_path = Path(path2save) / "partition"
images_save_path.mkdir(parents=True, exist_ok=True)
captions_save_path.mkdir(parents=True, exist_ok=True)
partition_save_path.mkdir(parents=True, exist_ok=True)

def process(input_tar, images_save_path, captions_save_path, partition_save_path, index):
    with tarfile.open(input_tar, 'r') as input_tar_file:
        partition_file_path = partition_save_path / f'part{index}.txt'
        with open(partition_file_path, 'w') as partition_file:
            for member in input_tar_file.getmembers():
                if member.isfile():
                    input_data = input_tar_file.extractfile(member)
                    
                    if member.name.endswith('.jpg'):
                        # Process image
                        img = Image.open(BytesIO(input_data.read()))
                        image_save_path = images_save_path / Path(member.name).name
                        img.save(image_save_path, 'JPEG')
                    
                    elif member.name.endswith('height.txt') or member.name.endswith('width.txt'):
                        pass

                    elif member.name.endswith('.txt'):
                        # Process caption
                        caption_save_path = captions_save_path / Path(member.name).name
                        with open(caption_save_path, 'wb') as f:
                            f.write(input_data.read())
    json_name = "data_info.json"
    with open(os.path.join(partition_save_path, json_name), 'w') as json_file:
        data_info = []
        image_files = [file for file in os.listdir(images_save_path)]
        for image_file in image_files:
            image_path = os.path.join(images_save_path, image_file)
            with Image.open(image_path) as img:
                height,width = img.size
            path = os.path.join('images', image_file)
            text_file = os.path.join(captions_save_path, image_file.replace('.jpg', '.txt'))
            # Open text file
            with open(text_file, 'r') as f:
                # Process text data (e.g., read, analyze, etc.)
                prompt = f.read()
            ratio = 1
            data_info.append({
            "height": height,
            "width": width,
            "ratio": ratio,
            "path": path,
            "prompt": prompt,
            })
        json.dump(data_info,json_file)

# Get a list of all .tar files in the folder
start = 1 #1
end = 4000 #100
tar_files = [template_path.format(i) for i in range(start, end + 1)]

# Apply the function to each .tar file in the folder
for index, tar_file in enumerate(tar_files):
    process(tar_file, images_save_path, captions_save_path, partition_save_path, start + index - 1)

print('Done')