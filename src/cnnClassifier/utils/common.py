import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(yaml_path: Path) -> ConfigBox:
    """Read yaml file and return 

    Args: 
        yaml_path(str) path like input 

    Raises:
        ValueError if yaml file is empty
        e: Empty file
    
    Returns:
        ConfigBox: ConfigBox type"""
    
    try:
        with open(yaml_path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {yaml_file.name} is loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories:list, verbose=True):
    """Create a list of directories

    Args:
        path_to_directories(list) list of path directories
        verbose: (bool, optional) ignore if multiple dirs is to be created, default true
    
    Return None
    """

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data
    
    Args:
        path(Path): path to json file
        data(dict): data to be saved in json file
    """

    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """ load json files data 
        Args: 
            path(Path) path for json file
        Returns:
            ConfigBox: data as class attribute instead of dict
    """

    with open(path) as json_file:
        content = json.load(json_file)
    
    logger.info(f"json file is loaded successfully from {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """Save data in binary format
    Args:
        data(Any): the data that we want to save
        path(Path): binary file path
    """

    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path)->Any:
    """Load Binary data
    Args:
        path(Path): binary file Path
    Returns:
            Any type
    """
    with open(path) as bin_file:
        data = joblib.load(bin_file)
    logger.info(f"bonary file is loaded successfully from {bin_file}")
    
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """get size  in KB
    
    Args:
        path(Path): path of the file
    
    Returns:
        str: size in KB"""
    
    size_in_KB = round(os.path.getsize(path)/1024)
    return f"{size_in_KB} KB"
    

@ensure_annotations
def decode_image(imgstring, filename):
    """Decode image in base64 format and save it
    
    Args: 
        imgstring: image in base64 format
        filename: filename path to save the image"""
    
    imgdata = base64.b64decode(imgstring)
    
    with open(filename, 'wb') as f:
        f.write(imgdata)
        f.close()


@ensure_annotations
def encode_image_into_Base64(cropped_image_path):
    """Encode image into Base64
    Args:
        cropped_image_path: image to base64
    
    Returns:
        image in base64 format"""
    
    with open(cropped_image_path, "rb") as f:
        return base64.b64encode(f.read())
    
    