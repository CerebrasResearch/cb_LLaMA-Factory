import functools
from typing import List, Callable
import re
from io import BytesIO
import base64


class DatasetTransform:
    _registry = {}

    @staticmethod
    def register(name: str):
        def decorator(func: Callable):
            if name in DatasetTransform._registry:
                raise ValueError(f"Dataset transform '{name}' is already registered.")
            DatasetTransform._registry[name] = func
            return func
        return decorator

    @staticmethod
    def get_function(name):
        if name in DatasetTransform._registry:
            return DatasetTransform._registry[name]
        else:
            raise ValueError(f"Dataset transform '{name}' is not registered.")

    @staticmethod
    def apply_transforms(dataset, transform_list: List[str]):

        for transform_name in transform_list:
            transform_func = DatasetTransform.get_function(transform_name)
            if not callable(transform_func):
                raise ValueError(f"Transform '{transform_name}' is not a callable function.")
            print(f"Applying transform: {transform_name}")
            dataset = dataset.map(transform_func, batched=False)

        return dataset


@DatasetTransform.register("synthchart_transform")
def synthchart_transform(example):
    transformed_texts = []
    loss_mask = []
    num_images = len(example["images"])
    for i, text_dict in enumerate(example["texts"]):
        if num_images:
            if i < num_images:
                transformed_texts.append({"role": "user", "content": "<image>" + text_dict["user"]})
            else:
                transformed_texts.append({"role": "user", "content": text_dict["user"]})
        transformed_texts.append({"role": "assistant", "content": text_dict["assistant"]})
        loss_mask.extend([0, 1])
    
    example["texts"] = transformed_texts
    example["loss_mask"] = loss_mask
    return example


def convert_pil_to_base64(image_pil):
    """Convert a PIL image to a base64 string."""
    buffered = BytesIO()
    format = image_pil.format
    image_pil.save(buffered, format=format.lower())
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/{format};base64,{img_b64}"

@DatasetTransform.register("synthchart_transform_fast")
def synthchart_transform_fast(example):
    transformed_texts = []
    text_dict = example["texts"][0]
    transformed_texts.append({"role": "user", "content": "<image>" + text_dict["user"]})    
    transformed_texts.append({"role": "assistant", "content": text_dict["assistant"]})
    loss_mask = [0, 1]
    
    example["texts"] = transformed_texts
    example["loss_mask"] = loss_mask

    image_pil = example["images"][0]  # Assuming images is a list of PIL images and len 1
    image_url = convert_pil_to_base64(image_pil)
    example["_images"] = [image_url]  # Replace with the base64 string
    example["images"] = None
    return example



    

if __name__ == "__main__":
    print(DatasetTransform._registry)
    from datasets import load_dataset
    ds = load_dataset("/cb/home/aarti/.cache/huggingface/hub/datasets--ds4sd--SynthChartNet/snapshots/b913ef98a3d45f6136465963ddc71a7a6b0e1728", split="train")
    print(ds[0])
    ds = DatasetTransform.apply_transforms(ds, ["synthchart_transform_fast"])
    print(ds[0])
