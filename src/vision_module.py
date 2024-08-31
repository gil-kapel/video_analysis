import pandas as pd
from utils import extract_frames
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch
import pandas as pd
import numpy as np
import easyocr
from tqdm.auto import tqdm
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection
from torchvision import transforms
d = "mps" if torch.backends.mps.is_available() else "cpu"
if d == 'cpu':
    d = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(d)

def get_vision_data(video) -> str:
    frames = extract_frames(video)
    characters_per_frame = get_vision_characters(frames)
    #TODO: extract characters, tables, objects, entities, motion analysis
    return ' '.join(characters_per_frame)


def get_vision_characters(frames):
    vision_characters = {}
    for frame in frames:
        frame_characters = optical_character_recognition(frame)
        if len(frame_characters) > 0:
            vision_characters[frame] = frame_characters
    return vision_characters


def optical_character_recognition(image_arr: np.array, language: str = 'en'):
    """
    Extract text from an image using OCR.

    :param image_arr: Image in numpy array format.
    :param language: Language(s) for OCR, default is English ('en').
    :return: A list of detected text segments.
    """

    image = Image.fromarray(image_arr).convert("RGB")
    reader = easyocr.Reader([language])

    preprocess = transforms.Compose([
        MaxResizer(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    processed_image = preprocess(image).permute(1, 2, 0).numpy()
    ocr_results = reader.readtext(processed_image)
    extracted_text = [" ".join([result[1] for result in ocr_results])]
    return extracted_text


def table_extraction(frame):
    #TODO: implement table extraction from an image algorithm
    return pd.DataFrame()


def object_detection(frame):
    #TODO: implement object extraction from an image algorithm
    return []


def entity_detection(frame):
    #TODO: implement entity extraction from an image algorithm
    return []


class MaxResizer:
    def __init__(self, max_dim=800):
        self.max_dim = max_dim

    def __call__(self, image):
        width, height = image.size
        max_current_dim = max(width, height)
        scale_factor = self.max_dim / max_current_dim
        return image.resize((int(round(scale_factor * width)), int(round(scale_factor * height))))


def extract_table_from_file(file_path: str, language: str = None):
    language = ['en'] if language is None else language
    reader = easyocr.Reader(language)
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")

    image = Image.open(file_path).convert("RGB")
    model.to(device)

    preprocess = transforms.Compose([
        MaxResizer(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    pixel_values = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values)

    id2label = model.config.id2label
    id2label[len(id2label)] = "no object"
    detected_objects = outputs_to_objects(outputs, image.size, id2label)

    detection_thresholds = {
        "table": 0.5,
        "table rotated": 0.5,
        "no object": 10
    }

    table_crops = objects_to_crops(image, [], detected_objects, detection_thresholds)
    cropped_table = table_crops[0]['image'].convert("RGB")

    structure_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
    structure_model.to(device)

    preprocess_structure = transforms.Compose([
        MaxResizer(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    pixel_values = preprocess_structure(cropped_table).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = structure_model(pixel_values)

    structure_id2label = structure_model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"

    cells = outputs_to_objects(outputs, cropped_table.size, structure_id2label)
    cell_coords = extract_cell_coordinates(cells)
    data = perform_ocr(cell_coords, cropped_table, reader)

    return data


def convert_bbox_format(boxes):
    x_center, y_center, width, height = boxes.unbind(-1)
    return torch.stack([x_center - 0.5 * width, y_center - 0.5 * height,
                        x_center + 0.5 * width, y_center + 0.5 * height], dim=1)


def scale_bboxes(bboxes, size):
    img_width, img_height = size
    bboxes = convert_bbox_format(bboxes)
    return bboxes * torch.tensor([img_width, img_height, img_width, img_height], dtype=torch.float32)


def outputs_to_objects(outputs, img_size, id2label):
    probs = outputs.logits.softmax(-1).max(-1)
    pred_labels = probs.indices[0].cpu().numpy()
    pred_scores = probs.values[0].cpu().numpy()
    pred_bboxes = scale_bboxes(outputs['pred_boxes'][0].cpu(), img_size).tolist()

    return [
        {'label': id2label[int(label)], 'score': float(score), 'bbox': bbox}
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes)
        if id2label[int(label)] != 'no object'
    ]


def objects_to_crops(image, tokens, objects, class_thresholds, padding=10):
    table_crops = []

    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        bbox = [obj['bbox'][0] - padding, obj['bbox'][1] - padding,
                obj['bbox'][2] + padding, obj['bbox'][3] + padding]
        cropped_img = image.crop(bbox)

        for token in tokens:
            token['bbox'] = [
                token['bbox'][0] - bbox[0], token['bbox'][1] - bbox[1],
                token['bbox'][2] - bbox[0], token['bbox'][3] - bbox[1]
            ]

        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in tokens:
                token['bbox'] = [
                    cropped_img.size[0] - token['bbox'][3] - 1,
                    token['bbox'][0],
                    cropped_img.size[0] - token['bbox'][1] - 1,
                    token['bbox'][2]
                ]

        table_crops.append({'image': cropped_img, 'tokens': tokens})

    return table_crops


def extract_cell_coordinates(table_data):
    rows = sorted([entry for entry in table_data if entry['label'] == 'table row'], key=lambda x: x['bbox'][1])
    columns = sorted([entry for entry in table_data if entry['label'] == 'table column'], key=lambda x: x['bbox'][0])

    cell_coords = []

    for row in rows:
        row_cells = [
            {'column': col['bbox'], 'cell': [col['bbox'][0], row['bbox'][1], col['bbox'][2], row['bbox'][3]]}
            for col in columns
        ]
        row_cells.sort(key=lambda x: x['column'][0])
        cell_coords.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    return sorted(cell_coords, key=lambda x: x['row'][1])


def perform_ocr_for_cropped_image(cell_coords, cropped_image, ocr_reader):
    data = {}
    max_columns = 0

    for idx, row in enumerate(tqdm(cell_coords)):
        row_text = []
        for cell in row["cells"]:
            cell_img = np.array(cropped_image.crop(cell["cell"]))
            text_result = ocr_reader.readtext(cell_img)
            if text_result:
                row_text.append(" ".join([res[1] for res in text_result]))

        max_columns = max(max_columns, len(row_text))
        data[idx] = row_text

    for row, text in data.items():
        text.extend([""] * (max_columns - len(text)))

    return pd.DataFrame.from_dict(data, orient='index')
