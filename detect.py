import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from config import label_map, rev_label_map, label_color_map
from model import SSD300

device = torch.device('cuda:0')


def detect():
    # load the image
    image_path = 'data/JPEGImages/COCO_train2014_000000565884.jpg'
    original_image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(size=(300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(original_image)
    image = image.to(device)

    # load the model
    checkpoint_path = 'weights/ssd300.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SSD300(n_classes=len(label_map), device=device)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])

    # detect objects
    model.eval()
    with torch.no_grad():
        predicted_bboxes, predicted_scores = model(image.unsqueeze(0))
    detected_bboxes, detected_labels, detected_scores = model.detect_objects(
        predicted_bboxes,
        predicted_scores,
        min_score=0.2,
        max_overlap=0.5,
        top_k=200
    )

    detected_bboxes = detected_bboxes[0].to('cpu')
    detected_labels = detected_labels[0].to('cpu').tolist()
    detected_scores = detected_scores[0].to('cpu').tolist()

    # transfer bounding boxes to absolute coordinates
    w = original_image.width
    h = original_image.height
    dims = torch.FloatTensor([w, h, w, h]).unsqueeze(0)
    detected_bboxes *= dims

    # transfer label indices to string names
    detected_labels = [rev_label_map[i] for i in detected_labels]

    if detected_labels == ['background']:
        print('No object detected.')
        original_image.show()
        return

    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype('C:/Windows/Fonts/Arial.ttf', size=15)

    for i in range(detected_bboxes.size(0)):
        color = label_color_map[detected_labels[i]]

        # draw bounding box
        bbox = detected_bboxes[i].tolist()
        draw.rectangle(xy=bbox, outline=color, width=4)

        # draw label
        text = f'{detected_labels[i].lower()} {detected_scores[i]:.4f}'
        text_size = font.getsize(text)  # (width, height)
        text_location = [bbox[0], bbox[1]]
        text_bbox_location = [bbox[0], bbox[1], bbox[0] + text_size[0], bbox[1] + text_size[1]]
        draw.rectangle(xy=text_bbox_location, fill=color, outline=color)
        draw.text(xy=text_location, text=text, fill='black', font=font)

    annotated_image.show()
    return


if __name__ == '__main__':
    detect()
