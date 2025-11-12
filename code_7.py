import os

def convert_bbox_to_polygon(label_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for file in os.listdir(label_path):
        if not file.endswith('.txt'):
            continue
        with open(os.path.join(label_path, file), 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip malformed lines
            cls, x, y, w, h = map(float, parts)

            # Convert bbox to polygon (rectangle corners)
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y - h / 2
            x3 = x + w / 2
            y3 = y + h / 2
            x4 = x - w / 2
            y4 = y + h / 2

            polygon = f"{int(cls)} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"
            new_lines.append(polygon)

        with open(os.path.join(output_path, file), 'w') as f:
            f.write('\n'.join(new_lines))

# Convert train and val labels
convert_bbox_to_polygon(
    label_path='labels/train',
    output_path='label_seg/train'
)

convert_bbox_to_polygon(
    label_path='labels/val',
    output_path='label_seg/val'
)