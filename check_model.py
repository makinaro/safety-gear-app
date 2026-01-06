from ultralytics import YOLO

model = YOLO('best.pt')
print('Model classes:')
for i, name in model.names.items():
    print(f'  {i}: {name}')
print(f'\nTotal classes: {len(model.names)}')
