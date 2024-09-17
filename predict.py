import argparse
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json

# Function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    
    print("Loading checkpoint...")
    checkpoint = torch.load(filepath)
    model_name = checkpoint['architecture']
    model = getattr(models, model_name)(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.eval()

    print("Checkpoint is loaded successfully!!")

    return model

# Process a PIL image for use in a PyTorch model
def process_image(image_path):

    print("Start processing image....")

    # Define the transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load image
    image = Image.open(image_path)
    # Apply the transformations
    np_image = preprocess(image).float()
    
    print("processing image is completed!!!")

    return np_image

# Predict the class from an image file
def predict(image_path, model, cat_to_name, device, topk=5):
    
    print("Start prediction...")

    model.cpu()
    model.eval()

    # Process the image
    pil_img = Image.open(image_path)
    image = process_image(image_path)
    image = image.type(torch.FloatTensor).unsqueeze_(0)
    model, image = model.to(device), image.to(device)
    # Make prediction
    with torch.no_grad():
        output = model.forward(image)
    
    # Get probabilities and class indices
    prediction = F.softmax(output.data, dim=1)
    topk_probs, topk_idx = prediction.topk(topk) 
    topk_probs = topk_probs.cpu().numpy().flatten()
    topk_idx = topk_idx.cpu().numpy().flatten()
    
    # Convert indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    topk_classes = [idx_to_class[idx] for idx in topk_idx]
    topk_names = [cat_to_name[i] for i in topk_classes]

    print("prediction is completed!!!")
    return topk_classes, topk_names, topk_probs

# Main function to parse arguments and run predict
def main():
    parser = argparse.ArgumentParser(description='Predict image class using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top classes to return')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    
    args = parser.parse_args()
    
    # Load category names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        cat_to_name = {}

    # Load model checkpoint
    model = load_checkpoint(args.checkpoint)
    
    # Move model to GPU if available and requested
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Make prediction
    classes, names, probs = predict(args.image_path, model, cat_to_name, device, topk=args.top_k)
    
    # Print results
    print("Top K Classes and Probabilities:")
    for i in range(len(classes)):
        print(f"Class: {classes[i]}\t, Probability: {probs[i]*100:.3f}%\t, Name: {names[i]}")

    print('finished!!!')

if __name__ == '__main__':
    main()

## Command to run the file
# python predict.py path/to/image.jpg path/to/checkpoint.pth --top_k  --category_names path/to/category_names.json --gpu
# EX: python predict.py D:\Fatma\projects\aipnd-project\flowers\test\34\image_06961.jpg D:\Fatma\projects\aipnd-project\checkpoints\checkpoint.pth --top_k 5 --gpu 