import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import json
import argparse
from collections import OrderedDict
import os

# Function to load and transform data
def load_data(data_dir):
    print("Loading data...")
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.25, saturation=0.1, hue=0.01),
                transforms.RandomPerspective(0.2),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # Define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True, num_workers=12),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True, num_workers=12),
        'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True, num_workers=12)
    }

    print("Data loaded successfully!!!")

    return image_datasets, dataloaders

# Function to build and customize model architecture
def build_model(architecture='vgg16', hidden_units=512, output_size=102, dropout=0.2):

    print("Preparing model...")
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088 
    elif architecture == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = 25088 
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024 
    elif architecture == 'densenet161':
        model = models.densenet161(pretrained=True)
        input_size = 2208
    else:
        raise ValueError(f"Architecture '{architecture}' not supported.")
    
    # Freeze feature parameters
    for param in model.parameters():
        param.requires_grad = False
    

        classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(input_size, hidden_units)),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout(dropout)),
                ('fc2', nn.Linear(hidden_units, output_size)),
                ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier

    print("The model is prepared successfully!!!")
    return model

# Function to validate the model
def validate(model, criterion, dataloader, device):
    model.eval()
    accuracy, loss = 0, 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            loss += criterion(output, labels).item()
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return loss / len(dataloader), accuracy / len(dataloader)

# Function to train the model
def train_model(model, dataloaders, criterion, optimizer, device, epochs=10, print_every=20):
    steps = 0

    print(f"Starting training with {epochs} epochs...")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        model.train()
        running_loss = 0
        
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss, valid_accuracy = validate(model, criterion, dataloaders['valid'], device)
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss:.3f}.. "
                      f"Validation accuracy: {valid_accuracy*100:.2f}%")
                running_loss = 0

    print("Training completed!!!")

# Function to test the model on the test dataset
def test_model(model, dataloaders, criterion, device):

    model.eval()
    test_loss = 0
    accuracy = 0

    with torch.no_grad():
        for images, labels in dataloaders['test']:
            images, labels = images.to(device), labels.to(device)

            outputs = model.forward(images)
            batch_loss = criterion(outputs, labels)

            test_loss += batch_loss.item()

            # accuracy
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test Accuracy: {accuracy / len(dataloaders['test']) * 100:.2f}%")

# Function to save the model checkpoint
def save_checkpoint(model, image_datasets, save_dir, architecture, hidden_units, output_size):
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint = {
        'architecture': architecture,
        'hidden_units': hidden_units,
        'output_size': output_size,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier if architecture in ['vgg16','vgg19', 'densenet121', 'densenet161'] else None
        }
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))
    print("Model saved successfully!!!")

# Main function to parse arguments and run training
def main():
    parser = argparse.ArgumentParser(description="Train a deep learning model.")
    
    parser.add_argument('data_dir', type=str, help='Path to the dataset folder.')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Set directory to save checkpoints')
    parser.add_argument('--arch', type=str, choices=['vgg16', 'vgg19', 'densenet121', 'densenet161'], default='vgg16', help='Model architecture (vgg16, vgg19, densenet121, densenet161), default to vgg16')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Set the learning rate hyperparameter, defaults to 0.001')
    parser.add_argument('--hidden_units', type=int, default=512, help='Set the hidden unit amount, defaults to 512')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs, defaults to 20.')
    parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU if available.')

    args = parser.parse_args()

    # Set the device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    #Info
    print(f"Info -- ({args.epochs}) Epochs.    Learning Rate: {args.learning_rate}.    Hidden Units: {args.hidden_units}.    NN Arch: {args.arch}")

    # Load data
    image_datasets, dataloaders = load_data(args.data_dir)

    # Build model
    model = build_model(architecture=args.arch, hidden_units=args.hidden_units, output_size=len(image_datasets['train'].classes))
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Train the model
    train_model(model, dataloaders, criterion, optimizer, device, epochs=args.epochs)

    # Test the model
    test_model(model, dataloaders, criterion, device)

    # Save the checkpoint
    save_checkpoint(model, image_datasets, args.save_dir, architecture=args.arch, hidden_units=args.hidden_units, output_size=len(image_datasets['train'].classes))

if __name__ == '__main__':
    main()

## Command to run file
# python train.py data_dir --arch  --epochs  --learning_rate  --hidden_units  --gpu
# EX: python train.py flowers --arch vgg16 --epochs 1 --learning_rate 0.001 --hidden_units 512 --gpu