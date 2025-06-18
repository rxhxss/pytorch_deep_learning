import gdown
import zipfile
import os
import pathlib
def download_and_extract_dataset():
    # Dataset URL (replace with your actual Google Drive link)
    dataset_url = 'https://drive.google.com/uc?id=1EY9sFE08r4C0tO5DVtwbXcJCxucfFIEo'
    output_zip = 'FACECOM.zip'
    extract_to = './data'

    # Download zip file
    gdown.download(dataset_url, output_zip, quiet=False)

    # Extract zip file
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Remove zip file
    os.remove(output_zip)

    print(f"Dataset extracted to {extract_to}")

# Call this function at the start of your training script
if not os.path.exists('./data/FACECOM'):
    download_and_extract_dataset()

import torch
import torchvision
import data_setup,engine,evaluation_metrics,model_builder,utils

#Setting up device agnostic code
device="cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS=os.cpu_count()
EPOCHS=5


train_dir="/content/data/Comys_Hackathon5/Task_A/train"
test_dir="/content/data/Comys_Hackathon5/Task_A/val"



auto_transforms,model=model_builder.model_creation(num_classes=2,device=device)
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params=model.parameters(),lr=0.001)


train_dataloader,test_data_loader,class_names=data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=auto_transforms,
    batch_size=32,
   num_workers=NUM_WORKERS)


results=engine.train(model,train_dataloader,test_data_loader,optimizer,loss_fn,
                     EPOCHS,device)


metrics=evaluation_metrics.evaluate_model(model,test_dir,auto_transforms,32,device,num_classes=2)
evaluation_metrics.print_metrics(metrics)
utils.save_model(model,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
