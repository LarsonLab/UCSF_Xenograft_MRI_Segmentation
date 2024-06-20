import torch
from tqdm import tqdm

def train(model_name, model, optimizer, criterion, train_loader, val_loader, device, num_epochs, clear_mem):
    torch.cuda.empty_cache() 
    print(f"Using device: {device}")
    print(f'Model sent to {device}')
    model = model.to(device)
    all_dice_train_losses = []
    all_dice_val_losses = []
    iters = 0

    for epoch in range(num_epochs): 
        print(f"Epoch {epoch+1} / {num_epochs}")
        
        model.train()    
        dice_train_losses = []

        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch:{epoch+1}/{num_epochs}"): 
            try:
                img = batch[0].float().to(device)
                msk = batch[1].float().to(device)
                optimizer.zero_grad()
                output = model(img)
                loss = criterion(output, msk)
                loss.backward()
                optimizer.step()
                
                dice_train_losses.append(loss.item())
                iters += 1
            except Exception as e:
                print(f"Error during training at iteration {i}: {e}")
        
        all_dice_train_losses.append(sum(dice_train_losses) / len(dice_train_losses))

        model.eval()
        dice_val_losses = []
        dice_val_preds = []
        dice_val_labels = []
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader): 
                try:
                    torch.cuda.empty_cache()  
                    img = batch[0].float().to(device)
                    msk = batch[1].float().to(device)
                    output = model(img)
                    loss = criterion(output, msk)
                    dice_val_losses.append(loss.item())
                    dice_val_labels.append(msk)
                    dice_val_preds.append(output)
                except Exception as e:
                    print(f"Error during validation at iteration {i}: {e}")
        
        all_dice_val_losses.append(sum(dice_val_losses) / len(dice_val_losses))
        print(f'Epoch {epoch+1} completed. Train Loss: {all_dice_train_losses[-1]}, Val Loss: {all_dice_val_losses[-1]}')

        torch.cuda.empty_cache() 
        
    epoch_num = 1
    plot_dir = log_directory_test + "/plots" + str(epoch_num)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    save_plots(
        epoch_num,
        all_dice_train_losses,
        all_dice_val_losses,
        plot_dir
    )
       
    results = {
        'model_name': model_name,
        'train_losses': all_dice_train_losses,
        'val_losses': all_dice_val_losses,
    }
    print(results)
        
    if clear_mem:
        del model, optimizer, criterion
        torch.cuda.empty_cache()    
    return results