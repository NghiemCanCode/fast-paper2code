from torch.utils.data import DataLoader

def get_dataloader(dataset, batch_size, shuffle=True):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=1, #TODO
        pin_memory=False,
        shuffle=shuffle
    )
    return dataloader
