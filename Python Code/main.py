# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms as T
from ML_models import SimpleNN
from dataSet import CustomDataset

# Optional, for visualization purposes.
import matplotlib.pyplot as plt

model = ""
default = "SNN"
LEARNING_RATE = 0.001


def main():
    transform = T.Compose([T.ToTensor()])

    csv_file_path = "../../BaseData/kddcup99_converted.csv"
    dataset = CustomDataset(csv_file_path, transform=transform)

    # Step 4: Create a DataLoader
    batch_size = 32
    train_loader = T.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = T.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # Step 5: Iterate through the D

    model = SimpleNN()

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.NLLLoss()

    for epoch in range(30):
        model.train()  # Tells PyTorch that we want to accumulate derivatives during the computations
        for batch, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()  # resets the derivatives used by the optimizer
            out = model(data)

            loss = loss_func(torch.log(out), targets)

            loss.backward()  # computes derivatives
            # updates the model weights based on the previous derviatives.
            optimizer.step()

        model.eval()  # tells pytorch not to track derivatives

        # we test the model after every training epoch in order to measure how the model
        # generalizes to unseen data
        correct = 0
        for batch, (data, targets) in enumerate(test_loader):
            out = model(data)
            # this just takes the maximum probability label as the predicted label
            best_guesses = out.argmax(dim=1)
            correct += (best_guesses == targets).sum()

        print(f'Test Correct: {correct/len(dataset)}')

    torch.save(model.state_dict(), 'test.pt')


if __name__ == '__main__':
    main()
