# CS5330 
# Author: Yao Zhong, zhong.yao@northeastern.edu

# import statements
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np


#load digit MNIST
def loadData():
    # Get the training data
    data = datasets.MNIST(
        # put the data into this folder
        root = "data/digits",
        train=True,
        download=True,
        # the transformers of the image
        transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    )
    
    return data

#Helper function that plots the first 6 images from the MNIST digits
def showExamples(training_data):
    size = len(training_data)
    figure=plt.figure(figsize=(8,6)) #8x6 inches window
    cols, rows = 3, (size+2)//3
    for i in range(cols*rows):
        img, label = training_data[i]
        figure.add_subplot(rows, cols, i+1)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


# The structue of the generator model
class GeneratorModel(nn.Module):
    def __init__(self):
        super(GeneratorModel, self).__init__()
        input_dim =100 + 10 #additional 10 dimentions for the embedding of the digit
        # 28*28 is 784, which is the size of the digit images
        output_dim = 784
        #embeddings for the 10 digits
        self.label_embedding = nn.Embedding(10,10)
        # This is from the tutorial architecture.
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
        
    # computes a forward pass for the network
    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x,c],1) #append the emnedings after the noise (100+10)
        output = self.hidden_layer1(x)
        output = self.hidden_layer2(output)
        output = self.hidden_layer3(output)
        output = self.output_layer(output)
        return output
    
# The structue of the discriminator model
class DiscriminatorModel(nn.Module):
    def __init__(self):
        super(DiscriminatorModel, self).__init__()
        # 28*28 is 784, which is the size of the digit images
        input_dim = 784+10 #+10 is the embeding
        # output dim is the possibility of this image is a digit or not
        output_dim = 1
        #embeddings for the 10 digits
        self.label_embedding = nn.Embedding(10,10)

        # This is from the tutorial architecture.
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )
        
    # computes a forward pass for the network
    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x,c], 1)
        output = self.hidden_layer1(x)
        output = self.hidden_layer2(output)
        output = self.hidden_layer3(output)
        output = self.output_layer(output)
        return output

# The function that trians the network and plots the result of the trianing and testing
def train_GAN(dataloader, G, D, loss, G_optimizer, D_optimizer, epochs, device):
    #holder for the result of each epoch
    G_loss = []
    D_loss = []
    counter = []

    #call the train_loop and test_loop for each epoch
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataloader, G, D, loss, G_optimizer, D_optimizer, G_loss, D_loss, counter, t, device)
    print("Done!")

    # Plot the training and testing result
    fig = plt.figure()
    plt.plot(counter, G_loss, color='blue')
    plt.plot(counter, D_loss, color='red')
    plt.legend(['Generater Loss', 'Discriminator Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    return

# The train_loop for one epoch, the statistics are saved in losses
def train_loop(dataloader, G, D, loss, G_optimizer, D_optimizer, G_loss, D_loss, counter, epochIdx, device):
    # size of the whole set, used in counter
    size = len(dataloader.dataset)
    for batchIdx, (data, target) in enumerate(dataloader):
        batch_size = len(data)
        # Generate noise and move it the device
        noise = torch.randn(batch_size,100).to(device)
        fake_labels = torch.randint(0,10,(batch_size, )).to(device)
        #forward
        generated_data = G(noise, fake_labels).to(device) #batch_size x 784

        true_data = data.view(batch_size, -1).to(device) #batch_size x 784
        digit_labels = target.to(device) # batch_size
        true_labels = torch.ones(batch_size).to(device)
        
        # Clear optimizer gradients
        D_optimizer.zero_grad()
        # Forward pass with true data as input
        D_true_output = D(true_data, digit_labels).to(device).view(batch_size)
        # Compute Loss
        D_true_loss = loss(D_true_output, true_labels)
        # Forward pass with generated data as input
        D_generated_output = D(generated_data.detach(), fake_labels).to(device).view(batch_size)
        # Compute Loss 
        D_generated_loss = loss(
            D_generated_output, torch.zeros(batch_size).to(device)
        )
        # Average the loss
        D_average_loss = (
            D_true_loss + D_generated_loss
        ) / 2
               
        # Backpropagate the losses for Discriminator model      
        D_average_loss.backward()
        D_optimizer.step()

        # Clear optimizer gradients
        G_optimizer.zero_grad()
        
        # It's a choice to generate the data again
        generated_data = G(noise, fake_labels).to(device) # batch_size X 784
        # Forward pass with the generated data
        D_generated_output = D(generated_data, fake_labels).to(device).view(batch_size)
        # Compute loss
        generator_loss = loss(D_generated_output, true_labels)
        # Backpropagate losses for Generator model.
        generator_loss.backward()
        G_optimizer.step()
        
       

     
        # log in the terminal for each 10 batches
        if batchIdx % 10 == 0:
            current = (batchIdx+1)*len(data)
            print(f"D_loss: {D_average_loss.data.item():>7f}  G_loss: {generator_loss.data.item():>7f}  [{current:>5d}/{size:>5d}]")
            D_loss.append(D_average_loss.data.item())
            G_loss.append(generator_loss.data.item())
            counter.append(batchIdx*len(data)+epochIdx*size)
        
        if ((batchIdx + 1)% 500 == 0 and (epochIdx + 1)%2 == 0):
            
            with torch.no_grad():
                noise = torch.randn(batch_size,100).to(device)
                fake_labels = torch.randint(0,10,(batch_size, )).to(device)
                generated_data = G(noise,fake_labels).view(batch_size, 28, 28)
                for x in generated_data:
                    plt.imshow(x.detach().cpu().numpy(), interpolation='nearest',cmap='gray')
                    plt.show()
                    break

# main function
def main(argv):
    
    random_seed = 47
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Set the settings for the training
    learning_rate = 1e-4
    batch_size = 64
    epochs = 5
    
    #load traing data
    data = loadData()
    # Uncomment to show the first six images.
    # showExamples(data)

    # create dataloaders
    dataloader = DataLoader(data, batch_size, shuffle=True)

    # Create the discriminator and generator
    generator = GeneratorModel()
    discriminator = DiscriminatorModel()
    # move them to device
    generator.to(device)
    discriminator.to(device)

    # use binary cross-entropy loss
    loss = nn.BCELoss()
    # use Adam optimizer for both model
    generator_optimizer = torch.optim.Adam(generator.parameters(), learning_rate)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), learning_rate)

    train_GAN(dataloader, generator, discriminator, loss, generator_optimizer, discriminator_optimizer, epochs, device)

    # Show the final generated examples
    figure=plt.figure(figsize=(8,6)) #8x6 inches window
    cols, rows = 3, 4
    with torch.no_grad():
        noise = torch.randn(10,100).to(device)
        labels = torch.tensor([0,1,2,3,4,5,6,7,8,9], dtype=torch.int32).to(device)
        generated_data = generator(noise, labels).view(10, 28, 28)
        for idx, x in enumerate(generated_data):
            figure.add_subplot(rows, cols, idx+1)
            plt.axis("off")
            plt.title(labels[idx].item())
            plt.imshow(x.detach().cpu().numpy().squeeze(), interpolation='nearest',cmap='gray')
    plt.show()

    # Save the trained models
    torch.save(discriminator, 'digit_c_gan_D_model.pth')
    torch.save(generator, 'digit_c_gan_G_model.pth')

    return

if __name__ == "__main__":
    main(sys.argv)