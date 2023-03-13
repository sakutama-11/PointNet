from Model import *
from sampler import *


batch_size = 64
num_points = 1024
num_labels = 1
epochs = 100


def main():
    pointnet = PointNet(num_points, num_labels)
    # summary(pointnet, input_size=(batch_size, num_points, 3))


    new_param = pointnet.state_dict()
    new_param['pointNet.0.transT.6.bias'] = torch.eye(3, 3).view(-1)
    new_param['pointNet.3.featT.6.bias'] = torch.eye(64, 64).view(-1)
    pointnet.load_state_dict(new_param)
    pointnet = pointnet.float()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(pointnet.parameters(), lr=0.001)

    loss_list = []
    accuracy_list = []
    
    dataset = PcdDataset(512, num_points)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for input_data, labels in dataloader:
            pointnet.zero_grad()

            input_data = input_data.view(-1, 3)
            labels = labels.view(-1, 1)
            output = pointnet(input_data.float())
            output = nn.Sigmoid()(output)
            
            error = criterion(output, labels)
            # calculate grad
            error.backward()
            # update weights
            optimizer.step()

        with torch.no_grad():
            output[output > 0.5] = 1
            output[output < 0.5] = 0
            accuracy = (output==labels).sum().item()/batch_size
        

        loss_list.append(error.item())
        accuracy_list.append(accuracy)
        torch.save(pointnet.state_dict(), 'pointnet_weight.pth')
        print('epoch : {}   Loss : {}'.format(epoch, error.item()))
        print('epoch : {}   Accuracy : {}'.format(epoch, accuracy))
    
            
            
if __name__ == '__main__':
    main()