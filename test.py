from Model import *
from sampler import *


batch_size = 64
num_points = 1024
num_labels = 1


def test():
    pointnet = PointNet(num_points, num_labels)
    pointnet.load_state_dict(torch.load('pointnet_weight.pth'))
    pointnet = pointnet.float()

    # target = 0
    

    dataset = PcdDataset(4, num_points, for_test=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    pointnet.eval()
    for input_data, labels in dataloader:
        input_data = input_data.view(-1, 3)
        labels = labels.view(-1, 1)
        with torch.no_grad():
            pred = pointnet(input_data.float())
            pred = nn.Sigmoid()(pred)
            pred[pred > 0.5] = 1
            pred[pred < 0.5] = 0
            for predicted, actual in zip(pred, labels):
                print(f'Predicted: "{predicted.item()}", Actual: "{actual.item()}"')
                if predicted.item() == actual.item():
                    print("correct!")
                else:
                    print("incorrect...") 
            
if __name__ == '__main__':
    test()