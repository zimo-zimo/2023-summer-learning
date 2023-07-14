import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import gradio as gr
from PIL import Image
from model import *
from CustomDataset import *
from gradio_visualize import *


def unpickle(file):
  with open(file, 'rb') as fo:
      dict1 = pickle.load(fo,encoding='bytes')
  return dict1
  


def train(epoch_num, trainloader, testloader, class_names):

  # net = Net().to(device)
  net = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
  loss_data = []
  test_loss = []
  accuracy_data = []
  min_loss = 100000

  for epoch in range(epoch_num):
    running_loss = 0.0
    for i, itdata in enumerate(trainloader, 0):
      inputs, labels = itdata
      inputs = inputs.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()

      outputs = net(inputs.float())
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if i % 500 == 499:    # print every 2000 mini-batches
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i * 32):.4f}')
    print(f'[{epoch + 1}] loss: {running_loss / (i * 32):.4f}')
    loss_data.append(round(running_loss/50000, 4))
    if running_loss < min_loss:
      min_loss= running_loss
      PATH = './cifar_net.pth'
      torch.save(net.state_dict(), PATH)

    # test to get accuracy per epoch
    correct = 0
    total = 0
    test_loss_data = 0
    with torch.no_grad():
      for testdata in testloader:
        images, labels = testdata
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images.float())

        loss = criterion(outputs, labels)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        test_loss_data += loss.item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    accuracy_data.append(accuracy)
    test_loss.append(round(test_loss_data/10000, 4))

  print('Finished Training')
  print('loss data: ', loss_data)
  print('test loss: ', test_loss)
  print('accuracy data: ', accuracy_data)

  return loss_data, test_loss, accuracy_data, class_names


def getimg(sample):

    image=sample.reshape(-1,1024)
    r=image[0,:].reshape(32,32)
    g=image[1,:].reshape(32,32)
    b=image[2,:].reshape(32,32)
    rgb_img = np.dstack((r, g, b))
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    rgb_img = (rgb_img * std + mean) * 255.0
    pil_img = Image.fromarray(rgb_img.astype('uint8'))
    return pil_img

if __name__ == "__main__":
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  # get training data and label
  data_batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
  data=np.empty((0, 3072), dtype=np.uint8)
  label=[]
  batch_name="/content/cifar-10-batches-py/"
  for data_batch in data_batches:
    data_batch_name=batch_name+data_batch
    my_dict=unpickle(data_batch_name)
    data_=my_dict[b"data"]
    label_=[int(element) for element in my_dict[b"labels"]]
    data=np.vstack((data, data_))
    label.extend(label_)

  # get testing data and label
  data_batch_name="/content/cifar-10-batches-py/test_batch"
  my_dict=unpickle(data_batch_name)
  testing_data=my_dict[b"data"]
  testing_label=[int(element) for element in my_dict[b"labels"]]

  # get class names
  result = unpickle("/content/cifar-10-batches-py/batches.meta")
  class_names=[byte_string.decode() for byte_string in result[b"label_names"]]

  mean = (0.4914, 0.4822, 0.4465)
  std = (0.2023, 0.1994, 0.2010)
  data = data.astype(np.float32) / 255.0
  data = data.reshape(-1, 3, 32, 32)
  for i in range(3):
      data[:, i, :, :] = (data[:, i, :, :] - mean[i]) / std[i]
  
  testing_data = testing_data.astype(np.float32) / 255.0
  testing_data = testing_data.reshape(-1, 3, 32, 32)
  for i in range(3):
      testing_data[:, i, :, :] = (testing_data[:, i, :, :] - mean[i]) / std[i]

  # change data type into torch.Tensor
  train_data = torch.from_numpy(data)
  test_data = torch.from_numpy(testing_data)
  train_label = torch.Tensor(label).long()
  test_label = torch.Tensor(testing_label).long()

  # datasets
  train_dataset = CustomDataset(train_data, train_label)
  test_dataset = CustomDataset(test_data, test_label)
  trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
  testloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=2)
  
  epoch_num=70
  loss_data, test_loss, accuracy_data, class_names = train(epoch_num, trainloader, testloader, class_names)    

  ## gradio
  # image processing
  def process_image(input_idx):

      # show image
      input_image, true_label = test_dataset[int(input_idx)]  # [3, 32, 32]
      show_image = getimg(input_image)              # [32, 32, 3]

      # predict label
      model = ResNet(ResidualBlock, [2, 2, 2, 2]) # Net()
      model.load_state_dict(torch.load("./cifar_net.pth"))
      model.eval()
      with torch.no_grad():
        outputs = model(input_image.float().unsqueeze(0)) 
        _, predicted = torch.max(outputs.data, 1)

      fig, ax1 = plt.subplots()

      ax1.plot(loss_data, color='#43BDE5')
      ax1.set_xlabel("Epoch")
      ax1.set_ylabel("Train Loss", color='#43BDE5')

      ax2 = ax1.twinx()
      ax2.plot(accuracy_data, color='#FF7043')
      ax2.set_ylabel("Accuracy", color='#FF7043')

      ax3 = ax1.twinx()
      ax3.spines['right'].set_position(('outward', 60))  
      ax3.plot(test_loss, color='magenta') 
      ax3.set_ylabel("Test Loss", color='magenta')

      tmp_filename = "/content/drive/MyDrive/summer2023/task1/temp_plot.png"
      plt.savefig(tmp_filename)
      chart = Image.open(tmp_filename)

      predicted_class = class_names[predicted.item()]
      true_class = class_names[true_label.item()]

      return chart, predicted_class, true_class, show_image
  
  # interface
  app_interface = gr.Interface(fn=process_image, 
              inputs=gr.Textbox(label="Input an index between 0-9999"), 
              outputs=[gr.outputs.Image(type="pil", label="Loss and Accuracy Plot"),
                  gr.Textbox(label="Predicted class"),
                  gr.Textbox(label="True class"),
                  gr.outputs.Image(type="pil", label="Image")], 
              title="Image Classification with Loss & Accuracy Visualization")

  app_interface.launch(share=True)


