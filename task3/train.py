import torch 
from torch import nn
from datapreprocess import *
from model import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    writer = SummaryWriter('/content/drive/MyDrive/summer2023/task3/log')

    batch_size = 64
    train_iter, test_iter, vocab_size, train_len, test_len = dataPreprocess(batch_size)

    print("vocab size: ", vocab_size)

    embedding_dim = 100
    hidden_dim = 128
    num_layers = 1
    bidirectional = True
    dropout = 0.5
    rnn_type = 'lstm'
    net = TextClassifier(vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, dropout, rnn_type).to(device)

    
    learning_rate = 0.001
    epochs = 20
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    best_acc = 0.0

    for epoch in range(epochs):
        training_loss = 0.0
        training_acc = 0.0 
        net.train()
        for itdata in train_iter:
            inputs, labels = itdata.text, itdata.label
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            training_acc += torch.sum(torch.round(outputs) == labels)
        print(f'[Epoch {epoch + 1}] training loss: {training_loss / train_len :.4f}, training accuracy: {100 * training_acc / train_len :.4f}')
        writer.add_scalar('train_loss', training_loss / train_len, epoch+1)
        writer.add_scalar('train_acc', 100 * training_acc / train_len, epoch+1)
        
        # test accuracy per epoch
        correct = 0
        testing_loss = 0.0
        with torch.no_grad():
            for testdata in test_iter:
                inputs, labels = testdata.text, testdata.label
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                # _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                correct += torch.sum(torch.round(outputs) == labels)
                testing_loss += loss.item()
        accuracy = 100 * correct / test_len
        print(f'[Epoch {epoch + 1}] testing loss : {testing_loss / test_len :.4f}, testing accuracy : {accuracy :.4f}')
        writer.add_scalar('test_loss', testing_loss / test_len, epoch+1)
        writer.add_scalar('test_acc', accuracy, epoch+1)

        if accuracy > best_acc:
            best_acc= accuracy
            PATH = '/content/drive/MyDrive/summer2023/task3/best_model.pth'
            torch.save(net.state_dict(), PATH)

    print('Finished Training')
    writer.close()

    # prepare to count predictions for each class
    correct_pred = [0,0]
    total_pred = [0,0]

    with torch.no_grad():
      for testdata in test_iter:
          inputs, labels = testdata.text, testdata.label
          inputs, labels = inputs.to(device), labels.to(device)

          outputs = net(inputs).squeeze(1)
          loss = criterion(outputs, labels)
          
          for label, output in zip(labels, outputs):
              if int(torch.round(output)) == int(label.item()):
                  correct_pred[int(round(label.item()))] += 1
              total_pred[int(round(label.item()))] += 1

    # print accuracy for each 
    if total_pred[0] != 0:
        accuracy = 100 * float(correct_pred[0]) / total_pred[0]
    print(f'Accuracy for positive is {accuracy:.3f} %')
    if total_pred[1] != 0:
        accuracy = 100 * float(correct_pred[1]) / total_pred[1]
    print(f'Accuracy for negative is {accuracy:.3f} %')

            

    
