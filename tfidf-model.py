from torch._C import dtype
from torch.utils import data
from tfidf_utils import *
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
from os import path
import matplotlib.pyplot as plt

dataset_path = "./dataset/fnc-1"
file_train_instances = "train_stances.csv"
file_train_bodies = "train_bodies.csv"
file_test_instances = "competition_test_stances_unlabeled.csv"
file_test_bodies = "competition_test_bodies.csv"
file_predictions = 'predictions_test.csv'

class MLP(nn.Module):
    def __init__(self, input_dim, hid_dim, num_class, dropout):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(input_dim, hid_dim, dtype=torch.double)
        self.linear2 = nn.Linear(hid_dim, num_class, dtype=torch.double)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        output = self.dropout(self.linear2(x))

        return self.log_softmax(output)
        
def train(model, device, train_loader, epochs, optimizer, criterion, clip):
    model.train()
    total_acc = 0
    total_loss = 0
    i = 1
    train_loss = []
    train_accuracy = []
    for epoch in range(epochs):
        for train_set, train_stances in train_loader:
            sample = torch.as_tensor(train_set)
            sample, train_stances = sample.to(device), train_stances.to(device)
            labels = torch.as_tensor(train_stances)

            
            outputs = model(sample)
            pred = torch.argmax(outputs, dim=1)
            accuracy = pred.eq(labels).sum()/train_loader.batch_size*100

            total_acc += accuracy
            train_accuracy.append(total_acc/i)

            loss = criterion(outputs, labels)

            total_loss += loss
            train_loss.append(total_loss/i)
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            
            
            if i % 50 == 0:
                print(f'Epoch: {epoch+1}\t Train Step: {i:3d}\t Loss: {loss:.3f}\t Accuracy: {accuracy:.3f}%')
            i += 1
        print(f'Epoch: {epoch+1} finished')
    

    
    with torch.no_grad():
        plt.plot(range(len(train_loss)), train_loss)  #marker='o'
        plt.xlabel('Train Step')
        plt.ylabel('Train Loss')
        plt.show()
        plt.savefig(f'./train_loss_result.png')
        plt.clf()
        plt.plot(range(len(train_accuracy)), train_accuracy)  #marker='o'
        plt.xlabel('Train Step')
        plt.ylabel('Train Accuracy')
        plt.show()
        plt.savefig(f'./train_accuracy_result.png')
    

def evaluate(model, device, test_loader):
    model.eval()
    pred = []
    for test_set, _ in test_loader:
        sample = torch.as_tensor(test_set)
        sample = sample.to(device)
        output = model(sample)
        pred += torch.argmax(output, dim=1).tolist()
    
    return pred




def main():
    fnc_train_raw = FNCData(
        path.join(dataset_path, file_train_instances), path.join(dataset_path, file_train_bodies)
    )
    fnc_test_raw = FNCData(
       path.join(dataset_path, file_test_instances), path.join(dataset_path, file_test_bodies) 
    )

    x_train, y_train, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = pipeline_train(
        fnc_train_raw, fnc_test_raw, lim_unigram=5000
    )

    x_test = pipeline_test(fnc_test_raw, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)


    train_loader = DataLoader(FNCDataset(x_train, y_train), batch_size=500, shuffle=True)
    test_loader = DataLoader(FNCDataset(x_test), batch_size=500, shuffle=False)
    

    input_dim = len(x_train[0])
    hid_dim = 100
    output_dim = 4
    dropout = 0.6
    clip_ratio = 5
    epochs = 90

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_dim, hid_dim, output_dim, dropout).to(device)
    optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().to(device)

    print("train start")
    train(model, device, train_loader, epochs, optimizer, criterion, clip_ratio)
    
    torch.save(model.state_dict(), 'fnc_tfidf_model.pth')
    model.load_state_dict(torch.load('./fnc_tfidf_model.pth'))
    eval_pred = evaluate(model, device, test_loader)
    save_predictions(eval_pred, fnc_test_raw, path.join(dataset_path, file_predictions))


if __name__ == "__main__":
    main()
