import torch
from bert_utils import *
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import Adam
import torch.nn.functional as F
from os import path
import pickle

dataset_path = "./dataset/fnc-1"
file_train_instances = "train_stances.csv"
file_train_bodies = "train_bodies.csv"
file_test_instances = "competition_test_stances_unlabeled.csv"
file_test_bodies = "competition_test_bodies.csv"
file_predictions = 'predictions_test.csv'


def train(model, device, tokenizer):
    model.to(device)
    fnc_train = FNCData(path.join(dataset_path, file_train_instances), path.join(dataset_path, file_train_bodies))
    train_dataset = FNCDataset(fnc_train)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    optimizer = Adam(model.parameters(), lr=1e-5)

    itr = 1
    p_itr = 100
    epochs = 3
    total_loss = 0
    total_len = 0
    total_correct = 0
    train_loss = []
    train_accuracy = []

    model.train()
    for epoch in range(epochs):
        for headline, article, label in train_loader:
            optimizer.zero_grad()
            
            # encoding and zero padding
            encoded_list = []
            for t in zip(headline, article):
                encoded = tokenizer.encode(t[0], t[1], add_special_tokens=True, truncation=True)
                encoded_list.append(encoded[:min(512, len(encoded))])
            padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
            
            sample = torch.as_tensor(padded_list)
            sample, label = sample.to(device), label.to(device)
            labels = torch.as_tensor(label)
            outputs = model(sample, labels=labels)
            loss, logits = outputs.loss, outputs.logits

            pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
            correct = pred.eq(labels)
            total_correct += correct.sum().item()
            total_len += len(labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            if itr % p_itr == 0:
                train_loss.append(total_loss/p_itr)
                train_accuracy.append(total_correct/total_len)
                print(f'[Epoch {epoch+1}/{epochs}] Iteration {itr} -> Train Loss: {total_loss/p_itr:.4f}, Accuracy: {total_correct/total_len:.3f}')
                total_loss = 0
                total_len = 0
                total_correct = 0

            itr+=1

        torch.save(model.state_dict(), 'fnc_model.pth')

def eval(model, device, tokenizer):
    model.load_state_dict(torch.load('./fnc_model.pth'))
    model.to(device)

    fnc_test = FNCData(path.join(dataset_path, file_test_instances), path.join(dataset_path, file_test_bodies))
    test_dataset = FNCDataset(fnc_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    pred = []
    eval_count = 0
    for headline, article in test_loader:
        encoded_list = []
        for t in zip(headline, article):
            encoded = tokenizer.encode(t[0], t[1], add_special_tokens=True, truncation=True)
            encoded_list.append(encoded[:min(512, len(encoded))])
        padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
        sample = torch.as_tensor(padded_list)
        sample = sample.to(device)
        outputs = model(sample)
        _, logits = outputs.loss, outputs.logits

        pred += torch.argmax(F.softmax(logits, dim=1), dim=1).tolist()
        eval_count += test_loader.batch_size
        print(f"{eval_count}/{len(test_loader)}")
    
    with open('pred.pickle', 'wb') as f:
        pickle.dump(pred, f)
    #save_predictions(fnc_test, pred, path.join('./bert', file_predictions))
    


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

    train(model, device, tokenizer)
    eval(model, device, tokenizer)
    fnc_test = FNCData(path.join(dataset_path, file_test_instances), path.join(dataset_path, file_test_bodies))
    save_predictions(fnc_test, 'pred.pickle', path.join('./bert', file_predictions))

    
    

    

if __name__ == "__main__":
    main()