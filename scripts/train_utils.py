import time
import torch
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

class TrainUtils():
    def __init__(self, model, train_loader, test_loader, optimizer, criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.best_model = None
        self.best_val_F1 = 0

    def train(self, epoch, is_multi = True):
        self.model.train()
        running_loss = 0.0
        losses = []
        predictions_list = []
        labels_list = []
        time_start = time.time()
        for i, data in enumerate(self.train_loader, 0):
            # Get the inputs; data is a list of [inputs, labels]            
            inputs, labels = data
            inputs = inputs.to(self.device)
            inputs = inputs.float()

            labels = labels.to(self.device)
            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = self.model(inputs)
            labels.reshape(inputs.shape[0],-1)

            # Depending on if it's multi-class prediction, save predictions accordingly
            if is_multi:
                loss = self.criterion(outputs, labels)
                prediction = outputs.detach()
                predictions_list.extend(torch.argmax(prediction,1).float().cpu().numpy())
            else:
                loss = self.criterion(outputs, labels.float())
                prediction = outputs.view(-1).detach()
                predictions_list.extend((prediction>0.5))

            labels_list.extend(labels.float().view(-1).cpu().numpy())
            loss.backward()
            self.optimizer.step()

            # Print statistics
            running_loss += loss.item()
            losses.append(loss.item())
            if i % 200 == 199:    # print every 200 mini-batches
                time_end = time.time()
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                print("time ",(time_end - time_start))
                print(sklm.classification_report(labels_list, predictions_list))
                running_loss = 0.0
            epoch_f1=sklm.classification_report(labels_list, predictions_list,output_dict=True)['weighted avg']['f1-score']
        return np.mean(losses), epoch_f1


    def evaluate(self, is_multi = True,model_title=None):
        self.model.eval()
        running_loss = 0.0
        losses = []
        f1s = []
        predictions_list = []
        labels_list = []
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                # Get the inputs; data is a list of [inputs, labels]
                
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # Forward + backward + optimize
                outputs = self.model(inputs.float())
                labels.reshape(inputs.shape[0],-1)

                # Depending on if it's multi-class prediction, save predictions accordingly
                if is_multi:
                    loss = self.criterion(outputs, labels)
                    prediction = outputs.detach()
                    predictions_list.extend(torch.argmax(prediction,1).float().cpu().numpy())
                else:
                    loss = self.criterion(outputs, labels.float())
                    prediction = outputs.view(-1).detach()
                    predictions_list.extend((prediction>0.5))

                labels_list.extend(labels.float().view(-1).cpu().numpy())

                # Print statistics
                running_loss += loss.item()
                losses.append(loss.item())
            print('=========================================================')
            print('\t test loss: %.3f' %
                  (running_loss / 200))
            print("test classification report:")
            print(sklm.classification_report(labels_list, predictions_list))
            print('=========================================================')
            scores = sklm.classification_report(labels_list, predictions_list,output_dict=True)
            epoch_f1=scores['weighted avg']['f1-score']
            
            # Save and keep track of best performing model parameters
            if(epoch_f1>self.best_val_F1 and model_title!=None):
                folder = "trained_models/"+model_title+"/"
                if not os.path.exists(folder):
                    os.makedirs(folder)
                torch.save(self.model.state_dict(), folder+"model_best.pt")
                self.best_val_F1 = epoch_f1
                
        return np.mean(losses),epoch_f1
    
    def saveModel(self, model_title,train_losses=None,validation_losses=None,train_F1s=None,validation_F1s=None,config=None):
        folder = "trained_models/"+model_title+"/"
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Save model checkpoint
        torch.save(self.model.state_dict(), folder+"checkpoint.pt")
        
        # Save data and graphs for losses and F1 scores if available
        if(train_losses!=None and validation_losses!=None):
            lossGraph = plt.figure(dpi=200)
            plt.plot(train_losses,label="Train")
            plt.plot(validation_losses,label="Test")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(folder+"/losses.png")
            plt.close()
            pickle.dump(train_losses, open(folder+"train_losses.p", "wb"))
            pickle.dump(validation_losses, open(folder+"validation_losses.p", "wb"))
        if(train_F1s!=None and validation_F1s!=None):
            lossGraph = plt.figure(dpi=200)
            plt.plot(train_F1s,label="Train")
            plt.plot(validation_F1s,label="Test")
            plt.xlabel("Epoch")
            plt.ylabel("F1 Score")
            plt.legend()
            plt.savefig(folder+"/f1s.png")
            plt.close()
            pickle.dump(train_F1s, open(folder+"train_F1s.p", "wb"))
            pickle.dump(validation_F1s, open(folder+"validation_F1s.p", "wb"))
            
        # Save used config for easier reproduction when loading model
        if(config!=None):
            pickle.dump(config,open(folder+"config.p","wb"))

    def loadModel(self, model_title):
        folder = "trained_models/"
        model = folder + model_title + ".pt"
        model_weights = torch.load(model)
        self.model.load_state_dict(model_weights)

    def getModel(self):
        return self.model
