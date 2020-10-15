import time
import torch
import sklearn.metrics as sklm

class TrainUtils():
    def __init__(self, model, train_loader, test_loader, optimizer, criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, epoch, is_multi = True):
        self.model.train()
        running_loss = 0.0
        predictions_list = []
        labels_list = []
        time_start = time.time()
        for i, data in enumerate(self.train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(inputs.float())
            labels.reshape(inputs.shape[0],-1)

            #depending on if it's multi-class prediction, save predictions accordingly
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

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                time_end = time.time()
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                print("time ",(time_end - time_start))
                print(sklm.classification_report(labels_list, predictions_list))
                running_loss = 0.0


    def evaluate(self, is_multi = True):
        self.model.eval()
        running_loss = 0.0
        predictions_list = []
        labels_list = []
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # forward + backward + optimize
                outputs = self.model(inputs.float())
                labels.reshape(inputs.shape[0],-1)

                #depending on if it's multi-class prediction, save predictions accordingly
                if is_multi:
                    loss = self.criterion(outputs, labels)
                    prediction = outputs.detach()
                    predictions_list.extend(torch.argmax(prediction,1).float().cpu().numpy())
                else:
                    loss = self.criterion(outputs, labels.float())
                    prediction = outputs.view(-1).detach()
                    predictions_list.extend((prediction>0.5))

                labels_list.extend(labels.float().view(-1).cpu().numpy())

                # print statistics
                running_loss += loss.item()
            print('=========================================================')
            print('\t test loss: %.3f' %
                  (running_loss / 200))
            print("test classification report:")
            print(sklm.classification_report(labels_list, predictions_list))
            print('=========================================================')

    def saveModel(self, model_title):
        folder = "trained_models/"
        path = folder + model_title + ".pt"
        torch.save(self.model.state_dict(), path)

    def loadModel(self, model_title):
        folder = "trained_models/"
        model = folder + model_title + ".pt"
        model_weights = torch.load(model)
        self.model.load_state_dict(model_weights)

    def getModel(self):
        return self.model
