import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(embed_size, vocab_size)
        self.vocab_size = vocab_size
    
    def forward(self, features, captions):
        seqlen = captions.shape[1]
        batch_size = features.shape[0]
        outputs = []
        hidden = None
        # features (batch_size, embed_size)
        for i in range(seqlen):
            if i == 0:
                out, hidden = self.lstm(features.view(batch_size, 1, -1), hidden)
                size = out.shape[-1] // 2
                out = out[:, :, size:] + out[:, :, :size]
                out = torch.softmax(self.linear(out), dim=1)
                outputs.append(out)
            else:
                # Embedding
                out = outputs[-1]
                out = torch.argmax(out, dim=2)
                out = self.embedding(out)
                # LSTM
                out, hidden = self.lstm(out.view(batch_size, 1, -1), hidden)
                size = out.shape[-1] // 2
                out = out[:, :, size:] + out[:, :, :size]
                # Linear
                out = torch.softmax(self.linear(out), dim=1)
                outputs.append(out)
        outputs = [x.view(batch_size, 1, -1) for x in outputs]
        return torch.cat(outputs, dim=1)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass