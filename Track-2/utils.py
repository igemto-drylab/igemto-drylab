import torch
def savemodel(model, modelname, iter=0, directory="historical_weights"):
    torch.save(model.state_dict(), directory + "/" + modelname + ".pt")