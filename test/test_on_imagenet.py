import models


model = models.get_pretrained_squeezenet("../model_files/sqn_model.json", "../model_files/sqn_weights.json")
print("SqueezeNet model is loaded.")