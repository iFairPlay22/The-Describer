import torch

def predict(img_tensor, vocabulary, fullModel):
    
    # We don't train the models
    fullModel.eval()

    res = dict()

    with torch.no_grad():

        # Generate an caption from the image
        n_sampled_indices = fullModel.sample(img_tensor)

        # Convert word_ids to words
        n_predicted_caption = []
        for sampled_indices in n_sampled_indices:
            predicted_caption = vocabulary.translate(sampled_indices.cpu().numpy())
            n_predicted_caption.append(''.join(predicted_caption))

        res["indices"] = n_sampled_indices
        res["words"] = n_predicted_caption

    # We train the models
    fullModel.train()

    return res
