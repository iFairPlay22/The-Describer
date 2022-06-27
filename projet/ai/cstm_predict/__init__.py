import torch
import cstm_load as cstm_load
import cstm_model as cstm_model

def predict(img_tensor, vocabulary : cstm_load.Vocab, fullModel : cstm_model.FullModel):
    """ Predict the caption for an image """
    
    # We don't train the models
    fullModel.evalMode()

    res = dict()

    with torch.no_grad():

        # Generate an caption from the image
        n_sampled_indices = fullModel.sample(img_tensor)

        # Convert word_ids to words
        n_predicted_caption = []
        for sampled_indices in n_sampled_indices:
            predicted_caption = vocabulary.translate(sampled_indices.cpu().numpy())
            n_predicted_caption.append(predicted_caption)

        res["indices"] = n_sampled_indices
        res["words"] = " ".join(n_predicted_caption)

    # We train the models
    fullModel.trainMode()

    return res
