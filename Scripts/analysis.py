# %%
import pandas as pd

# %%
cnn_df = pd.read_csv('cnn-output.csv')

cnn_filtered_df = cnn_df.filter(
        [
            "Name",
            "epochs",
            "hidden_layer_size",
            "learning-rate",
            "optimizer",
            "epoch/accuracy",
            "epoch/val_accuracy",
            "epoch/loss",
            "epoch/val_loss",
        ]
    ).copy()

cnn_top_run = cnn_filtered_df.nlargest(1, 'epoch/val_accuracy')

# %%
aug_cnn_df = pd.read_csv('cnn-aug-output.csv')

aug_cnn_filtered_df = aug_cnn_df.filter(
        [
            "Name",
            "epochs",
            "hidden_layer_size",
            "learning-rate",
            "optimizer",
            "epoch/accuracy",
            "epoch/val_accuracy",
            "epoch/loss",
            "epoch/val_loss",
        ]
    ).copy()

aug_cnn_top_run = aug_cnn_filtered_df.nlargest(1, 'epoch/val_accuracy')

# %%
ft_df = pd.read_csv('ft-output.csv')

ft_filtered_df = ft_df.filter(
        [
            "Name",
            "epochs",
            "learning-rate",
            "optimizer",
            "epoch/accuracy",
            "epoch/val_accuracy",
            "epoch/loss",
            "epoch/val_loss",
        ]
    ).copy()

ft_top_run = ft_filtered_df.nlargest(1, 'epoch/val_accuracy')

# %%
ft_aug_df = pd.read_csv('ft-aug-output.csv')

ft_aug_filtered_df = ft_aug_df.filter(
        [
            "Name",
            "epochs",
            "learning-rate",
            "optimizer",
            "epoch/accuracy",
            "epoch/val_accuracy",
            "epoch/loss",
            "epoch/val_loss",
        ]
    ).copy()

ft_aug_top_run = ft_aug_filtered_df.nlargest(1, 'epoch/val_accuracy')

# %%
top_models = pd.concat([cnn_top_run, aug_cnn_top_run, ft_top_run, ft_aug_top_run])

# %%
print(top_models.head())


