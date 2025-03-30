from rfdetr import RFDETRBase
import polars as pl

model = RFDETRBase(device="cuda")
history = []


def callback2(data):
    history.append(data)


model.callbacks["on_fit_epoch_end"].append(callback2)

model.train(
    dataset_dir="./data",
    epochs=20,
    batch_size=16,
    lr=1e-4,
    num_workers=2,
    checkpoint_interval=4,
    output_dir="output",
)

pl.DataFrame(history).write_csv("history.csv")
