import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy, recall, precision, f1_score
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dgl.dataloading import GraphDataLoader
from datasets import load_dataset, load_from_disk

from src.config import hparams
from src.information_retrieval.data_processed.parsing2graph import P2G_double
from src.information_retrieval.models.embedding import BinaryClass_Double

class MyModel(pl.LightningModule):
    def __init__(self, lr, dropout, train_dataset, val_dataset, test_dataset, num_layers):
        super().__init__()
        self.lr = lr
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = hparams.batch_size
        self.model = BinaryClass_Double(dropout, num_layers)
        self.focal_loss = focal_loss()

    def forward(self, input_ids, token_type_ids, attention_mask, graph, g_num_nodes, mask):
        return self.model(input_ids, token_type_ids, attention_mask, graph, g_num_nodes, mask)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        train_loader = GraphDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        return train_loader

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, graph, g_num_nodes, label = batch
        mask = graph.ndata["mask"]
        logit = self(input_ids, token_type_ids, attention_mask, graph, g_num_nodes, mask)
        loss = self.focal_loss(logit, label)
        output = torch.argmax(logit, dim=1)
        acc = accuracy(output, label, task="binary")
        f1 = f1_score(output, label, task="binary")
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size)
        self.log("train_acc", acc, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log("train_f1", f1, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        return loss

    def val_dataloader(self):
        val_loader = GraphDataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return val_loader

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            input_ids, token_type_ids, attention_mask, graph, g_num_nodes, label = batch
            mask = graph.ndata["mask"]
            logit = self(input_ids, token_type_ids, attention_mask, graph, g_num_nodes, mask)
            loss = self.focal_loss(logit, label)
            output = torch.argmax(logit, dim=1)
            acc = accuracy(output, label, task="binary")
            f1 = f1_score(output, label, task="binary")
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True,batch_size=self.batch_size)
        self.log("val_f1", f1, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)

    def test_dataloader(self):
        test_loader = GraphDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return test_loader

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            input_ids, token_type_ids, attention_mask, graph, g_num_nodes, label = batch
            mask = graph.ndata["mask"]
            logit = self(input_ids, token_type_ids, attention_mask, graph, g_num_nodes, mask)
            loss = self.focal_loss(logit, label)
            output = torch.argmax(logit, dim=1)
            acc = accuracy(output, label, task="binary")
            f1 = f1_score(output, label, task="binary")
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True,batch_size=self.batch_size)
        self.log("test_f1", f1, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)


class focal_loss(nn.Module):
    def __init__(self, alpha=0.1, gamma=2, num_classes=2, size_average=False):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            # print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            # print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def train(model_name, lr, drop, norm_way, train_dataset, val_dataset, test_dataset, num_layers):
    log_name = f'{model_name}_lr{lr}_drop{drop}_{norm_way}_{num_layers}num_layers'
    logger = TensorBoardLogger(save_dir=hparams.save_log_path, name=log_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # 监控对象
        dirpath=hparams.save_model_path,  # 保存模型的路径
        filename=f'{model_name}' + '_{epoch:02d}_{val_loss:.2f}',  # 最优模型的名称
        save_top_k=1,  # 只保存最好的那个
        mode='min'  # 当监控对象指标最小时
    )

    trainer = Trainer(
        logger=logger,
        max_epochs=hparams.epochs,
        log_every_n_steps=8,
        accelerator='gpu',
        devices="auto",
        fast_dev_run=False,
        precision=16,
        callbacks=[checkpoint_callback]
    )
    model = MyModel(lr, drop, norm_way, train_dataset, val_dataset, test_dataset, num_layers)
    trainer.fit(model)


def test(PATH):
    # 加载之前训练好的最优模型参数
    model = MyModel.load_from_checkpoint(checkpoint_path=PATH)
    trainer = Trainer(fast_dev_run=False)
    result = trainer.test(model)
    print(result)


if __name__ == "__main__":
    max_len = 128
    model_name = "GCN_double_bertbase"
    dataset = load_from_disk(os.path.join(hparams.data_path, "BQ"))
    train_dataset = P2G_double(dataset["train"], hparams.pretrained_model, hparams.LTP, max_len)
    val_dataset = P2G_double(dataset["validation"], hparams.pretrained_model, hparams.LTP, max_len)
    test_dataset = P2G_double(dataset["test"], hparams.pretrained_model, hparams.LTP, max_len)
    seed_everything(hparams.seed)

    extra_config = {
        "lr": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5],
        "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
        "norm_way": ["BatchNorm", "LayerNorm"],
        "num_layers": [2, 3, 4, 5, 6]
    }

    for lr in extra_config["lr"]:
        for drop in extra_config["dropout"]:
            for norm_way in extra_config["norm_way"]:
                for num_layers in extra_config["num_layers"]:
                    train(model_name, lr, drop, norm_way, train_dataset, val_dataset, test_dataset, num_layers)
