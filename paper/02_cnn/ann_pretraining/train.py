import pytorch_lightning as pl
from nmnist import NMNISTFrames, NMNISTRaster
from cnn import CNN
from snn import SNN
from argparse import ArgumentParser


if __name__ == '__main__':
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--model', default='cnn', type=str)
    parser.add_argument('--n_time_bins', default=30, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if args.model == 'cnn':
        dataset = NMNISTFrames('data', batch_size=args.batch_size, precision=args.precision, num_workers=args.num_workers)
        model = CNN()
    
    elif args.model == 'snn':
        print('SNN')
        dataset = NMNISTRaster('data', batch_size=args.batch_size, n_time_bins=args.n_time_bins, precision=args.precision, num_workers=args.num_workers)
        model = SNN(batch_size=args.batch_size)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="valid_acc",
        filename=args.model + "-{step}-{epoch:02d}-{valid_loss:.4f}-{valid_acc:.4f}",
        save_top_k=1,
        mode="max",
    )
    
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], logger=None)
    trainer.fit(model, dataset)
