import argparse
import torch
import cvpl_tools.nnunet.triplanar as triplanar

DEVICE = torch.device("cuda:0")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and predict using MONAI UNet")

    subparsers = parser.add_subparsers(dest="command", help="Commands: train, predict_single, predict_folder")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--train_im", type=str, required=True, help="Path to the training images")
    train_parser.add_argument("--train_seg", type=str, required=True, help="Path to the training labels")
    train_parser.add_argument("--nepoch", type=int, default=500, help="Number of training epochs")
    train_parser.add_argument("--stack_channels", type=int, default=0,
                              help="Jump interval size in the third axis for 3 image channels")
    train_parser.add_argument("--triplanar", action="store_true")
    train_parser.add_argument("--cache_dir", type=str, required=True, help="Path for cache storage")
    train_parser.add_argument("--dataset_id", type=int, default=1, help="Unique for each dataset")
    train_parser.add_argument("--fold", type=str, default='0', help="Fold to train")
    train_parser.add_argument("--max_threshold", type=float, help="maximum threshold when preprocessing images")

    # Predict single image command
    predict_parser = subparsers.add_parser("predict", help="Predict a single image or multiple images")
    predict_parser.add_argument("--dataset_id", type=int, default=1, help="Unique for each dataset")
    predict_parser.add_argument("--fold", type=str, default='0', help="Fold to train")
    predict_parser.add_argument("--test_im", type=str, default="", help="Path to the testing images")
    predict_parser.add_argument("--test_seg", type=str, default="", help="Path to the testing labels")
    predict_parser.add_argument("--output", type=str, default="", help="Optional, the output tiff file to write")
    predict_parser.add_argument("--cache_dir", type=str, required=True,
                                help="Path for model, input, output and cache storage")
    predict_parser.add_argument("--triplanar", action="store_true")
    predict_parser.add_argument("--penalize_edge", action="store_true")
    predict_parser.add_argument('--weights', nargs='+', type=float, default=None)

    # Compute dice on volume
    dice_parser = subparsers.add_parser("dice", help="Compute dice between volumes")
    dice_parser.add_argument("--vol1", type=str, required=True,
                             help="Path to the first volume (.pt or .npy)")
    dice_parser.add_argument("--vol2", type=str, required=True,
                             help="Path to the second volume (.pt or .npy)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "train":
        # Assuming train_args dictionary contains all required arguments
        train_args = {
            "cache_url": args.cache_dir,
            "train_im": args.train_im,
            "train_seg": args.train_seg,
            "nepoch": args.nepoch,
            "stack_channels": args.stack_channels,
            "triplanar": args.triplanar,
            "dataset_id": args.dataset_id,
            "fold": args.fold,
            "max_threshold": args.max_threshold,
        }
        model = triplanar.train_triplanar(train_args)

    elif args.command == "predict":
        pred_args = {
            "cache_url": args.cache_dir,
            "test_im": args.test_im,
            "test_seg": args.test_seg,
            "output": args.output,
            "dataset_id": args.dataset_id,
            "fold": args.fold,
            "triplanar": args.triplanar,
            "penalize_edge": args.penalize_edge,
            "weights": args.weights,
        }
        triplanar.predict_triplanar(pred_args)

    elif args.command == "dice":
        vol1 = triplanar.load_volume_np_or_torch(args.vol1, convert_to_bin=True)
        vol2 = triplanar.load_volume_np_or_torch(args.vol2, convert_to_bin=True)
        dice = triplanar.dice_on_volume_pair(vol1, vol2)
        print(f'DICE of two volumes from {args.vol1} and {args.vol2} with shape={vol1.shape}:\ndice={dice:.4f}')
