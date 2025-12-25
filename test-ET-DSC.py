import csv
from tqdm  import tqdm
from model.parse_args_test import  parse_args
import scipy.io as scio

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param
from model.model_ET_DSC import Res_CBAM_block
from model.model_ET_DSC import ET_DSC
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score


class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args  = args
        self.ROC   = ROCMetric(1, args.ROC_thr)
        self.PD_FA = PD_FA(1,args.ROC_thr)
        self.mIoU  = mIoU(1)
        self.nIoU = nIoU()
        self.f1 = F1(1)
        self.pre = PRE(1)
        all_labels = []
        all_preds = []

        self.save_prefix = '_'.join([args.model, args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt=load_dataset(args.root, args.dataset,args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'ET-DSC':
            model       = ET_DSC(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Initialize evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Load trained model

        # checkpoint        = torch.load(r'D:\AAshare\ET-DSC\NUAA\mIoU__ET-DSC_NUAA-SIRST_epoch.pth.tar')
        checkpoint        = torch.load(r'D:\AAshare\ET-DSC\NUDT\mIoU__ET-DSC_NUDT-SIRST_epoch.pth.tar')
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(checkpoint['epoch'])
        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()
        with torch.no_grad():
            num = 0
            best_miou = 0
            best_back_miou = 0
            best_niou = 0
            best_back_niou = 0

            for i, (data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                    pred =preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)
                num += 1

                losses.    update(loss.item(), pred.size(0))
                self.ROC.  update(pred, labels)
                self.mIoU.update(pred, labels)
                self.nIoU.update(pred, labels)
                self.pre.update(pred, labels)
                self.PD_FA.update(pred, labels)
                self.f1.update(pred, labels)

                all_preds.append(pred.cpu().sigmoid().view(-1))
                all_labels.append(labels.cpu().view(-1))

                ture_positive_rate, false_positive_rate, recall, precision= self.ROC.get()
                _, mean_IOU,  = self.mIoU.get()
                mean_nIOU, back_niou = self.nIoU.get()
                f1 = self.f1.get()
                pre = self.pre.get()
            all_labels = torch.cat(all_labels).numpy().astype(int)
            all_preds = torch.cat(all_preds).numpy() 

            print("Precision:{:.2f}" .format(pre*100))
            print(f1)
            # print("F1-score(f1):{:.2f}" .format(f1*100))
            print("mean_IOU:{:.2f}" .format(mean_IOU*100))
            print("niou:{:.2f}".format(mean_nIOU*100))




def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





