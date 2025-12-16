# Basic module
import csv

from tqdm             import tqdm
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
# from model.feature_map import *

# Model
from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score


class Trainer(object):
    def __init__(self, args):
        # TensorBoard
        from torch.utils.tensorboard import SummaryWriter
        # log_dir = os.path.join(os.getcwd(), "logs" + os.sep + "now" + os.sep)
        log_dir = os.path.join(os.getcwd(), "logs", "now")
        # 如果目录已存在，则删除
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)  # 递归删除目录及其内容

        # 重新创建目录
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        # Initial
        self.args  = args
        self.ROC   = ROCMetric(1, args.ROC_thr)
        self.PD_FA = PD_FA(1,args.ROC_thr)
        self.mIoU  = mIoU(1)
        self.nIoU = nIoU()
        self.dsc = DSC(1)
        self.se = SE(1)
        all_labels = []
        all_preds = []
        self.pre = PRE(1)
        self.pd = Pd(1)
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
        if args.model   == 'DNANet':
            model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Initialize evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Load trained model
        # checkpoint        = torch.load('result/' + args.model_dir)
        # checkpoint        = torch.load('logs/now/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar')
        # checkpoint        = torch.load('result/NUDT-SIRST_DNANet_19_02_2025_10_33_23_wDS/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar')
        # checkpoint        = torch.load('result/34_4/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar')
        # checkpoint        = torch.load('result/NUDT-SIRST_DNANet_20_05_2025_23_19_03_wDS/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar')
        # checkpoint        = torch.load('result/NUDT-SIRST_DNANet_20_05_2025_23_19_03_wDS/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar')
        checkpoint        = torch.load('result/先收起来测试时再放出来/NUAA/NUAA-DNA/mIoU__DNANet_NUAA-SIRST_epoch.pth.tar')
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
                # feature_2_tb(labels, writer, i, "GT")
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    # preds = self.model(data, writer, i)  # 获取模型的预测结果，可能包含多个输出
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
                # self.mIoU. update(pred, labels)
                self.mIoU.update(pred, labels)
                self.nIoU.update(pred, labels)
                self.pd.update(pred, labels)
                self.pre.update(pred, labels)
                self.se.update(pred, labels)
                self.PD_FA.update(pred, labels)
                self.dsc.update(pred, labels)

                all_preds.append(pred.cpu().sigmoid().view(-1))
                # all_preds.append(pred.cpu().sigmoid().flatten())
                # all_labels.append(labels.cpu().flatten())
                all_labels.append(labels.cpu().view(-1))

                ture_positive_rate, false_positive_rate, recall, precision= self.ROC.get()
                _, mean_IOU, _, back_iou = self.mIoU.get()
                mean_nIOU, back_niou = self.nIoU.get()
                FA, PD = self.PD_FA.get(len(val_img_ids))
                dsc = self.dsc.get()
                se = self.se.get()
                # pd = self.pd.get()
                pre = self.pre.get()

            # all_preds = torch.cat(all_preds).numpy()
            # all_labels = torch.cat(all_labels).numpy()
            all_labels = torch.cat(all_labels).numpy().astype(int)
            all_preds = torch.cat(all_preds).numpy()  # preds 是 float (0~1), 保留原样

            # all_labels = all_labels[:100]
            # all_preds = all_preds[:100]
            fpr, tpr, _ = roc_curve(all_labels, all_preds)
            # 假设 fpr 和 tpr 是 numpy 数组或列表
            # with open('FES_ROC.txt', 'w') as f:
            #     f.write(" ".join(map(str, fpr)) + "\n")  # 第一行写 fpr
            #     f.write(" ".join(map(str, tpr)) + "\n")  # 第二行写 tpr

            mask = fpr <= 1e-3
            fpr_filtered = fpr[mask]
            tpr_filtered = tpr[mask]
            with open('FES_ROC.txt', 'w') as f:
                f.write(" ".join(map(str, fpr_filtered)) + "\n")  # 第一行写 fpr
                f.write(" ".join(map(str, tpr_filtered)) + "\n")  # 第二行写 tpr
            precision, reca, throd = precision_recall_curve(all_labels, all_preds)
            ap = average_precision_score(all_labels, all_preds)




            # plt.figure(figsize=(6, 6))
            # plt.plot(reca, precision, color='blue', lw=2, label='PR Curve')
            # plt.xlabel('Recall')
            # plt.ylabel('Precision')
            # plt.title('Precision-Recall Curve')
            # plt.legend()
            # plt.grid(True)
            # plt.show()


            # 绘图
            plt.figure(figsize=(6, 6))
            plt.plot(fpr_filtered, tpr_filtered, color='blue', lw=2, label='ROC Curve')
            plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')
            plt.xlabel('False Positive Rate')
            plt.xlim(0, 1e-3)  # 强制横坐标范围
            plt.ylabel('True Positive Rate (Recall)')
            plt.title('ROC Curve')

            # 可选：计算 AUC
            roc_auc = auc(fpr, tpr)
            plt.legend(loc='lower right')
            plt.text(0.6, 0.2, f'AUC = {roc_auc:.4f}', fontsize=12)

            plt.grid(True)
            plt.tight_layout()
            plt.savefig(dataset_dir + '/' + 'value_result/' + args.st_model + '_ROC_Curve.png')
            plt.show()

            print(f"Average Precision: {ap*100:.2f}")
            # 定义要保存的文件路径和名称，这里根据原代码保存路径逻辑进行调整，可根据实际需求更改
            # print("pd:{:.4f}()" .format(pd))
            print("Precision:{:.2f}" .format(pre*100))
            print("Recall(pd):{:.2f}" .format(se*100))
            print("F1-score(DSC):{:.2f}" .format(dsc*100))
            print("mean_IOU:{:.2f}" .format(mean_IOU*100))
            # print("back_iou:" , back_iou)
            print("niou:{:.2f}".format(mean_nIOU*100))
            print("auc:{:.2f}".format(roc_auc*100))
            # print("back_niou", back_niou)

            # miou2 = (mean_IOU + back_iou) / 2
            # niou2 = (mean_nIOU + back_niou) / 2
            # print("miou2", miou2)
            # print("niou2", niou2)

            # for fa_value, pd_value in zip(FA, PD):
            #     pass
                # writer.writerow([fa_value, pd_value])  # 逐行写入 FA 和 PD 的对应值
            # print("fa:".format(FA.min()))
            # print("pd:".format(PD.max()))



            # print("fa", FA.max())
            # pd  = PD.max()
            # print("pd:{:.2f}".format(pd*100))
            print("fa", FA)
            print("pd", PD)



            # save_file_path = dataset_dir + '/' + 'value_result' + '/' + args.st_model + '_PD_FA_' + str(
            #     255) + '.txt'
            #
            # # 使用 'w' 模式打开文件（写入模式，如果文件已存在会覆盖原有内容，若想追加内容可使用 'a' 模式）
            # with open(save_file_path, 'w', newline='') as f:
            #     writer = csv.writer(f, delimiter='\t')  # 使用制表符 '\t' 作为分隔符，可根据喜好更换为其他如逗号等
            #     writer.writerow(['False Alarm (FA)', 'Probability of Detection (PD)'])  # 写入表头
            #     for fa_value, pd_value in zip(FA, PD):
            #         writer.writerow([fa_value, pd_value])  # 逐行写入 FA 和 PD 的对应值
            # #
            # # scio.savemat(dataset_dir + '/' +  'value_result'+ '/' +args.st_model  + '_PD_FA_' + str(255),
            # #              {'number_record1': FA, 'number_record2': PD})
            #
            # save_result_for_test(dataset_dir, args.st_model,args.epochs, mean_IOU, recall, precision, FA, PD)

    def plot_auc_curve(self, save_path='ROC_curve.png'):
        fp_rates, tp_rates, _, _ = self.get()  # 获取全局FPR和TPR

        # 计算AUC（梯形法数值积分）
        auc_score = np.trapz(tp_rates, fp_rates)

        # 绘制曲线
        plt.figure()
        plt.plot(fp_rates, tp_rates, color='darkorange',
                 label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')  # 对角线
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





