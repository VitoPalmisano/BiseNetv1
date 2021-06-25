import argparse
from dataset.transform import *
from dataset.voc import VOCSegmentation as VOC
from torch.utils.data import DataLoader
import os
from model.build_BiSeNet import BiSeNet
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, \
    per_class_iu
from loss import DiceLoss


def get_transform():
    train_transform = Compose([
        RandomResizedCrop(320, (0.5, 2.0)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(180),
        ColorJitter(0.3, 0.3, 0.3, 0.1),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])
    val_transform = Compose([
        PadCenterCrop(size=512),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def val(args, model, dataloader):
    print('start val!')
    # label_info = get_label_info(csv_path)
    '''
    model.eval() is a kind of switch for some specific layers/parts of the model that
    behave differently during training and inference (evaluating) time. For example, 
    Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, 
    and .eval() will do it for you. In addition, the common practice for evaluating/validation 
    is using torch.no_grad() in pair with model.eval() to turn off gradients computation.
    After that we call model.train() to reswitch them on.
    '''
    with torch.no_grad():
        model.eval()
        # lista delle precisioni
        precision_record = []
        
        # inizializziamo una matrice per calcolare successivamente mIoU
        hist = np.zeros((args.num_classes, args.num_classes))
        
        # Itarates over what? Single images or batches?
        for i, (data, label) in enumerate(dataloader):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda().long()

            output = model(data)

            # get RGB predict image
            _, prediction = output.max(dim=1)  # B, H, W
            label = label.cpu().numpy()
            prediction = prediction.cpu().numpy()

            # compute per pixel accuracy
            precision = compute_global_accuracy(prediction, label)
            
            # Cosa fa fast_hist????
            hist += fast_hist(label.flatten(), prediction.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)
        
        # compute the mean precision:
        precision = np.mean(precision_record)
        # miou = np.mean(per_class_iu(hist))
        miou_list = per_class_iu(hist)[:-1]  # preché tutti tranne l'ultimo?????
        # miou_dict, miou = cal_miou(miou_list, csv_path)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        # miou_str = ''
        # for key in miou_dict:
        #     miou_str += '{}:{},\n'.format(key, miou_dict[key])
        # print('mIoU for each class:')
        # print(miou_str)
        return precision, miou


def train(args, model, optimizer, dataloader_train, dataloader_val):
    # E' l'oggetto che ci stampa a schermo ciò chee acca
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))
    # settiamo la loss
    if args.loss == 'dice':
        # classe definita da loro nel file loss.py
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    # inizializziamo i contatori
    max_miou = 0
    step = 0
    # iniziamo il training
    for epoch in range(args.num_epochs):
        # inizializziamo il learning rate
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        # iniziamo il train
        model.train()
        # cosa grafica sequenziale
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        # Crediamo che sia la lista delle loss di ogni batch:
        loss_record = []
        
        # per ogni immagine o per ogni batch??? Ipotizziamo sia su ogni singolo mini-batch
        for i, (data, label) in enumerate(dataloader_train):
            
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda().long()

            # Prendiamo: 
            # - risultato finale dopo FFM
            # - risultato del 16xdown del contextPath, dopo ARM, modificati (?)
            # - risultato del 32xdown del contextPath, dopo ARM, modificati (?)
            output, output_sup1, output_sup2 = model(data)
            
            # Calcoliammo la loss 
            # Principal loss function (l_p in the paper):
            loss1 = loss_func(output, label)
            # Auxilary loss functions (l_i, for i=2, 3 in the paper):
            loss2 = loss_func(output_sup1, label)
            loss3 = loss_func(output_sup2, label)
            
            # alfa = 1, compute equation 2:
            loss = loss1 + loss2 + loss3
            
            # codice grafica
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            
            '''
            zero_grad clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
            loss.backward() computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
            opt.step() causes the optimizer to take a step based on the gradients of the parameters.
            '''
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # incrementiamo il contatore
            step += 1
            # aggiungiamo i valori per il grafico
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        
        # salva il modello fin ora trainato
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.state_dict(),
                       os.path.join(args.save_model_path, 'model.pth'))

        # compute validation every 10 epochs
        if epoch % args.validation_step == 0 and epoch != 0:
            
            # chaiam la funzione val che da in output le metriche
            precision, miou = val(args, model, dataloader_val)
            
            # salva miou max e salva il relativo miglior modello
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(args.save_model_path, 'best_dice_loss.pth'))
                
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
    # proviamo a terminare il writer per vedere se stampa qualcosa
    writer.close()


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='data', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default="checkpoints", help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')

    # settiamo i nostri parametri
    args = parser.parse_args(params)

    # create dataset and dataloader
    train_path = args.data
    train_transform, val_transform = get_transform()
    
    # creiamo un oggetto di tipo VOC per il training
    dataset_train = VOC(train_path, image_set="train", transform=train_transform)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    
    # creiamo un oggetto di tipo VOC per la validation
    dataset_val = VOC(train_path, image_set="val", transform=val_transform)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = model.cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    # load pretrained model if exists
    # Non ce l'abbiamo
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # train
    # funzioni presenti in questo file
    train(args, model, optimizer, dataloader_train, dataloader_val)

    val(args, model, dataloader_val)


if __name__ == '__main__':
    # settiamo i parametri che vogliamo cambiare
    params = [
        '--num_epochs', '30',
        '--learning_rate', '1e-3',
        '--data', 'data',
        '--num_workers', '8',
        '--num_classes', '21',
        '--cuda', '0',
        '--batch_size', '16',
        '--save_model_path', './checkpoints_18_sgd',
        '--context_path', 'resnet18',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--optimizer', 'sgd',

    ]
    main(params)

