
import os
from metrics import *
from utils import *
from model import Att_Diffuse_model
from pcgrad import PCGrad
from torch import optim
from sasrec import SASRec


# from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup
def extract(data):
    seq= data[0]
    diff_loss = data[1] if len(data) == 2 else torch.zeros(1,device=seq.device)
    return seq, seq[:,-1], diff_loss

def item_num_create(args):
    length = {"ml-100k":1008,
              'yelp': 64669,
              'sports':12301,
              'baby':4731,
              'toys':7309,
              'beauty':6086
              }
    args.item_num = length[args.dataset]
    return args
def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        if args.model == 'adrec':
            opt= optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        else:
            opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        opt= optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError
    return opt

def choose_model(args):
    device = args.device
    if args.model in ['diffurec','adrec','dreamrec']:
        if args.model == 'adrec':
            # args.pcgrad=True
            args.pretrained=True
            args.freeze_emb=True
            pass
        if args.model == 'diffurec':
            args.split_onebyone=True
            args.parallel_ag = False
            args.is_causal = False
        model = Att_Diffuse_model(args)
    elif args.model == 'sasrec' or args.model == 'pretrain':
        model = SASRec(args)
    else:
        model=None
    return model.to(device)
# ("bert4rec" "core" "eulerformer" "fearec" "gru4rec" "trimlp")
def load_data(args):

    path_data = '../datasets/data/' + args.dataset + '/dataset.pkl'
    with open(path_data, 'rb') as f:
        data_raw = pickle.load(f)
    tra_data = Data_Train(data_raw['train'], args)
    val_data = Data_Val(data_raw['train'], data_raw['val'], args)
    test_data = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args)
    tra_data_loader = tra_data.get_pytorch_dataloaders()
    val_data_loader = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()

    return tra_data_loader, val_data_loader, test_data_loader

def model_train(model_joint,tra_data_loader, val_data_loader, test_data_loader, args, logger,train_time):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    # model_joint = torch.compile(model_joint, )
    torch.set_float32_matmul_precision('high')
    optimizer = PCGrad(optimizers(model_joint, args),args)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer.optim, T_max=500)
    best_metrics_dict = {'Best_HR@5': 0, 'Best_NDCG@5': 0, 'Best_HR@10': 0, 'Best_NDCG@10': 0, 'Best_HR@20': 0, 'Best_NDCG@20': 0}
    best_epoch = {'Best_epoch_HR@5': 0, 'Best_epoch_NDCG@5': 0, 'Best_epoch_HR@10': 0, 'Best_epoch_NDCG@10': 0, 'Best_epoch_HR@20': 0, 'Best_epoch_NDCG@20': 0}
    bad_count = 0
    best_model = None
    for epoch_temp in range(epochs):
        model_joint.train()
        if epoch_temp ==5 and args.model =='adrec':
            print(f'warm up finishied in epoch {epoch_temp}')
            logger.info(f'warm up finishied in epoch {epoch_temp}')
            model_joint.item_embedding.weight.requires_grad = True
        ce_losses = []
        dif_losses = []
        flag_update = 0
        pbr_train = tqdm(enumerate(tra_data_loader),desc='Epoch: {}'.format(epoch_temp),leave=False, total=len(tra_data_loader))
        # print('len',len(tra_data_loader))
        for index_temp, train_batch in pbr_train:
            train_batch = [x.to(device) for x in train_batch]
            optimizer.zero_grad()
            out_seq, last_item, *dif_loss = model_joint(train_batch[0], train_batch[1], train_flag=True)
            if len(dif_loss)>0:
                dif_loss=dif_loss[0]
            else:
                dif_loss=torch.zeros(1,device=args.device)
            ce_loss = model_joint.calculate_loss(out_seq, train_batch[1])  ## use this not above
            if args.model=='adrec' and args.loss=='mse':
                losses = [ce_loss, args.loss_scale * dif_loss]
            elif args.model=='dreamrec':
                losses =[dif_loss]
            else:
                losses=[ce_loss]
            optimizer.pc_backward(losses)
            ce_losses.append(ce_loss.item())
            dif_losses.append(dif_loss.item())
            optimizer.step()
            pbr_train.set_postfix_str(f'loss={ce_losses[-1]:.3f}')
            # if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
            #     print('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all[-1]))
            #     logger.info('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all[-1]))
        print(f"loss in epoch {epoch_temp}: ce_loss {sum(ce_losses)/len(ce_losses):.3f}, dif_loss {sum(dif_losses)/len(dif_losses):.3f}")
        logger.info(f"loss in epoch {epoch_temp}: ce_loss {sum(ce_losses)/len(ce_losses):.3f}, dif_loss {sum(dif_losses)/len(dif_losses):.3f}")
        lr_scheduler.step()
        # if epoch_temp == 10:
        #     args.eval_interval=3



        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            # logger.info(f"loss in epoch {epoch_temp}: ce_loss: {sum(ce_losses) / len(ce_losses):.3f}, dif_loss: {sum(dif_losses) / len(dif_losses):.3f}")
            # print(f"loss in epoch {epoch_temp}: ce_loss: {sum(ce_losses) / len(ce_losses):.3f}, dif_loss: {sum(dif_losses) / len(dif_losses):.3f}")
            # print('start predicting: ', datetime.datetime.now())
            # logger.info('start predicting: {}'.format(datetime.datetime.now()))
            metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
            model_joint.eval()
            with torch.no_grad():
                for val_batch in tqdm(val_data_loader,leave=False,desc='Denoising..., Epoch: {}'.format(epoch_temp)):
                    val_batch = [x.to(device) for x in val_batch]
                    out_seq, last_item, *_= model_joint(val_batch[0], val_batch[1], train_flag=False)

                    scores_rec_diffu = model_joint.calculate_score(last_item)    ### inner_production
                    # scores_rec_diffu = model_joint.routing_rep_pre(rep_diffu)   ### routing_rep_pre
                    # print(scores_rec_diffu.shape,val_batch[1][:,-1].shape)
                    metrics = hrs_and_ndcgs_k(scores_rec_diffu, val_batch[1][:,-1:], metric_ks)
                    for k, v in metrics.items():
                        metrics_dict[k].append(v)

            for key_temp, values_temp in metrics_dict.items():
                values_mean = round(np.mean(values_temp) * 100, 4)
                if values_mean > best_metrics_dict['Best_' + key_temp]:
                    flag_update = 1
                    bad_count = 0
                    best_metrics_dict['Best_' + key_temp] = values_mean
                    best_epoch['Best_epoch_' + key_temp] = epoch_temp
                    best_epoch_temp = epoch_temp

            if flag_update == 0:
                bad_count += 1
                print('patience to end: ', args.patience - bad_count)
            else:
                print(best_metrics_dict)
                print(best_epoch)
                logger.info(best_metrics_dict)
                logger.info(best_epoch)
                best_model = copy.deepcopy(model_joint)
            if bad_count >= args.patience:
                break
    # if args.model == 'adrec' and args.lambda_schedule:
    #     model_joint.diffu.net.lambda_uncertainty = schedule[best_epoch_temp]
    saved_dir = os.path.join('saved',args.model, args.dataset)
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    if args.model == 'pretrain':
        output_path = os.path.join(saved_dir,'pretrain.pth')
    else:
        output_path = os.path.join(saved_dir, str(train_time) + args.description + '.pth')
    # torch.save(best_model._orig_mod.state_dict(), str(output_path))
    torch.save(best_model.state_dict(), str(output_path))
    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    # if args.eval_interval > epochs:
    #     best_model = copy.deepcopy(model_joint)

    print('start testing: ', datetime.datetime.now())
    logger.info('start testing: {}'.format(datetime.datetime.now()))
    top_100_item = []
    model_joint.eval()
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in tqdm(test_data_loader,leave=False):
            test_batch = [x.to(device) for x in test_batch]
            out_seq, last_item, *_ = best_model(test_batch[0], test_batch[1], train_flag=False)
            scores_rec_diffu = best_model.calculate_score(last_item)   ### Inner Production
            # scores_rec_diffu = best_model.routing_rep_pre(rep_diffu)   ### routing

            _, indices = torch.topk(scores_rec_diffu, k=20)
            top_100_item.append(indices)
            metrics = hrs_and_ndcgs_k(scores_rec_diffu, test_batch[1][:,-1:], metric_ks)
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)

    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print('Test------------------------------------------------------')
    logger.info('Test------------------------------------------------------')
    print(test_metrics_dict_mean)
    logger.info(test_metrics_dict_mean)
    print('Best Eval---------------------------------------------------------')
    logger.info('Best Eval---------------------------------------------------------')
    print(best_metrics_dict)
    print(best_epoch)
    logger.info(best_metrics_dict)
    logger.info(best_epoch)
    print(args)




    return best_model, test_metrics_dict_mean

