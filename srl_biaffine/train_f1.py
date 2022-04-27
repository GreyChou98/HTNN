import tempfile

import torch
import torch.nn as nn
import argparse
import math
from model import SRLModel_attn
from dataloader import ExampleSet, new_collate_fn
from dataloader_overlap import ExampleSet_overlap, new_collate_fn_overlap
import logging
from collections import OrderedDict
import hashlib
from os import path
import os
import sys

from torch.multiprocessing import Process
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from distributed_utils import reduce_value, is_main_process, cleanup
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.nn.init import xavier_uniform_

from encoder import Encoder

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_model_name(hp):
    opt_dict = OrderedDict()
    # Only include important options in hash computation
    imp_opts = ['model', 'model_size', 'batch_size', 'real_batch_size', 'eval_steps', 'use_tag',
                'tag_dim', 'fine_tune', 'proj_dim', 'pool_method', 'just_last_layer', 'comb_method',
                'train_frac', 'seed', 'lr']
    hp_dict = vars(hp)
    for key in imp_opts:
        val = hp_dict[key]
        opt_dict[key] = val
        logging.info("%s\t%s" % (key, val))

    str_repr = str(opt_dict.items())
    hash_idx = hashlib.md5(str_repr.encode("utf-8")).hexdigest()
    model_name = "SRL_" + str(hash_idx)

    if hp.no_proj:
        model_name += "_no_proj"
        logging.info("no_proj\tTrue")
    if hp.no_layer_weight:
        model_name += "_no_layer_weight"
        logging.info("no_layer_weight\tTrue")

    if hp.fine_tune:
        model_name = "ft_" + model_name

    return model_name

def save_model(args, model, optimizer, scheduler, steps_done, max_f1, num_stuck_evals, location):
    """Save model."""
    save_dict = {}
    save_dict['weighing_params'] = model.module.encoder.weighing_params
    if args.fine_tune:
        save_dict['encoder'] = model.module.encoder.model.state_dict()
    save_dict['span_net'] = model.module.span_net.state_dict()
    save_dict['Treenet'] = model.module.Treenet.state_dict()
    save_dict['biaffine_net'] = model.module.biaffine_net.state_dict()
    if args.use_tag:
        save_dict['tag_emb'] = model.module.tag_emb.state_dict()
    save_dict.update({
        'steps_done': steps_done,
        'max_f1': max_f1,
        'num_stuck_evals': num_stuck_evals,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'rng_state': torch.get_rng_state(),
    })
    torch.save(save_dict, location)
    logging.info("Model saved at: %s" % (location))

def main_fun(rank, world_size, args):
    if not torch.cuda.is_available():
        raise EnvironmentError("not find GPU device for training")

    # initialize the process
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "41002"

    args.rank = rank
    args.world_size = world_size
    args.gpu = rank
    args.distributed = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()

    rank = args.rank
    # device = torch.device(args.device)
    device = torch.device(f'cuda:{args.gpu}')
    batch_size = args.batch_size
    if not args.test_model:
        args.lr *= args.world_size

    model_name = get_model_name(args)
    model_path = path.join(args.model_dir, model_name)
    best_model_path = path.join(model_path, 'best_models')

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        # print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        # tb_writer = SummaryWriter()

        if not path.exists(model_path):
            os.makedirs(model_path)
        if not path.exists(best_model_path):
            os.makedirs(best_model_path)

    encoder = Encoder(model=args.model, model_size=args.model_size,
                      fine_tune=args.fine_tune, cased=True)

    train_path = os.path.join(args.data_dir, 'train.json')
    val_path = os.path.join(args.data_dir, 'development.json')
    test_path = os.path.join(args.data_dir, 'test.json')

    if args.use_overlap:
        trainset = ExampleSet_overlap(train_path, encoder)
        valset = ExampleSet_overlap(val_path, encoder)
        testset = ExampleSet_overlap(test_path, encoder)
        collate_func = new_collate_fn_overlap
    else:
        trainset = ExampleSet(train_path, encoder)
        valset = ExampleSet(val_path, encoder)
        testset = ExampleSet(test_path, encoder)
        collate_func = new_collate_fn

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(testset)

    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))


    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=collate_func)

    val_loader = torch.utils.data.DataLoader(valset,
                                             batch_size=batch_size,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=collate_func)

    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=batch_size,
                                              sampler=test_sampler,
                                              pin_memory=True,
                                              num_workers=nw,
                                              collate_fn=collate_func)

    assert args.comb_method in ['avg', 'attn', 'attn2wise']


    model = SRLModel_attn(encoder, num_layers=args.num_layers, num_labels=67,
                          just_last_layer=args.just_last_layer, use_proj=args.use_proj,
                          use_tag=args.use_tag, tag_dim=args.tag_dim,
                          proj_dim=args.proj_dim, pred_dim=args.pred_dim,
                          dropout=args.dropout, new_tree=args.new_tree,
                          comb_method=args.comb_method, use_overlap=args.use_overlap).to(device)


    if not args.fine_tune:
        optimizer = torch.optim.Adam(model.get_other_params(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # if not args.fine_tune:
    #     optimizer = optim.SGD(model.get_other_params(), lr=args.lr,
    #                              momentum=0.9, weight_decay=0.005)
    # else:
    #     optimizer = optim.SGD(model.parameters(), lr=args.lr,
    #                              momentum=0.9, weight_decay=0.005)


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                mode='max', patience=3, factor=0.5, verbose=True)
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.n_epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    steps_done = 0
    max_f1 = 0
    init_num_stuck_evals = 0

    num_steps = (args.n_epochs * len(trainset)) // (args.real_batch_size * args.world_size)
    num_steps = (num_steps // args.eval_steps) * args.eval_steps

    location = path.join(model_path, "model.pt")
    if path.exists(location):
        logging.info("Loading previous checkpoint")
        checkpoint = torch.load(location, map_location=device)

        model.encoder.weighing_params = checkpoint['weighing_params']
        model.span_net.load_state_dict(checkpoint['span_net'])
        model.Treenet.load_state_dict(checkpoint['Treenet'])
        model.biaffine_net.load_state_dict(checkpoint['biaffine_net'])

        if args.use_tag:
            model.tag_emb.load_state_dict(checkpoint['tag_emb'])
        if args.fine_tune:
            model.encoder.model.load_state_dict(checkpoint['encoder'])
        optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(
            checkpoint['scheduler_state_dict'])

        steps_done = checkpoint['steps_done']
        init_num_stuck_evals = checkpoint['num_stuck_evals']
        max_f1 = checkpoint['max_f1']
        torch.set_rng_state(checkpoint['rng_state'].cpu())
        logging.info("Steps done: %d, Max Precision: %.4f" % (steps_done, max_f1))

        dist.barrier()
    else:
        # checkpoint_path = path.join(tempfile.gettempdir(), "initial_weights.pt")
        #
        # if rank == 0:
        #     for p in model.parameters():
        #         if p.dim() > 1 and p.requires_grad==True:
        #             nn.init.kaiming_normal_(p)
        #     torch.save(model.state_dict(), checkpoint_path)
        #
        # dist.barrier()
        # model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        for p in model.parameters():
            if p.dim() > 1 and p.requires_grad == True:
                nn.init.kaiming_normal_(p)


    n_params = sum([p.nelement() for p in model.parameters() if p.requires_grad == True])
    if rank == 0:
        logging.info("Total number of parameters for this model: %d " % (n_params))
        logging.info("Total training steps: %d" % num_steps)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                      find_unused_parameters=True)

    if not args.test_model:
        train(rank, device, args, model, train_loader, val_loader, optimizer, scheduler, model_path,
              best_model_path, init_steps=steps_done, max_f1=max_f1, eval_steps=args.eval_steps,
              num_steps=num_steps, init_num_stuck_evals=init_num_stuck_evals)

    val_acc_sum_all, val_total_num, val_precision_all, val_acc_sum_nonull, val_nonull_num, val_pred_nonull_num, val_precision_nonull, val_recall_nonull, val_f1, \
    test_acc_sum_all, test_total_num, test_precision_all, test_acc_sum_nonull, test_nonull_num, test_pred_nonull_num, test_precision_nonull, test_recall_nonull, \
    test_miss_recall_nonull, test_f1, test_miss_f1 = \
        final_eval(args, model, best_model_path, val_loader, test_loader, device)

    if rank == 0:
        # logging.info("Val Acc_all: %d Total: %d All Precision: %.4f" % (val_acc_sum_all, val_total_num, val_precision_all))
        logging.info("Val Nonull F1: %.4f Precision: %.4f Recall: %.4f Acc_nonull: %d Total_nonull: %d Pred_nonull: %d"
                     % (val_f1, val_precision_nonull, val_recall_nonull, val_acc_sum_nonull, val_nonull_num, val_pred_nonull_num))
        # logging.info("Test Acc_all: %d Total: %d All Precision: %.4f" % (test_acc_sum_all, test_total_num, test_precision_all))
        logging.info("Test Nonull F1: %.4f Precision: %.4f Recall: %.4f Acc_nonull: %d Total_nonull: %d Pred_nonull: %d"
                     % (test_f1, test_precision_nonull, test_recall_nonull, test_acc_sum_nonull, test_nonull_num, test_pred_nonull_num))
        # logging.info("Missing Test Acc_all: %d Total: %d All Precision: %.4f" % (test_miss_acc_sum_all, test_miss_total_num, test_miss_precision_all))
        logging.info("Missing Test Nonull F1: %.4f Precision: %.4f Recall: %.4f"
                     % (test_miss_f1, test_precision_nonull, test_miss_recall_nonull))
        # if os.path.exists(checkpoint_path) is True:
        #     os.remove(checkpoint_path)

    cleanup()

def train(rank, device, args, model, train_iter, val_iter, optimizer, scheduler, model_dir,
          best_model_dir, init_steps=0, eval_steps=1000, num_steps=120000, max_f1=0,
          init_num_stuck_evals=0):

    model.train()
    loss_function = nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)

    steps_done = init_steps
    num_stuck_evals = init_num_stuck_evals
    batch_size_fac = args.real_batch_size // args.batch_size
    optimizer.zero_grad()

    while (steps_done < num_steps) and (num_stuck_evals < 20):
        if rank == 0:
            logging.info("Epoch started")
        for idx, batch_data in enumerate(train_iter):
            text, spans, child_rel, new_tags, predicates, labels, len_info = batch_data
            text = text[0].to(device)
            spans = spans.to(device)
            child_rel = child_rel.to(device)
            new_tags = new_tags.to(device)
            predicates = predicates.to(device)
            labels = labels.to(device)
            len_info = len_info.to(device)

            # check = int((s1 != s1).sum())
            # if (check > 0):
            #     logging.info("s1 contains Nan")
            # else:
            #     logging.info("s1 does not contain Nan, it might be other problem")
            #
            # check = int((s2 != s2).sum())
            # if (check > 0):
            #     logging.info("s2 contains Nan")
            # else:
            #     logging.info("s2 does not contain Nan, it might be other problem")
            #
            # check = int((labels != labels).sum())
            # if (check > 0):
            #     logging.info("labels contains Nan")
            # else:
            #     logging.info("labels does not contain Nan, it might be other problem")
            #
            # check = int((len_info != len_info).sum())
            # if (check > 0):
            #     logging.info("len_info contains Nan")
            # else:
            #     logging.info("len_info does not contain Nan, it might be other problem")

            pred_label, label = model(text, spans, child_rel, labels, new_tags, predicates, len_info)
            loss = loss_function(pred_label, label)
            loss.backward()

            # grad accumulation
            if (idx + 1) % batch_size_fac == 0:
                optimizer.step()
                optimizer.zero_grad()
                steps_done += 1

            # loss = reduce_value(loss.detach().item(), average=True)
            # if not torch.isfinite(loss):
            #     print('WARNING: non-finite loss, ending training ', loss)
            #     sys.exit(1)

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            if ((idx + 1) % batch_size_fac == 0) and (steps_done % eval_steps == 0):
                acc_sum_all, total_num, precision_all, acc_sum_nonull, nonull_num, pred_nonull_num, precision_nonull, recall_nonull, f1 \
                    = evaluate(model, val_iter, device, test_missing=False)
                scheduler.step(f1)
                if rank == 0:
                    logging.info("Evaluating at % d" % steps_done)
                    location = path.join(model_dir, "model.pt")
                    save_model(args, model, optimizer, scheduler, steps_done,
                               max_f1, num_stuck_evals, location)

                    logging.info("Val All Precision: %.4f Acc_all: %d Total: %d" %
                                 (precision_all, acc_sum_all, total_num))
                    logging.info("Val Nonull F1: %.4f Precision: %.4f Recall: %.4f Acc_nonull: %d Total_nonull: %d Pred_nonull: %d (Max Precision: %.4f)" %
                                 (f1, precision_nonull, recall_nonull, acc_sum_nonull, nonull_num, pred_nonull_num, max_f1))
                    # tb_writer.add_scalar('f1', f1, steps_done)
                    # tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], steps_done)
                dist.barrier()
                if f1 > max_f1:
                    num_stuck_evals = 0
                    max_f1 = f1
                    if rank == 0:
                        logging.info("Max Nonull F1: %.4f" % max_f1)
                        location = path.join(best_model_dir, "model.pt")
                        save_model(args, model, optimizer, scheduler, steps_done,
                                   max_f1, num_stuck_evals, location)

                else:
                    num_stuck_evals += 1

                if num_stuck_evals >= 20:
                    logging.info("No improvement for 20 evaluations")
                    break
                sys.stdout.flush()

        if rank == 0:
            logging.info("Epoch done!\n")

    if rank == 0:
        logging.info("Training done!\n")


@torch.no_grad()
def evaluate(model, data_loader, device, test_missing=False, missing_num=0):
    model.eval()

    acc_sum_all = torch.zeros(1).to(device)
    acc_sum_nonull = torch.zeros(1).to(device)
    n_all = torch.zeros(1).to(device)
    n_nonull = torch.zeros(1).to(device)
    pred_n_nonull = torch.zeros(1).to(device)

    # all_res = []
    for batch_data in data_loader:
        text, spans, child_rel, new_tags, predicates, labels, len_info = batch_data
        text = text[0].to(device)
        spans = spans.to(device)
        child_rel = child_rel.to(device)
        new_tags = new_tags.to(device)
        predicates = predicates.to(device)
        labels = labels.to(device)
        len_info = len_info.to(device)
        pred_label, label = model(text, spans, child_rel, labels, new_tags, predicates,  len_info)
        pred_label = pred_label.argmax(1)

        pred_label_mask = ~(pred_label == 66)
        pred_nonull_sum = (pred_label_mask).float().sum().item()

        nonull_label_mask = ~(label == 66)
        nonull_label = torch.masked_select(label, nonull_label_mask)
        nonull_real_instance = torch.masked_select(pred_label, nonull_label_mask)

        acc_sum_all += (pred_label == label).float().sum().item()
        n_all += label.size(0)
        acc_sum_nonull += (nonull_real_instance == nonull_label).float().sum().item()
        n_nonull += nonull_label.size(0)
        pred_n_nonull += pred_nonull_sum

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    acc_sum_all = reduce_value(acc_sum_all, average=False)
    n_all = reduce_value(n_all, average=False)
    acc_sum_nonull = reduce_value(acc_sum_nonull, average=False)
    n_nonull = reduce_value(n_nonull, average=False)
    pred_n_nonull = reduce_value(pred_n_nonull, average=False)
        # batch_size = label.shape[0]
        # # span1 = batch_data.orig_span1
        # # span2 = batch_data.orig_span2
        # for idx in range(batch_size):
        #     all_res.append({
        #         # 'span1': span1[idx, :].tolist(),
        #         # 'span2': span2[idx, :].tolist(),
        #         'tp': torch.sum(label[idx, :] * pred[idx, :]),
        #         'pred': torch.sum(pred[idx, :]),
        #         'label_vec': label[idx, :],
        #         'pred_vec': pred[idx, :]})

    precision_all = acc_sum_all / n_all
    recall_nonull = acc_sum_nonull / n_nonull
    precision_nonull = acc_sum_nonull / pred_n_nonull
    f1 = (2 * recall_nonull * precision_nonull) / (recall_nonull + precision_nonull)

    model.train()

    if test_missing:
        n_all = n_all + missing_num
        n_nonull = n_nonull + missing_num
        precision_all_miss = acc_sum_all / n_all
        recall_nonull_miss = acc_sum_nonull / n_nonull
        f1_miss = (2 * recall_nonull_miss * precision_nonull) / (recall_nonull_miss + precision_nonull)
        return acc_sum_all.item(), n_all.item(), precision_all.item(), acc_sum_nonull.item(), n_nonull.item(), \
           pred_n_nonull.item(), precision_nonull.item(), recall_nonull.item(), recall_nonull_miss.item(), f1, f1_miss

    else:
        return acc_sum_all.item(), n_all.item(), precision_all.item(), acc_sum_nonull.item(), n_nonull.item(), \
               pred_n_nonull.item(), precision_nonull.item(), recall_nonull.item(), f1



def final_eval(args, model, best_model_dir, val_loader, test_loader, device):
    location = path.join(best_model_dir, "model.pt")
    if path.exists(location):
        logging.info("Loading best checkpoint")
        checkpoint = torch.load(location, map_location=device)

        model.module.encoder.weighing_params = checkpoint['weighing_params']
        model.module.span_net.load_state_dict(checkpoint['span_net'])
        model.module.Treenet.load_state_dict(checkpoint['Treenet'])
        model.module.biaffine_net.load_state_dict(checkpoint['biaffine_net'])

        if args.use_tag:
            model.module.tag_emb.load_state_dict(checkpoint['tag_emb'])

        if args.fine_tune:
            model.module.encoder.model.load_state_dict(checkpoint['encoder'])

        val_acc_sum_all, val_total_num, val_precision_all, val_acc_sum_nonull, \
        val_nonull_num, val_pred_nonull_num, val_precision_nonull, val_recall_nonull, val_f1 \
            = evaluate(model, val_loader, device, test_missing=False)
        test_acc_sum_all, test_total_num, test_precision_all, test_acc_sum_nonull, \
        test_nonull_num, test_pred_nonull_num, test_precision_nonull, test_recall_nonull, test_miss_recall_nonull, test_f1, test_miss_f1 \
            = evaluate(model, test_loader, device, test_missing=True, missing_num=args.missing_num)

    return val_acc_sum_all, val_total_num, val_precision_all, val_acc_sum_nonull, val_nonull_num,  val_pred_nonull_num, val_precision_nonull, val_recall_nonull, val_f1, \
           test_acc_sum_all, test_total_num, test_precision_all, test_acc_sum_nonull, test_nonull_num, test_pred_nonull_num, test_precision_nonull, test_recall_nonull, \
           test_miss_recall_nonull, test_f1, test_miss_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str,
        default="/public/sist/home/zhanglw/zh/TTIC/srl2")
    parser.add_argument(
        "--model_dir", type=str,
        default="/public/sist/home/zhanglw/zh/srl_int/checkpoints_no_tag_bert_base/")
    # parser.add_argument(
    #     "--data_dir", type=str,
    #     default="/p300/srl")
    # parser.add_argument(
    #     "--model_dir", type=str,
    #     default="/public/sist/home/zhanglw/zh/srl_int/checkpoints_no_tag_bert_base/")
    parser.add_argument(
        "--previous_treenet", type=str,
        default="/public/sist/home/zhanglw/zh/srl_int/checkpoints_no_tag_bert_base/")
    parser.add_argument("--test_model", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--new_tree", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--use_overlap", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--real_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--comb_method", type=str, default='attn')
    parser.add_argument("--use_tag", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--use_proj", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--tag_dim", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--pred_dim", type=int, default=256)
    parser.add_argument("--model", type=str, default="bert")
    parser.add_argument("--model_size", type=str, default="base")
    parser.add_argument("--just_last_layer", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--fine_tune", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--fine_tune_treenet", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--no_proj", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--no_layer_weight", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--pool_method", default="avg", type=str)
    parser.add_argument("--train_frac", default=1.0, type=float,
                        help="Can reduce this for quick testing.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--eval", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--slurm_id', help="Slurm ID",
                        default=None, type=str)

    parser.add_argument("--missing_num", type=int, default=0)

    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    opt = parser.parse_args()
    world_size = opt.world_size
    processes = []
    for rank in range(world_size):
        p = Process(target=main_fun, args=(rank, world_size, opt))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()