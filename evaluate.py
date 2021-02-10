import sys
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torchvision.models import ResNet

from utils import *


def evaluate(model, data_loader, neval_batches, device, args):
    if isinstance(model, DLRM_Net):
        return evaluate_dlrm(model, data_loader, neval_batches, device)
    elif args.dataset == "imagenet":
        return evaluate_imagenet(model, data_loader, neval_batches, device)
    else:
        return evaluate_bert(model, data_loader, neval_batches, device)

def evaluate_dlrm(model, data_loader, neval_batches, device):
    loss_fn = nn.BCELoss(reduction="mean")
    aggregator = Aggregator()

    with torch.no_grad():
        pbar = tqdm(total=len(data_loader), file=sys.stdout, leave=False)
        for it, (X, lS_o, lS_i, T) in enumerate(data_loader):
            lS_i = [S_i.to(device) for S_i in lS_i] if isinstance(lS_i, list) \
                else lS_i.to(device)
            lS_o = [S_o.to(device) for S_o in lS_o] if isinstance(lS_o, list) \
                else lS_o.to(device)
            Z = model(X.to(device), lS_o, lS_i)

            # loss
            loss = loss_fn(Z, T.to(device))

            aggregator.update(Z, T)
            auroc = roc_auc_score(aggregator.targets, aggregator.preds)
            pbar.set_description("Val loss {:4f}, AUROC {:.4f}".format(loss, auroc))
            pbar.update(1)
            if it >= neval_batches > 0:
                pbar.close()
                return auroc
        pbar.close()

    return auroc

def evaluate_imagenet(model, data_loader, neval_batches, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    cnt = 0
    with torch.no_grad():
        pbar = tqdm(total=min(len(data_loader), neval_batches), file=sys.stdout, leave=False)
        for image, target, paths in data_loader:
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            _ = criterion(output, target)
            cnt += 1
            acc1 = accuracy_topk(output, target, topk=(1,))
            top1.update(acc1[0], target.shape[0])
            pbar.set_description("val acc: {:.4f}".format(top1.avg.item()))
            pbar.update(1)
            if cnt >= neval_batches > 0:
                pbar.close()
                return top1.avg.item()
        pbar.close()

    return top1.avg.item()


def evaluate_snr_cascading(model, data_loader, neval_batches, device):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    cnt = 0
    start_recorders(model)
    with torch.no_grad():
        pbar = tqdm(total=min(len(data_loader), neval_batches), file=sys.stdout, leave=False)
        for image, target in data_loader:
            image = image.to(device)
            target = target.to(device)

            # Save clean activation feature maps
            start_recording_clean(model)
            output = model(image)
            stop_recording_clean(model)

            # Evaluate cascading noise
            output = model(image)

            cnt += 1
            acc1 = accuracy_topk(output, target, topk=(1,))
            top1.update(acc1[0], target.shape[0])
            pbar.set_description("val acc: {:.4f}".format(top1.avg.item()))
            pbar.update(1)
            if cnt >= neval_batches > 0:
                pbar.close()
                return top1.avg.item()
        pbar.close()

    stop_recorders(model)
    return top1.avg.item()


def evaluate_bert(model, data_loader, neval_batches, device):
    def eval_step(inputs, meter):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        output = model(**inputs)
        preds = output['logits']
        target = inputs['labels']

        acc1 = accuracy_topk(preds, target, topk=(1,))
        meter.update(acc1[0], target.shape[0])

    if isinstance(data_loader, tuple):
        matched_data_loader = data_loader[0]
        mismatched_data_loader = data_loader[1]

        model.eval()
        top1_matched = AverageMeter('Acc@1', ':6.2f')
        top1_mismatched = AverageMeter('Acc@1', ':6.2f')
        cnt = 0
        with torch.no_grad():
            pbar = tqdm(total=min(len(data_loader), neval_batches), file=sys.stdout, leave=False)
            for matched_inputs, mismatched_inputs in zip(matched_data_loader, mismatched_data_loader):
                eval_step(matched_inputs, top1_matched)
                eval_step(mismatched_inputs, top1_mismatched)

                cnt += 1
                pbar.set_description("val matched acc: {:.4f}, val mismatched acc{:.4f}".format(
                    top1_matched.avg.item(), top1_mismatched.avg.item()))
                pbar.update(1)
                if cnt >= neval_batches > 0:
                    pbar.close()
                    return (top1_matched.avg.item() + top1_mismatched.avg.item()) / 2
            pbar.close()

        # No single metric, so just return the mean.
        return (top1_matched.avg.item() + top1_mismatched.avg.item()) / 2

    else:
        top1 = AverageMeter('Acc@1', ':6.2f')
        cnt = 0
        with torch.no_grad():
            pbar = tqdm(total=min(len(data_loader), neval_batches), file=sys.stdout, leave=False)
            for inputs in data_loader:
                eval_step(inputs, top1)

                cnt += 1
                pbar.set_description("val acc: {:.4f}".format(top1.avg.item()))
                pbar.update(1)
                if cnt >= neval_batches > 0:
                    pbar.close()
                    return top1.avg.item()
            pbar.close()

        return top1.avg.item()
