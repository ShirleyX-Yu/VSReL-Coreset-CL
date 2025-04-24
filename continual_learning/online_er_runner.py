# -*-coding:utf8-*-

import os.path
import torch
import torchvision
import numpy as np

from continual_learning import online_rho_buffer
import utils


class OnlineERRunner(object):
    def __init__(self, local_path, buffer_size, model_params, use_cuda, transforms, selection_params, train_params,
                 update_mode, remove_mode, seed, scheduler_params=None):
        self.local_path = local_path
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        self.buffer_size = buffer_size
        self.model_params = model_params
        self.use_cuda = use_cuda
        self.transforms = transforms
        self.selection_params = selection_params
        self.train_params = train_params
        self.update_mode = update_mode
        self.remove_mode = remove_mode
        self.scheduler_params = scheduler_params
        self.seed = seed
        # build buffer
        self.buffer = online_rho_buffer.OnlineRhoBuffer(
            local_path=os.path.join(self.local_path, 'buffer'),
            buffer_size=self.buffer_size,
            selection_params=self.selection_params,
            model_params=self.model_params,
            use_cuda=self.use_cuda,
            transforms=self.transforms,
            remove_mode=self.remove_mode,
            add_mode=self.update_mode,
            seed=seed
        )
        # build model
        self.model = utils.build_model(model_params=self.model_params)
        self.to_pil = torchvision.transforms.ToPILImage()
        self.seen_tasks = 0
        self.results = []

    def train_single_task(self, train_loader, eval_loaders, verbose=True, do_evaluation=True):
        self.model.train()
        if self.train_params['use_cuda']:
            self.model.cuda()
        # make loss function
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        # make optimizer
        if self.train_params['opt_type'] == 'adam':
            opt = torch.optim.Adam(lr=self.train_params['lr'], params=self.model.parameters())
        else:
            opt = torch.optim.SGD(lr=self.train_params['lr'], params=self.model.parameters())
        if self.scheduler_params is not None:
            scheduler = utils.make_scheduler(scheduler_params=self.scheduler_params, optimizer=opt)
        else:
            scheduler = None
        for i in range(self.train_params['epochs']):
            step = 0
            for data in train_loader:
                d_id, sp, aug_sp, lab = data
                if self.use_cuda:
                    aug_sp = aug_sp.cuda()
                    lab = lab.cuda()
                if not self.buffer.is_empty():
                    buffer_data = self.buffer.get_sub_data(data_size=self.train_params['mem_batch_size'])
                    b_sp, b_lab = buffer_data
                    if self.train_params['use_cuda']:
                        b_sp = b_sp.cuda()
                        b_lab = b_lab.cuda()
                    in_sps = torch.cat([aug_sp, b_sp], dim=0)
                    in_labs = torch.cat([lab, b_lab], dim=0)
                    out_logits = self.model(in_sps)
                    ce_losses = loss_fn(out_logits, in_labs)
                    loss = torch.mean(ce_losses)
                else:
                    out_logits = self.model(aug_sp)
                    ce_losses = loss_fn(out_logits, lab)
                    loss = torch.mean(ce_losses)
                if verbose and step % 100 == 0:
                    print('loss at step ', step, 'is:', loss.item())
                opt.zero_grad()
                loss.backward()
                opt.step()
                if self.use_cuda:
                    lab = lab.cpu()
                    ce_losses = ce_losses.cpu()
                lab = lab.numpy()
                d_id = d_id.numpy()
                ce_losses = ce_losses.clone().detach().numpy()
                sps = []
                for j in range(sp.shape[0]):
                    sps.append(self.to_pil(sp[j, :]))
                self.buffer.update_buffer(
                    d_ids=d_id,
                    sps=sps,
                    labs=lab,
                    logits=None,
                    ce_loss=ce_losses
                )
                step += 1
            if scheduler is not None:
                scheduler.step()
            print('finish training epoch:', i)
            if do_evaluation and i % 10 == 0:
                accs, losses = self.evaluate_model(eval_loaders=eval_loaders, on_cuda=self.use_cuda)
                print('\taccuracy on test is:', np.mean(accs), accs, losses)
                if len(accs) > 1:
                    print('\tprevious tasks accuracy is:', np.mean(accs[:-1]))
        if self.train_params['use_cuda']:
            self.model.cpu()
        accs, losses = self.evaluate_model(eval_loaders=eval_loaders, on_cuda=False)
        print('\tlosses on testset is:', losses)
        self.results.append(accs)
        return accs

    def evaluate_model(self, eval_loaders, on_cuda=False):
        status = self.model.training
        self.model.eval()
        accs = []
        eval_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        losses = []
        with torch.no_grad():
            for i in range(self.seen_tasks + 1):
                loss = 0
                correct = 0
                for data, target in eval_loaders[i]:
                    if on_cuda:
                        data, target = data.cuda(), target.cuda()
                    output = self.model(data)
                    loss += eval_loss_fn(output, target).cpu().item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                avg_acc = 100. * correct / len(eval_loaders[i].dataset)
                avg_loss = loss / len(eval_loaders[i].dataset)
                accs.append(avg_acc)
                losses.append(avg_loss)
        self.model.train(status)
        return accs, losses

    def next_task(self, dump_buffer=False):
        self.buffer.next_task()
        if dump_buffer:
            self.buffer.dump_buffer(task_id=self.seen_tasks)
        self.seen_tasks += 1
