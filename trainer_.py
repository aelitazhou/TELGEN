from functools import partial

import numpy as np
import torch
from torch_scatter import scatter
import copy

from data.utils import barrier_function
import time


class Trainer:
    def __init__(self,
                 device,
                 loss_target,
                 loss_type,
                 micro_batch,
                 ipm_steps,
                 ipm_alpha,
                 loss_weight):
        assert 0. <= ipm_alpha <= 1.
        self.ipm_steps = ipm_steps           # number of steps being supervised
        self.ipm_alpha = ipm_alpha
        self.step_weight = torch.tensor([ipm_alpha ** (ipm_steps - l - 1)
                                         for l in range(ipm_steps)],
                                        dtype=torch.float, device=device)[None]    # step decay factor
        self.device = device
        # self.best_val_loss = 1.e8
        self.best_val_objgap = 100.
        self.best_val_consgap = 100.
        self.patience = 0
        self.device = device
        self.loss_target = loss_target.split('+')
        self.loss_weight = loss_weight
        if loss_type == 'l2':
            self.loss_func = partial(torch.pow, exponent=2)
        elif loss_type == 'l1':
            self.loss_func = torch.abs
        else:
            raise ValueError
        self.micro_batch = micro_batch

    def train(self, dataloader, model, optimizer):
        model.train()
        optimizer.zero_grad()

        update_count = 0
        micro_batch = int(min(self.micro_batch, len(dataloader)))
        loss_scaling_lst = [micro_batch] * (len(dataloader) // micro_batch) + [len(dataloader) % micro_batch]

        train_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _,  = model(data)    # predicted obj and con
            loss = self.get_loss(vals, data)

            train_losses += loss.detach() * data.num_graphs
            num_graphs += data.num_graphs

            update_count += 1
            loss = loss / float(loss_scaling_lst[0])  # scale the loss
            loss.backward()

#             if update_count >= micro_batch or i == len(dataloader) - 1:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(),
#                                                max_norm=1.0,
#                                                error_if_nonfinite=True)

            if update_count >= micro_batch or i == len(dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=1.0,
                                               error_if_nonfinite=False)
                optimizer.step()
                optimizer.zero_grad()
                update_count = 0
                loss_scaling_lst.pop(0)

        return train_losses.item() / num_graphs
    
    
    # self.get_loss_new: loss, primal_loss, obj_loss, cons_loss
    def train_harp(self, dataloader, model, optimizer):
        model.train()
        optimizer.zero_grad()

        update_count = 0
        micro_batch = int(min(self.micro_batch, len(dataloader)))
        loss_scaling_lst = [micro_batch] * (len(dataloader) // micro_batch) + [len(dataloader) % micro_batch]

        train_losses = 0.
        primal_losses = 0.
        obj_losses = 0.
        cons_losses = 0.
        econs_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            #print('dataloader', i)
            data = data.to(self.device)
            vals, c, ec = model(data)    # predicted obj and con
            loss, primal_loss, obj_loss, cons_loss, econs_loss = self.get_loss_harp(vals, data)

            train_losses += loss.detach() * data.num_graphs
            primal_losses += primal_loss.detach() * data.num_graphs
            obj_losses += obj_loss.detach() * data.num_graphs
            cons_losses += cons_loss.detach() * data.num_graphs
            econs_losses += econs_loss.detach() * data.num_graphs
            
            num_graphs += data.num_graphs

            update_count += 1
            loss = loss / float(loss_scaling_lst[0])  # scale the loss
            loss.backward()

            if update_count >= micro_batch or i == len(dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=1.0,
                                               error_if_nonfinite=False)
                optimizer.step()
                optimizer.zero_grad()
                update_count = 0
                loss_scaling_lst.pop(0)

        return train_losses.item() / num_graphs, primal_losses.item() / num_graphs, obj_losses.item() / num_graphs, cons_losses.item() / num_graphs, econs_losses.item() / num_graphs
    
    
    # self.get_loss_new: loss, primal_loss, obj_loss, cons_loss
    def train_new(self, dataloader, model, optimizer):
        model.train()
        optimizer.zero_grad()

        update_count = 0
        micro_batch = int(min(self.micro_batch, len(dataloader)))
        loss_scaling_lst = [micro_batch] * (len(dataloader) // micro_batch) + [len(dataloader) % micro_batch]

        train_losses = 0.
        primal_losses = 0.
        obj_losses = 0.
        cons_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)    # predicted obj and con
            loss, primal_loss, obj_loss, cons_loss = self.get_loss_new(vals, data)

            train_losses += loss.detach() * data.num_graphs
            primal_losses += primal_loss.detach() * data.num_graphs
            obj_losses += obj_loss.detach() * data.num_graphs
            cons_losses += cons_loss.detach() * data.num_graphs
            
            num_graphs += data.num_graphs

            update_count += 1
            loss = loss / float(loss_scaling_lst[0])  # scale the loss
            loss.backward()

            if update_count >= micro_batch or i == len(dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=1.0,
                                               error_if_nonfinite=False)
                optimizer.step()
                optimizer.zero_grad()
                update_count = 0
                loss_scaling_lst.pop(0)

        return train_losses.item() / num_graphs, primal_losses.item() / num_graphs, obj_losses.item() / num_graphs, cons_losses.item() / num_graphs
    
    
    # self.get_loss_newcon: loss, primal_loss, obj_loss, cons_loss
    # con_loss: softmax, sum, mean
    def train_newcon(self, dataloader, model, optimizer):
        model.train()
        optimizer.zero_grad()

        update_count = 0
        micro_batch = int(min(self.micro_batch, len(dataloader)))
        loss_scaling_lst = [micro_batch] * (len(dataloader) // micro_batch) + [len(dataloader) % micro_batch]

        train_losses = 0.
        primal_losses = 0.
        obj_losses = 0.
        cons_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)    # predicted obj and con
            loss, primal_loss, obj_loss, cons_loss = self.get_loss_newcon(vals, data)

            train_losses += loss.detach() * data.num_graphs
            primal_losses += primal_loss.detach() * data.num_graphs
            obj_losses += obj_loss.detach() * data.num_graphs
            cons_losses += cons_loss.detach() * data.num_graphs
            
            num_graphs += data.num_graphs

            update_count += 1
            loss = loss / float(loss_scaling_lst[0])  # scale the loss
            loss.backward()

            if update_count >= micro_batch or i == len(dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=1.0,
                                               error_if_nonfinite=False)
                optimizer.step()
                optimizer.zero_grad()
                update_count = 0
                loss_scaling_lst.pop(0)

        return train_losses.item() / num_graphs, primal_losses.item() / num_graphs, obj_losses.item() / num_graphs, cons_losses.item() / num_graphs
    
    
    
    # self.get_loss_newnewcon: loss, primal_loss, obj_loss, cons_loss
    # con_loss: amax, mean
    def train_newnewcon(self, dataloader, model, optimizer):
        model.train()
        optimizer.zero_grad()

        update_count = 0
        micro_batch = int(min(self.micro_batch, len(dataloader)))
        loss_scaling_lst = [micro_batch] * (len(dataloader) // micro_batch) + [len(dataloader) % micro_batch]

        train_losses = 0.
        primal_losses = 0.
        obj_losses = 0.
        cons_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            #print('dataloader', i)
            vals, _ = model(data)    # predicted obj and con
            loss, primal_loss, obj_loss, cons_loss = self.get_loss_newnewcon(vals, data)

            train_losses += loss.detach() * data.num_graphs
            primal_losses += primal_loss.detach() * data.num_graphs
            obj_losses += obj_loss.detach() * data.num_graphs
            cons_losses += cons_loss.detach() * data.num_graphs
            
            num_graphs += data.num_graphs

            update_count += 1
            loss = loss / float(loss_scaling_lst[0])  # scale the loss
            loss.backward()

            if update_count >= micro_batch or i == len(dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=1.0,
                                               error_if_nonfinite=False)
                optimizer.step()
                optimizer.zero_grad()
                update_count = 0
                loss_scaling_lst.pop(0)

        return train_losses.item() / num_graphs, primal_losses.item() / num_graphs, obj_losses.item() / num_graphs, cons_losses.item() / num_graphs
    
    
    
    
    # self.get_loss_newnewcon: loss, primal_loss, obj_loss, cons_loss
    # con_loss: amax, mean
    # add_layer
    def train_newnewcon_add_layer(self, ipm, dataloader, model, optimizer):
        
        train_losses = 0.
        primal_losses = 0.
        obj_losses = 0.
        cons_losses = 0.
        num_graphs = 0
        
        for n in range(len(ipm)): 
            #print('ipm', ipm[n])
            model[n].train()
            optimizer.zero_grad()

            update_count = 0
            micro_batch = int(min(self.micro_batch, len(dataloader[n])))
            loss_scaling_lst = [micro_batch] * (len(dataloader[n]) // micro_batch) + [len(dataloader[n]) % micro_batch]
            #print('loss_scaling_lst', loss_scaling_lst) 
            for i, data in enumerate(dataloader[n]):
                data = data.to(self.device)
                vals, _ = model[n](data)    # predicted obj and con
                loss, primal_loss, obj_loss, cons_loss = self.get_loss_newnewcon(vals, data)

                train_losses += loss.detach() * data.num_graphs
                primal_losses += primal_loss.detach() * data.num_graphs
                obj_losses += obj_loss.detach() * data.num_graphs
                cons_losses += cons_loss.detach() * data.num_graphs

                num_graphs += data.num_graphs

                update_count += 1
                loss = loss / float(loss_scaling_lst[0])  # scale the loss
                loss.backward()

                if update_count >= micro_batch or i == len(dataloader[n]) - 1:
                    torch.nn.utils.clip_grad_norm_(model[n].parameters(),
                                                   max_norm=1.0,
                                                   error_if_nonfinite=False)
                    optimizer.step()
                    optimizer.zero_grad()
                    update_count = 0
                    loss_scaling_lst.pop(0)
                    
            if (n+1) < len(ipm):
                #print('give para to next model: model', n+1)
                model[n+1].load_state_dict(model[n].state_dict())
            else:
                #print('give para to next model: model', 0)
                model[0].load_state_dict(model[n].state_dict())

        return train_losses.item() / num_graphs, primal_losses.item() / num_graphs, obj_losses.item() / num_graphs, cons_losses.item() / num_graphs
    
    
    
    
    # self.get_loss_newnewcon: loss, primal_loss, obj_loss, cons_loss
    # con_loss: amax, mean
    # add_layer: ipm = [16, 20, 24]
    # shuffle data to amke sure the training process is fair
    # all_train_data: a shuffled list containing all training data (ipm=16/20/24)
    def train_newnewcon_add_layer_shuffle(self, dataloader, model, optimizer):
        
        train_losses = 0.
        primal_losses = 0.
        obj_losses = 0.
        cons_losses = 0.
        num_graphs = 0

        update_count = 0
        micro_batch = int(min(self.micro_batch, len(all_train_data)))
        loss_scaling_lst = [micro_batch] * (len(all_train_data) // micro_batch) + [len(all_train_data) % micro_batch]

        for i, data in enumerate(all_train_data):
            data = data.to(self.device)
            ipm_step = data.gt_primals.shape[1]
            idx = ipm.index(ipm_step)
            model_now = model[idx]
            #print('ipm_idx', ipm_step)
            #print('data.gt_primals.shape', data.gt_primals.shape)
            #print('load temp_model', temp_model['encoder.vals.0.weight'][0])
            model_now.load_state_dict(temp_model)
            
            model_now.train()
            optimizer.zero_grad()
            
            vals, _ = model_now(data)    # predicted obj and con
            step_weight = torch.tensor([self.ipm_alpha ** (ipm_step - l - 1) for l in range(ipm_step)],
                                        dtype=torch.float, device=self.device)[None]
            loss, primal_loss, obj_loss, cons_loss = self.get_loss_newnewcon_shuffle(vals, data, ipm_step, step_weight)
            
            train_losses += loss.detach() * data.num_graphs
            primal_losses += primal_loss.detach() * data.num_graphs
            obj_losses += obj_loss.detach() * data.num_graphs
            cons_losses += cons_loss.detach() * data.num_graphs
            
            num_graphs += data.num_graphs
            update_count += 1
            loss = loss / float(loss_scaling_lst[0])  # scale the loss
            loss.backward()

            if update_count >= micro_batch or i == len(all_train_data) - 1:
                torch.nn.utils.clip_grad_norm_(model_now.parameters(),
                                               max_norm=1.0,
                                               error_if_nonfinite=False)
                optimizer.step()
                optimizer.zero_grad()
                update_count = 0
                loss_scaling_lst.pop(0)
            #print('updated model_now', model_now.state_dict()['encoder.vals.0.weight'][0])
            temp_model = copy.deepcopy(model_now.state_dict())
            
        return train_losses.item() / num_graphs, primal_losses.item() / num_graphs, obj_losses.item() / num_graphs, cons_losses.item() / num_graphs, temp_model, ipm_step
    
    
    
    
    # self.get_loss_exact: loss, primal_loss, obj_loss, cons_loss, remove pad 'nan'
    # con_loss: amax, mean
    # use the exact iteration as ipm step for every data instance
    def train_exact(self, dataloader, model, optimizer):
        
        model.train()
        optimizer.zero_grad()

        update_count = 0
        micro_batch = int(min(self.micro_batch, len(dataloader)))
        loss_scaling_lst = [micro_batch] * (len(dataloader) // micro_batch) + [len(dataloader) % micro_batch]

        train_losses = 0.
        primal_losses = 0.
        obj_losses = 0.
        cons_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)    # predicted obj and con
            loss, primal_loss, obj_loss, cons_loss = self.get_loss_exact(vals, data)

            train_losses += loss.detach() * data.num_graphs
            primal_losses += primal_loss.detach() * data.num_graphs
            obj_losses += obj_loss.detach() * data.num_graphs
            cons_losses += cons_loss.detach() * data.num_graphs
            
            num_graphs += data.num_graphs

            update_count += 1
            loss = loss / float(loss_scaling_lst[0])  # scale the loss
            loss.backward()

            if update_count >= micro_batch or i == len(dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=1.0,
                                               error_if_nonfinite=False)
                optimizer.step()
                optimizer.zero_grad()
                update_count = 0
                loss_scaling_lst.pop(0)

        return train_losses.item() / num_graphs, primal_losses.item() / num_graphs, obj_losses.item() / num_graphs, cons_losses.item() / num_graphs
    
    
    
    
    
    # self.get_loss_newnewcon: loss, primal_loss, obj_loss, cons_loss
    # con_loss: amax, mean
    # add_layer: avg model parameter
    def train_newnewcon_add_layer_(self, ipm, dataloader, model, optimizer):
        
        train_losses = 0.
        primal_losses = 0.
        obj_losses = 0.
        cons_losses = 0.
        num_graphs = 0
        
        avg_model = copy.deepcopy(model[0])
        for p_key in model[0].state_dict():
            avg_param = 0
            for i in range(len(model)):
                avg_param += model[i].state_dict()[p_key]
            avg_param = avg_param / len(model)
            avg_model.state_dict()[p_key].copy_(avg_param)
            
        for m in range(len(model)):
            model[m].load_state_dict(avg_model.state_dict())
        
        for n in range(len(ipm)): 
            #print('ipm', ipm[n])
            model[n].train()
            optimizer.zero_grad()

            update_count = 0
            micro_batch = int(min(self.micro_batch, len(dataloader[n])))
            loss_scaling_lst = [micro_batch] * (len(dataloader[n]) // micro_batch) + [len(dataloader[n]) % micro_batch]
            #print('loss_scaling_lst', loss_scaling_lst) 
            for i, data in enumerate(dataloader[n]):
                data = data.to(self.device)
                vals, _ = model[n](data)    # predicted obj and con
                loss, primal_loss, obj_loss, cons_loss = self.get_loss_newnewcon(vals, data)

                train_losses += loss.detach() * data.num_graphs
                primal_losses += primal_loss.detach() * data.num_graphs
                obj_losses += obj_loss.detach() * data.num_graphs
                cons_losses += cons_loss.detach() * data.num_graphs

                num_graphs += data.num_graphs

                update_count += 1
                loss = loss / float(loss_scaling_lst[0])  # scale the loss
                loss.backward()

                if update_count >= micro_batch or i == len(dataloader[n]) - 1:
                    torch.nn.utils.clip_grad_norm_(model[n].parameters(),
                                                   max_norm=1.0,
                                                   error_if_nonfinite=False)
                    optimizer.step()
                    optimizer.zero_grad()
                    update_count = 0
                    loss_scaling_lst.pop(0)

        return train_losses.item() / num_graphs, primal_losses.item() / num_graphs, obj_losses.item() / num_graphs, cons_losses.item() / num_graphs
    
    

    
    # train_: return train_loss, obj_loss, con_loss of every layer of mlp, use with get_loss_()
    def train_(self, dataloader, model, optimizer):
        model.train()
        optimizer.zero_grad()

        update_count = 0
        micro_batch = int(min(self.micro_batch, len(dataloader)))
        loss_scaling_lst = [micro_batch] * (len(dataloader) // micro_batch) + [len(dataloader) % micro_batch]

        train_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)    # predicted obj and con
            loss, layer_obj_loss, layer_con_loss = self.get_loss_(vals, data)

            train_losses += loss.detach() * data.num_graphs
            num_graphs += data.num_graphs

            update_count += 1
            loss = loss / float(loss_scaling_lst[0])  # scale the loss
            loss.backward()

            if update_count >= micro_batch or i == len(dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=1.0,
                                               error_if_nonfinite=True)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(),
#                                                max_norm=1.0,
#                                                error_if_nonfinite=False)
                optimizer.step()
                optimizer.zero_grad()
                update_count = 0
                loss_scaling_lst.pop(0)

        return train_losses.item() / num_graphs, layer_obj_loss, layer_con_loss
    

    @torch.no_grad()
    def eval(self, dataloader, model, scheduler = None):
        model.eval()

        val_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)
            loss = self.get_loss(vals, data)
            val_losses += loss * data.num_graphs
            num_graphs += data.num_graphs
        val_loss = val_losses.item() / num_graphs

        if scheduler is not None:
            scheduler.step(val_loss)
        return val_loss

    
    def get_loss(self, vals, data):
        loss = 0.

        if 'obj' in self.loss_target:
            pred = vals[:, -self.ipm_steps:]
            c_times_x = data.obj_const[:, None] * pred
            obj_pred = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')
            obj_pred = (self.loss_func(obj_pred) * self.step_weight).mean()
            loss = loss + obj_pred
            #print('obj', obj_pred)
        if 'barrier' in self.loss_target:
            raise NotImplementedError("Need to discuss only on the last step or on all")
            # pred = vals * self.std + self.mean
            # Ax = scatter(pred.squeeze()[data.A_col[data.A_tilde_mask]] *
            #              data.A_val[data.A_tilde_mask],
            #              data.A_row[data.A_tilde_mask],
            #              reduce='sum', dim=0)
            # loss = loss + barrier_function(data.rhs - Ax).mean()  # b - x >= 0.
            # loss = loss + barrier_function(pred.squeeze()).mean()  # x >= 0.
            
        if 'primal' in self.loss_target:
            primal_loss = (self.loss_func(
                vals[:, -self.ipm_steps:] -
                data.gt_primals[:, -self.ipm_steps:]
            ) * self.step_weight).mean()
            loss = loss + primal_loss * self.loss_weight['primal']
            print('primal', primal_loss, primal_loss * self.loss_weight['primal'])
        if 'objgap' in self.loss_target:
            obj_loss = (self.loss_func(self.get_obj_metric(data, vals, hard_non_negative=False)) * self.step_weight).mean()
            loss = loss + obj_loss * self.loss_weight['objgap']
            print('objgap', obj_loss, obj_loss * self.loss_weight['objgap'])
        if 'constraint' in self.loss_target: 
            constraint_gap = self.get_constraint_violation(vals, data)
            cons_loss = (self.loss_func(constraint_gap) * self.step_weight).mean()
            loss = loss + cons_loss * self.loss_weight['constraint']
            print('constraint', cons_loss, cons_loss * self.loss_weight['constraint'])
        return loss
    
    
    def get_loss_new(self, vals, data):
        loss = 0.
        if 'primal' in self.loss_target:
            primal_loss = (self.loss_func(
                vals[:, -self.ipm_steps:] -
                data.gt_primals[:, -self.ipm_steps:]
            ) * self.step_weight).mean()
            loss = loss + primal_loss * self.loss_weight['primal']
        if 'objgap' in self.loss_target:
            obj_loss = (self.loss_func(self.get_obj_metric(data, vals, hard_non_negative=False)) * self.step_weight).mean()
            loss = loss + obj_loss * self.loss_weight['objgap']
        if 'constraint' in self.loss_target: 
            constraint_gap = self.get_constraint_violation(vals, data)
            cons_loss = (self.loss_func(constraint_gap) * self.step_weight).mean()
            loss = loss + cons_loss * self.loss_weight['constraint']
        return loss, primal_loss, obj_loss, cons_loss
    
    
    
    def get_loss_newcon(self, vals, data):
        loss = 0.
        if 'primal' in self.loss_target:
            primal_loss = (self.loss_func(
                vals[:, -self.ipm_steps:] -
                data.gt_primals[:, -self.ipm_steps:]
            ) * self.step_weight).mean()
            loss = loss + primal_loss * self.loss_weight['primal']
        if 'objgap' in self.loss_target:
            obj_loss = (self.loss_func(self.get_obj_metric(data, vals, hard_non_negative=False)) * self.step_weight).mean()
            loss = loss + obj_loss * self.loss_weight['objgap']
        if 'constraint' in self.loss_target: 
            constraint_gap = self.get_constraint_violation(vals, data)
            temp_loss = self.loss_func(constraint_gap) * self.step_weight   # shape [*, 24(ipm step num)]
            num_ins = temp_loss.shape[0]
            cons_loss = torch.softmax(temp_loss, dim=0) * temp_loss    # softmax result: sum of every column of temp_loss = 1
            cons_loss = cons_loss.sum(dim=0)
            cons_loss = cons_loss.mean()
            cons_loss = cons_loss/num_ins
            loss = loss + cons_loss * self.loss_weight['constraint']
        return loss, primal_loss, obj_loss, cons_loss
    
    
     
    def get_loss_newnewcon(self, vals, data):
        loss = 0.
        if 'primal' in self.loss_target:
            primal_loss = (self.loss_func(
                vals[:, -self.ipm_steps:] -
                data.gt_primals[:, -self.ipm_steps:]
            ) * self.step_weight).mean()
            loss = loss + primal_loss * self.loss_weight['primal']
        if 'objgap' in self.loss_target:
            obj_loss = (self.loss_func(self.get_obj_metric(data, vals, hard_non_negative=False)) * self.step_weight).mean()
            loss = loss + obj_loss * self.loss_weight['objgap']
        if 'constraint' in self.loss_target: 
            constraint_gap = self.get_constraint_violation(vals, data)
            temp_loss = self.loss_func(constraint_gap) * self.step_weight   # shape [*, 24(ipm step num)]
            num_ins = temp_loss.shape[0]
            cons_loss = torch.amax(temp_loss, dim=0)    # torch.max result: max of every column
            cons_loss = cons_loss.mean()
            cons_loss = cons_loss/num_ins
            loss = loss + cons_loss * self.loss_weight['constraint']
        return loss, primal_loss, obj_loss, cons_loss
    
    
    def get_loss_harp(self, vals, data):
        loss = 0.
        if 'primal' in self.loss_target:
            primal_loss = (self.loss_func(
                vals[:, -self.ipm_steps:] -
                data.gt_primals[:, -self.ipm_steps:]
            ) * self.step_weight).mean()
            loss = loss + primal_loss * self.loss_weight['primal']
        if 'objgap' in self.loss_target:
            obj_loss = (self.loss_func(self.get_obj_metric(data, vals, hard_non_negative=False))*self.step_weight).mean()
            loss = loss + obj_loss * self.loss_weight['objgap']
        if 'constraint' in self.loss_target: 
            con_gap, econ_gap = self.get_constraint_violation_harp(vals, data)
            temp_loss_con = self.loss_func(con_gap) * self.step_weight   # shape [*, 24(ipm step num)]
            temp_loss_econ = self.loss_func(econ_gap) * self.step_weight   # shape [*, 24(ipm step num)]

            cons_loss = torch.amax(temp_loss_con, dim=0)    # torch.max result: max of every column
            cons_loss = cons_loss.mean()
            cons_loss = cons_loss/(temp_loss_con.shape[0])
            
            econs_loss = torch.amax(temp_loss_econ, dim=0)    # torch.max result: max of every column
            econs_loss = econs_loss.mean()
            econs_loss = econs_loss/(temp_loss_econ.shape[0])
            
            loss = loss + cons_loss * self.loss_weight['constraint'] + econs_loss * self.loss_weight['constraint'] 
            
        return loss, primal_loss, obj_loss, cons_loss, econs_loss
    
    
    def get_loss_newnewcon_shuffle(self, vals, data, ipm_step, step_weight):
        loss = 0.
        if 'primal' in self.loss_target:
            #print('vals.shape', vals.shape)
            #print(vals[:, -ipm_step:].shape)
            #print('data.gt_primals.shape', data.gt_primals.shape)
            #print(data.gt_primals[:, -ipm_step:].shape)
            primal_loss = (self.loss_func(
                vals[:, -ipm_step:] -
                data.gt_primals[:, -ipm_step:]
            ) * step_weight).mean()
            loss = loss + primal_loss * self.loss_weight['primal']
        if 'objgap' in self.loss_target:
            obj_loss = (self.loss_func(self.get_obj_metric(data, vals, hard_non_negative=False)) * step_weight).mean()
            loss = loss + obj_loss * self.loss_weight['objgap']
        if 'constraint' in self.loss_target: 
            constraint_gap = self.get_constraint_violation(vals, data)
            temp_loss = self.loss_func(constraint_gap) * step_weight   # shape [*, 24(ipm step num)]
            num_ins = temp_loss.shape[0]
            cons_loss = torch.amax(temp_loss, dim=0)    # torch.max result: max of every column
            cons_loss = cons_loss.mean()
            cons_loss = cons_loss/num_ins
            loss = loss + cons_loss * self.loss_weight['constraint']
        return loss, primal_loss, obj_loss, cons_loss

    
#     # loss for every batch in dataloader 
#     def get_loss_exact(self, vals, data):
#         # example: 
#         # ipm = 4,   vals: [[x, x, x, x],    [x, x, x, x], ......], shape: (# of vars in one batch of data instance,ipm)  
#         # data.gt_primals: [[x, x, nan, nan],[x, x, x, nan], ......], shape: (# of vars in one batch of data instance,ipm) 
        
#         loss = 0.
        
#         exact_list = []    # remove 'nan' and get the list containing all exact values, length = data.gt_primals/vals.shape[0] 
#         exact_sw = []      # exact step weight, length = data.gt_primals/vals.shape[0] 
#         for i in range(data.gt_primals.shape[0]):
#             nonan_idx = (~torch.isnan(data.gt_primals[i])).nonzero(as_tuple=True)[0]   # get not nan position in gt_primals
#             exact_list.append(nonan_idx)
#             sw = torch.tensor([self.ipm_alpha ** (nonan_idx.shape[0] - l - 1) for l in range(nonan_idx.shape[0])],
#                                         dtype=torch.float, device=device)[None]    # step decay factor
#             exact_sw.append(sw)
#         if 'primal' in self.loss_target:
#             primal_loss = [] 
#             for i in range(vals.shape[0]):
#                 p_loss = self.loss_func(vals[i, exact_list[i]]-data.gt_primals[i, exact_list[i]])
#                 primal_loss.append(p_loss * exact_sw[i])
#             primal_loss = (torch.stack(primal_loss)).mean()
#             loss = loss + primal_loss * self.loss_weight['primal']
#         if 'objgap' in self.loss_target:
#             obj_loss = []
#             for i in range(vals.shape[0]):
#                 o_loss = self.loss_func(self.get_obj_metric_exact(i, exact_list[i], data, vals[i], hard_non_negative=False))
#                 obj_loss.append(o_loss * exact_sw[i])
#             obj_loss = (torch.stack(obj_loss)).mean()
#             loss = loss + obj_loss * self.loss_weight['objgap']
#         if 'constraint' in self.loss_target: 
#             con_loss = []
#             for i in range(vals.shape[0]):
#                 c_loss = self.loss_func(self.get_constraint_violation_exact(i, exact_list[i], vals[i], data))
#                 con_loss.append(c_loss * exact_sw[i])
#             con_loss = torch.stack(con_loss)
#             num_ins = con_loss.shape[0]
#             con_loss = ((torch.amax(con_loss, dim=0)).mean())/num_ins
#             loss = loss + cons_loss * self.loss_weight['constraint']
#         return loss, primal_loss, obj_loss, cons_loss



#     # exact_list, pred is the ith slice, data is the batch data from dataloader
#     def get_obj_metric_exact(self, i, exact_list, data, pred, hard_non_negative=False):
#         if hard_non_negative:
#             pred = torch.relu(pred[exact_list])
#         for i in range(pred.shape[0]):
#             c_times_x = data.obj_const[i, None] * pred[exact_list]             
#             obj_pred = c_times_x.sum()         
#             c_times_xgt = data.obj_const[i, None] * data.gt_primals[i, exact_list]                 
#             obj_gt = c_times_xgt.sum()
#         return (obj_pred - obj_gt) / obj_gt                         

    
#     # to be mended
#     def get_constraint_violation_exact(self, i, exact_list, pred, data):
#         pred = pred[exact_list]
#         Ax = scatter(pred[data.A_col, :] * data.A_val[:, None], data.A_row, reduce='sum', dim=0)
#         constraint_gap = Ax - data.rhs[i, None]             
#         constraint_gap = torch.relu(constraint_gap)              
#         return constraint_gap




    # loss for every batch in dataloader 
    def get_loss_exact(self, vals, data):
        # example: ipm = 4, 2 st pairs, 2 vars in each st pair
        # ipm = 4,   vals: [[x, x, x, x],    
        #                   [x, x, x, x], 
        #                   [x, x, x, x], 
        #                   [x, x, x, x],......], shape: (# of vars in one batch of data instance, ipm)  
        # data.gt_primals: [[x, x, nan, nan],
        #                   [x, x, nan, nan],
        #                   [x, x, x, nan], 
        #                   [x, x, x, nan],......], shape: (# of vars in one batch of data instance, ipm) 
        
        # chunks calulation
        # seperate data.gt_primals into different parts by checking the len(data.gt_primals[i]), the ones with the same len
        # belong to the same instance and shall be calculated together
        # instances with the same len (iteration num) can be calculated together (can include multiple instances)
        # this is very complex to implement, since A, b, c are all stacked together
        
        # try batch = 1 first
        # the only thing need to do is trime off the 'nan'
        non_nan_mask = ~torch.isnan(data.gt_primals)                                   # remove 'nan' 
        val_true = data.gt_primals[non_nan_mask].reshape(data.gt_primals.shape[0], -1) # shape: (# of vars, # of true iter)
        exact_sw = torch.tensor([self.ipm_alpha ** (val_true.shape[1] - l - 1) for l in range(val_true.shape[1])],
                                        dtype=torch.float, device=self.device)[None]    # step decay factor
        val_trimed = vals[:, 0:val_true.shape[1]]
        loss = 0. 
        if 'primal' in self.loss_target:
            primal_loss = (self.loss_func(val_trimed - val_true) * exact_sw).mean()
            loss = loss + primal_loss * self.loss_weight['primal']
        if 'objgap' in self.loss_target:
            obj_loss = (self.loss_func(self.get_obj_metric_exact(data, val_true, val_trimed, hard_non_negative=False)) * exact_sw).mean()
            loss = loss + obj_loss * self.loss_weight['objgap']
        if 'constraint' in self.loss_target: 
            constraint_gap = self.get_constraint_violation_exact(val_trimed, data)
            temp_loss = self.loss_func(constraint_gap) * exact_sw   # shape [*, 24(ipm step num)]
            num_ins = temp_loss.shape[0]
            cons_loss = torch.amax(temp_loss, dim=0)    # torch.max result: max of every column
            cons_loss = cons_loss.mean()
            cons_loss = cons_loss/num_ins
            loss = loss + cons_loss * self.loss_weight['constraint']
        return loss, primal_loss, obj_loss, cons_loss

    
    def get_obj_metric_exact(self, data, true, pred, hard_non_negative=False):
        if hard_non_negative:
            pred = torch.relu(pred)
        c_times_x = data.obj_const[:, None] * pred             
        obj_pred = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')    
        c_times_xgt = data.obj_const[:, None] * true              
        obj_gt = scatter(c_times_xgt, data['vals'].batch, dim=0, reduce='sum')
        return (obj_pred - obj_gt) / obj_gt                         
    
    
    def get_constraint_violation_exact(self, vals, data):
        """
        Ax - b

        :param vals:
        :param data:
        :return:
        """
        Ax = scatter(vals[data.A_col, :] * data.A_val[:, None], data.A_row, reduce='sum', dim=0)
        constraint_gap = Ax - data.rhs[:, None]                   # data.rhs=b
        constraint_gap = torch.relu(constraint_gap)               # ax-b > 0, gap>0, otw, gap=0
        return constraint_gap

    
    
    
    # get_loss_ return: loss, obj_loss, con_loss of every layer of the output mlp 
    def get_loss_(self, vals, data):
        loss = 0.

        if 'obj' in self.loss_target:
            pred = vals[:, -self.ipm_steps:]
            c_times_x = data.obj_const[:, None] * pred
            obj_pred = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')
            obj_pred = (self.loss_func(obj_pred) * self.step_weight).mean()
            loss = loss + obj_pred
        if 'barrier' in self.loss_target:
            raise NotImplementedError("Need to discuss only on the last step or on all")
            # pred = vals * self.std + self.mean
            # Ax = scatter(pred.squeeze()[data.A_col[data.A_tilde_mask]] *
            #              data.A_val[data.A_tilde_mask],
            #              data.A_row[data.A_tilde_mask],
            #              reduce='sum', dim=0)
            # loss = loss + barrier_function(data.rhs - Ax).mean()  # b - x >= 0.
            # loss = loss + barrier_function(pred.squeeze()).mean()  # x >= 0.
        if 'primal' in self.loss_target:
            primal_loss = (self.loss_func(
                vals[:, -self.ipm_steps:] -
                data.gt_primals[:, -self.ipm_steps:]
            ) * self.step_weight).mean()
            loss = loss + primal_loss * self.loss_weight['primal']
        if 'objgap' in self.loss_target:
            layer_obj_loss = self.loss_func(self.get_obj_metric(data, vals, hard_non_negative=False))
            obj_loss = (self.loss_func(self.get_obj_metric(data, vals, hard_non_negative=False)) * self.step_weight).mean()
            loss = loss + obj_loss * self.loss_weight['objgap']
        if 'constraint' in self.loss_target: 
            constraint_gap, b_vals = self.get_constraint_violation_(vals, data)
            layer_con_loss = self.loss_func(constraint_gap)
            cons_loss = (self.loss_func(constraint_gap) * self.step_weight).mean()
            loss = loss + cons_loss * self.loss_weight['constraint']
        return loss, constraint_gap, b_vals
    

    def get_constraint_violation(self, vals, data):
        """
        Ax - b

        :param vals:
        :param data:
        :return:
        """
        pred = vals[:, -self.ipm_steps:]
        Ax = scatter(pred[data.A_col, :] * data.A_val[:, None], data.A_row, reduce='sum', dim=0)
        constraint_gap = Ax - data.rhs[:, None]                   # data.rhs=b
        constraint_gap = torch.relu(constraint_gap)               # ax-b > 0, gap>0, otw, gap=0
        return constraint_gap
    
    
    def get_constraint_violation_sc(self, vals, data):
        """
        Ax - b

        :param vals:
        :param data:
        :return:
        """
        pred = vals[:, -self.ipm_steps:]
        Ax = scatter(pred[data.A_col, :] * data.A_val[:, None], data.A_row, reduce='sum', dim=0)
        con_gap = Ax - data.rhs[:, None]                   # data.rhs=b
        constraint_gap = torch.relu(con_gap)               # ax-b > 0, gap>0, otw, gap=0
        return constraint_gap, con_gap, data.rhs[:, None]
    
    
    
    def get_constraint_violation_harp_sc(self, vals, data):
        pred = vals[:, -self.ipm_steps:]
        A1x = scatter(pred[data.A1_col, :] * data.A1_val[:, None], data.A1_row, reduce='sum', dim=0)
        A2x = scatter(pred[data.A2_col, :] * data.A2_val[:, None], data.A2_row, reduce='sum', dim=0)
        con_gap = A2x - data.rhs2[:, None]                   # data.rhs=b
        con_gap = torch.relu(con_gap)               # ax-b > 0, gap>0, otw, gap=0
        econ_gap = A1x - data.rhs1[:, None]         # ax-b = 0, no gap, otw, gap > 0 or < 0 
        return con_gap, econ_gap, data.rhs1[:, None]
    
    
    
    def get_constraint_violation_harp(self, vals, data):
        """
        A1x - b1: econ
        A2x - b2: con
        """
        pred = vals[:, -self.ipm_steps:]
        A1x = scatter(pred[data.A1_col, :] * data.A1_val[:, None], data.A1_row, reduce='sum', dim=0)
        A2x = scatter(pred[data.A2_col, :] * data.A2_val[:, None], data.A2_row, reduce='sum', dim=0)
        con_gap = A2x - data.rhs2[:, None]                   # data.rhs=b
        con_gap = torch.relu(con_gap)               # ax-b > 0, gap>0, otw, gap=0
        econ_gap = A1x - data.rhs1[:, None]         # ax-b = 0, no gap, otw, gap > 0 or < 0 
        return con_gap, econ_gap
    
    
    def get_constraint_violation_(self, vals, data):
        """
        Ax - b

        :param vals:
        :param data:
        :return:
        """
        pred = vals[:, -self.ipm_steps:]
        Ax = scatter(pred[data.A_col, :] * data.A_val[:, None], data.A_row, reduce='sum', dim=0)
        constraint_gap = Ax - data.rhs[:, None]                   # data.rhs=b
        constraint_gap = torch.relu(constraint_gap)               # ax-b > 0, gap>0, otw, gap=0
        return constraint_gap, data.rhs[:, None] 

    
    def get_obj_metric(self, data, pred, hard_non_negative=False):
        # if hard_non_negative, we need a relu to make x all non-negative
        # just for metric usage, not for training
        pred = pred[:, -self.ipm_steps:]
        if hard_non_negative:
            pred = torch.relu(pred)
        c_times_x = data.obj_const[:, None] * pred                   # data.obj_const = c
        obj_pred = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')     # scatter(src, index, dim, reduce): inplace matrix operation on src using index and reduce, add all ipm step gap for each instance
        x_gt = data.gt_primals[:, -self.ipm_steps:]                  # data.gt_primals = the intermediate values of x (decision variables) in inters
        c_times_xgt = data.obj_const[:, None] * x_gt                
        obj_gt = scatter(c_times_xgt, data['vals'].batch, dim=0, reduce='sum')
        return (obj_pred - obj_gt) / obj_gt                          # this gap is a vector of ipm_steps length for every instances
    
    
    def get_obj_metric_harpleq1(self, data, pred, hard_non_negative=False):
        pred = pred[:, -self.ipm_steps:]
        if hard_non_negative:
            pred = torch.relu(pred)
        c_times_x = data.obj_const[:, None] * pred                   
        obj_pred = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')     
        x_gt = data.gt_primals[:, -self.ipm_steps:]                  
        c_times_xgt = data.obj_const[:, None] * x_gt                
        obj_gt = scatter(c_times_xgt, data['vals'].batch, dim=0, reduce='sum')
        return (obj_pred - obj_gt) / obj_gt, obj_pred/obj_gt, obj_pred
    

    def obj_metric(self, dataloader, model):
        model.eval()

        obj_gap = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)
            obj_gap.append(np.abs(self.get_obj_metric(data, vals, hard_non_negative=True).detach().cpu().numpy()))

        return np.concatenate(obj_gap, axis=0)
    
    

    def constraint_metric(self, dataloader, model):
        """
        minimize ||Ax - b||^p in case of equality constraints
         ||relu(Ax - b)||^p in case of inequality

        :param dataloader:
        :param model:
        :return:
        """
        model.eval()

        cons_gap = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)
            cons_gap.append(np.abs(self.get_constraint_violation(vals, data).detach().cpu().numpy()))

        return np.concatenate(cons_gap, axis=0)
    
    
    def constraint_metric_harp(self, dataloader, model):
        model.eval()

        cons_gap = []
        econs_gap = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, cons, econs = model(data)
            con_gap, econ_gap = self.get_constraint_violation_harp(vals, data)
            cons_gap.append(np.abs(con_gap.detach().cpu().numpy()))
            econs_gap.append(np.abs(econ_gap.detach().cpu().numpy()))

        return np.concatenate(cons_gap, axis=0), np.concatenate(econs_gap, axis=0)
    

    def constraint_metric_(self, dataloader, model):
        """
        minimize ||Ax - b||^p in case of equality constraints
         ||relu(Ax - b)||^p in case of inequality

        :param dataloader:
        :param model:
        :return:
        """
        model.eval()

        cons_gap = []
        b_vals = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)
            con_gap, b_val = self.get_constraint_violation_(vals, data)
            cons_gap.append(np.abs(congap.detach().cpu().numpy()))
            b_vals.append(b_val)

        return np.concatenate(cons_gap, axis=0), np.concatenate(b_vals, axis=0)
    

    
    @torch.no_grad()
    def eval_metrics(self, dataloader, model):
        """
        both obj and constraint gap

        :param dataloader:
        :param model:
        :return:
        """
        model.eval()

        cons_gap = []
        obj_gap = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)
            cons_gap.append(np.abs(self.get_constraint_violation(vals, data).detach().cpu().numpy()))
            obj_gap.append(np.abs(self.get_obj_metric(data, vals, hard_non_negative=True).detach().cpu().numpy()))
        
        obj_gap = np.concatenate(obj_gap, axis=0)
        cons_gap = np.concatenate(cons_gap, axis=0)
        return obj_gap, cons_gap
    
    
    
    @torch.no_grad()
    def eval_metrics_harp(self, dataloader, model):
        model.eval()

        cons_gap = []
        econs_gap = []
        obj_gap = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, c, ec = model(data)
            con, econ = self.get_constraint_violation_harp(vals, data)
            cons_gap.append(np.abs(con.detach().cpu().numpy()))
            econs_gap.append(np.abs(econ.detach().cpu().numpy()))
            obj_gap.append(np.abs(self.get_obj_metric(data, vals, hard_non_negative=True).detach().cpu().numpy()))
        
        obj_gap = np.concatenate(obj_gap, axis=0)
        cons_gap = np.concatenate(cons_gap, axis=0)
        econs_gap = np.concatenate(econs_gap, axis=0)
        return obj_gap, cons_gap, econs_gap
    

    # eval_metrics_: return normalized obj gap 
    @torch.no_grad()
    def eval_metrics_(self, dataloader, model):
        """
        both obj and constraint gap

        :param dataloader:
        :param model:
        :return:
        """
        model.eval()

        cons_gap = []
        objs_gap = []
        objs_nocgap = []
        b_vals = []
        ttt = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            
            tt = time.time()
            vals, _ = model(data)
            ttt += time.time()-tt
            
            con_gap, b_val = self.get_constraint_violation_(vals, data)
            obj_gap = self.get_obj_metric(data, vals, hard_non_negative=True)
            
            if torch.max(con_gap) > 0:
                print('before nocon', obj_gap)
                obj_nocgap = obj_gap * (1/(1+torch.max(con_gap)))
                print(torch.max(con_gap), 1/(1+torch.max(con_gap)))
                print('after nocon', obj_nocgap)
            else:
                obj_nocgap = obj_gap
                
            cons_gap.append(np.abs(con_gap.detach().cpu().numpy()))
            objs_gap.append(np.abs(obj_gap.detach().cpu().numpy()))
            objs_nocgap.append(np.abs(obj_nocgap.detach().cpu().numpy()))
        print('time used:', ttt)
            
        objs_gap = np.concatenate(objs_gap, axis=0)
        cons_gap = np.concatenate(cons_gap, axis=0)
        objs_nocgap = np.concatenate(objs_nocgap, axis=0)
        return objs_gap, cons_gap, objs_nocgap
    
    
    
    # eval_metrics_harpleq1: return obj, con, noconobj gap, normalized mlu
    # num_pair: num of st pair for every instance
    # k: k shortest path
    @torch.no_grad()
    def eval_metrics_harpleq1(self, k, num_pair, edges_num, batchsize, dataloader, model):
        model.eval()

        cons_gap = []
        objs_gap = []
        objs_nocgap = []
        objs_mlu = []
        objs_norm_mlu = []
        norm_MLUs = []
        MLUs = []
        
#         A_col = []
#         A_row = []
#         A_val = []
#         Val = []
        
        ttt = 0
        for d, data in enumerate(dataloader):
            data = data.to(self.device)
            
            tt = time.time()
            vals, _ = model(data)       
            
            con_gap, con_val, b_val = self.get_constraint_violation_sc(vals, data)
            obj_gap, obj_norm_mlu, obj_mlu = self.get_obj_metric_harpleq1(data, vals, hard_non_negative=True)

            
            if torch.max(con_gap[:, -1]) > 0:
#                 print('before nocon', obj_gap)
                obj_nocgap = obj_gap * (1/(1+torch.max(con_gap[:, -1])))
#                 print(torch.max(con_gap), 1/(1+torch.max(con_gap)))
#                 print('after nocon', obj_nocgap)
            else:
                obj_nocgap = obj_gap
            
            # mlu
            # need to make the first constraint Ax - b = 0, which is Ax = - 1 in our case 
            # num of the first constraint: num_st pairs
            # num of the second constraint: num_egdes
            # vals shape: (batchsize * num_vars, ipm_step), num_vars = num_st pairs * k(shortest path) + 1
            # con_gap shape: (batchsize * (num_st pairs + num_egdes), ipm_step)
            # data.rhs.shape:  (batchsize * (num_st pairs + num_egdes), 1)
            num_1stcon = num_pair
            num_2ndcon = edges_num
            num_con = num_1stcon+num_2ndcon
            #print('num_1stcon, num_2ndcon:', num_1stcon, num_2ndcon)
            norm_mlus = []
            mlus = []
            a_row_idx = []
            a_col_idx = []
            val_ = vals[:, -1]
            for i in range(int(con_gap.shape[0]/num_con)):
                # Ax'-b = con_val, x' = (con_val+b)/A, in order to enforce Ax = b, x = b/A, x = x'*(b/(con_val+b))
                con_val_i = con_val[i*num_con: i*num_con+num_1stcon][:, -1]
                b_val_i = b_val[i*num_con: i*num_con+num_1stcon]
                mask = torch.where(con_val_i != 0)  
                mask_ = torch.where(con_val_i == 0)  # when no violation
        
                sc = torch.div(b_val_i[mask], (con_val_i[mask].reshape(b_val_i[mask].shape)+b_val_i[mask]))
                sc[torch.isinf(sc)] = 1
                
                vals_i = val_[i*(k*num_pair+1):(i+1)*(k*num_pair+1)]
                
                for j in range(num_pair):    # num_pair=sc.shape[0]
                    if torch.all(vals_i[j*k:(j+1)*k] == 0):
                        vals_i[j*k:(j+1)*k] = torch.softmax(torch.ones(vals_i[j*k:(j+1)*k].shape), dim=0)
                    else:
                        vals_i[j*k:(j+1)*k] = sc[j] * vals_i[j*k:(j+1)*k] 
                
                val_[i*(k*num_pair+1):(i+1)*(k*num_pair+1)] = vals_i
                val_[-1] = 0
                
                a_row_idx.append([c for c in range((i+1)*(num_con)-num_2ndcon, (i+1)*num_con)])
                
            a_row_idx = torch.tensor(a_row_idx).reshape(-1)
            a_col_idx = torch.tensor(a_col_idx).reshape(-1) 
            # get rid of the first con and the last col of the second con for this instance
            keep_2ndcon = torch.cat([torch.where(data.A_row==idx)[0] for idx in a_row_idx])
            a_row = data.A_row[keep_2ndcon]
            a_col = data.A_col[keep_2ndcon]
            a_val = data.A_val[keep_2ndcon]

            mlu = scatter(val_[a_col] * a_val, a_row, reduce='sum', dim=0)
            mlu = torch.tensor([torch.max(mlu[int(i*num_con): int((i+1)*num_con)]) for i in range(int(con_gap.shape[0]/num_con))])
            print('MLU', mlu)
            x_gt = data.gt_primals[:, -1]   
            c_times_xgt = torch.mul(data.obj_const, x_gt)       
            obj_gt = scatter(c_times_xgt, data['vals'].batch, dim=0, reduce='sum')
            print('OPT', obj_gt)
            norm_mlu = mlu.to('cuda')/obj_gt
            print('norm MLU', norm_mlu)
            print('---------------------')
            ttt += time.time()-tt   
            cons_gap.append(np.abs(con_gap.detach().cpu().numpy()))
            objs_gap.append(np.abs(obj_gap.detach().cpu().numpy()))
            objs_nocgap.append(np.abs(obj_nocgap.detach().cpu().numpy()))
            objs_norm_mlu.append(obj_norm_mlu.detach().cpu().numpy())
            objs_mlu.append(obj_mlu.detach().cpu().numpy())
            
            norm_MLUs.append(norm_mlu.detach().cpu().numpy())
            MLUs.append(mlu.detach().cpu().numpy())
            
#             A_row.append(a_row.detach().cpu().numpy())
#             A_col.append(a_col.detach().cpu().numpy())
#             A_val.append(a_val.detach().cpu().numpy())
#             Val.append(vals[:, -1].detach().cpu().numpy())
            
            
        print('time used:', ttt)
            
        objs_gap = np.concatenate(objs_gap, axis=0)
        cons_gap = np.concatenate(cons_gap, axis=0)
        objs_nocgap = np.concatenate(objs_nocgap, axis=0)
        objs_norm_mlu = np.concatenate(objs_norm_mlu, axis=0)
        objs_mlu = np.concatenate(objs_mlu, axis=0)
        
        norm_MLUs = np.concatenate(norm_MLUs, axis=0)
        MLUs = np.concatenate(MLUs, axis=0)
        
#         A_row = np.concatenate(A_row, axis=0)
#         A_col = np.concatenate(A_col, axis=0)
#         A_val = np.concatenate(A_val, axis=0)
#         Val = np.concatenate(Val, axis=0)
        
        return objs_gap, cons_gap, objs_nocgap, objs_norm_mlu, objs_mlu, norm_MLUs, MLUs
    
    
    
    
    # eval_metrics_harpecon: return obj, con, noconobj gap, normalized mlu
    # num_pair: num of st pair for every instance
    # k: k shortest path
    @torch.no_grad()
    def eval_metrics_harpecon(self, k, num_pair, edges_num, batchsize, dataloader, model):
        model.eval()

        cons_gap = []
        objs_gap = []
        objs_nocgap = []
        objs_mlu = []
        objs_norm_mlu = []
        norm_MLUs = []
        MLUs = []
        
        ttt = 0
        for d, data in enumerate(dataloader):
            print(d)
            data = data.to(self.device)
            
            tt = time.time()
            vals, c, ec = model(data)       
            
            con_gap, econ_gap, econ_b_val = self.get_constraint_violation_harp_sc(vals, data)    # econ_gap is the gap val 
            obj_gap, obj_norm_mlu, obj_mlu = self.get_obj_metric_harpleq1(data, vals, hard_non_negative=True)
            #print('con_gap, econ_gap, econ_b_val', con_gap.shape, econ_gap.shape, econ_b_val.shape)
            #print('obj_gap, obj_norm_mlu, obj_mlu', obj_gap.shape, obj_norm_mlu.shape, obj_mlu.shape)
            '''
            print('econ_gap', econ_gap)
            print('con_gap', con_gap)
            print('econ_b_val', econ_b_val)
            print('obj_gap', obj_gap)
            print('obj_norm_mlu', obj_norm_mlu)
            print('obj_mlu', obj_mlu)
            '''
            
            if torch.max(con_gap[:, -1]) > 0:
#                 print('before nocon', obj_gap)
                obj_nocgap = obj_gap * (1/(1+torch.max(con_gap[:, -1])))
#                 print(torch.max(con_gap), 1/(1+torch.max(con_gap)))
#                 print('after nocon', obj_nocgap)
            else:
                obj_nocgap = obj_gap
            
            # mlu
            # need to make the first constraint Ax - b = 0, which is Ax = - 1 in our case 
            # num of the first constraint: num_st pairs
            # num of the second constraint: num_egdes
            # vals shape: (batchsize * num_vars, ipm_step), num_vars = num_st pairs * k(shortest path) + 1
            # con_gap shape: (batchsize * (num_st pairs + num_egdes), ipm_step)
            # data.rhs.shape:  (batchsize * (num_st pairs + num_egdes), 1)
            num_1stcon = num_pair
            num_2ndcon = edges_num
            num_con = num_1stcon+num_2ndcon
            #print('num_1stcon, num_2ndcon:', num_1stcon, num_2ndcon)
            norm_mlus = []
            mlus = []
            a_row_idx = []
            a_col_idx = []
            #print('vals', vals.shape)
            val_ = vals[:, -1]    # last ipm step
            #print('val_', val_.shape)
            #print('check two con seperately:', con_gap.shape[0]/num_2ndcon, econ_gap.shape[0]/num_1stcon)
            
            for i in range(int(econ_gap.shape[0]/num_1stcon)):       # number of instance in this batch, <= batchsize
                # prepare for scale back the vals using the equality constraint
                # Ax'-b = econ_gap, x' = (econ_gap+b)/A, in order to enforce Ax = b, x = b/A, x = x'*(b/(econ_gap+b))
                econ_val_i = econ_gap[i*num_1stcon: (i+1)*num_1stcon][:, -1]
                econ_b_val_i = econ_b_val[i*num_1stcon: (i+1)*num_1stcon]
                mask = torch.where(econ_val_i != 0)  
                #print('mask', mask)
                
                # scaling back...
                sc = torch.ones(econ_b_val_i.shape).to('cuda')
                sc[mask] = torch.div(econ_b_val_i[mask], (econ_val_i[mask].reshape(econ_b_val_i[mask].shape)+econ_b_val_i[mask]))
                sc[torch.isinf(sc)] = 1
                vals_i = val_[i*(k*num_pair+1):(i+1)*(k*num_pair+1)]
                # scale back based one every pair with the variable number = k 
                for j in range(num_pair):    # num_pair=sc.shape[0], this is also the number of the first constraint
                    if torch.all(vals_i[j*k:(j+1)*k] == 0):
                        vals_i[j*k:(j+1)*k] = torch.softmax(torch.ones(vals_i[j*k:(j+1)*k].shape), dim=0)
                    else:
                        vals_i[j*k:(j+1)*k] = sc[j] * vals_i[j*k:(j+1)*k] 
                
                val_[i*(k*num_pair+1):(i+1)*(k*num_pair+1)] = vals_i
                val_[-1] = 0    # get rid of the last col of the second con for this instance
            
            # mlu calculation
            mlu = scatter(val_[data.A2_col] * data.A2_val, data.A2_row, reduce='sum', dim=0)
            mlu = torch.tensor([torch.max(mlu[int(i*num_2ndcon): int((i+1)*num_2ndcon)]) for i in range(int(con_gap.shape[0]/num_2ndcon))])
            print('MLU', mlu)
            x_gt = data.gt_primals[:, -1]   
            c_times_xgt = torch.mul(data.obj_const, x_gt)       
            obj_gt = scatter(c_times_xgt, data['vals'].batch, dim=0, reduce='sum')
            print('OPT', obj_gt)
            norm_mlu = mlu.to('cuda')/obj_gt
            print('norm MLU', norm_mlu)
            print('---------------------')
            ttt += time.time()-tt   
            cons_gap.append(np.abs(con_gap.detach().cpu().numpy()))
            objs_gap.append(np.abs(obj_gap.detach().cpu().numpy()))
            objs_nocgap.append(np.abs(obj_nocgap.detach().cpu().numpy()))
            objs_norm_mlu.append(obj_norm_mlu.detach().cpu().numpy())
            objs_mlu.append(obj_mlu.detach().cpu().numpy())
            
            norm_MLUs.append(norm_mlu.detach().cpu().numpy())
            MLUs.append(mlu.detach().cpu().numpy())
            
            '''
            if d == 0:
                break
            '''
        print('time used:', ttt)
            
        objs_gap = np.concatenate(objs_gap, axis=0)
        cons_gap = np.concatenate(cons_gap, axis=0)
        objs_nocgap = np.concatenate(objs_nocgap, axis=0)
        objs_norm_mlu = np.concatenate(objs_norm_mlu, axis=0)
        objs_mlu = np.concatenate(objs_mlu, axis=0)
        
        norm_MLUs = np.concatenate(norm_MLUs, axis=0)
        MLUs = np.concatenate(MLUs, axis=0)

        
        return objs_gap, cons_gap, objs_nocgap, objs_norm_mlu, objs_mlu, norm_MLUs, MLUs
    
    
    
    @torch.no_grad()
    def eval_metrics_new(self, dataloader, model):
        """
        both obj and constraint gap

        :param dataloader:
        :param model:
        :return:
        """
        model.eval()

        cons_gap = []
        objs_gap = []
        objs_nocon_gap = []
        objs_nocon_gap_noabs = []
        objs_gap_noabs = []

        ttt = 0
        idx = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            
            tt = time.time()
            vals, _ = model(data)
            ttt += time.time()-tt
            
            con_gap = self.get_constraint_violation(vals, data)
            print('min vals', torch.min(vals))
            obj_gap = self.get_obj_metric(data, vals, hard_non_negative=True)
            print(con_gap.shape, obj_gap.shape)
              
            if torch.max(con_gap) > 0:
                obj_nocon_gap = obj_gap * (1/(1+torch.max(con_gap[:, -1])))
                idx.append(torch.argmax(con_gap[:, -1]))
                print('nocon_val', torch.max(con_gap[:, -1]), 1/(1+torch.max(con_gap[:, -1])), torch.argmax(con_gap[:, -1]))
            
            if torch.max(con_gap) > 0:
                vals_nocon = vals * (1/(1+torch.max(con_gap[:, -1])))
                obj_new_vals = self.get_obj_metric(data, vals_nocon, hard_non_negative=True)
                con_new_vals = self.get_constraint_violation(vals_nocon, data)
#                 print('obj_new_vals', obj_new_vals)
                print('con_new_vals', torch.max(con_new_vals))
            #print(con_gap.shape, obj_gap.shape, obj_nocon_gap.shape)
            
            cons_gap.append(np.abs(con_gap.detach().cpu().numpy()))
            objs_gap.append(np.abs(obj_gap.detach().cpu().numpy()))
            objs_nocon_gap.append(np.abs(obj_nocon_gap.detach().cpu().numpy()))
            
            objs_nocon_gap_noabs.append(obj_nocon_gap.detach().cpu().numpy())
            objs_gap_noabs.append(obj_gap.detach().cpu().numpy())
            
        print('time used:', ttt)
            
        objs_gap = np.concatenate(objs_gap, axis=0)
        objs_nocon_gap = np.concatenate(objs_nocon_gap, axis=0)
        cons_gap = np.concatenate(cons_gap, axis=0)
        
        objs_gap_noabs = np.concatenate(objs_gap_noabs, axis=0)
        objs_nocon_gap_noabs = np.concatenate(objs_nocon_gap_noabs, axis=0)
        

        return objs_gap, objs_nocon_gap, cons_gap, objs_gap_noabs, objs_nocon_gap_noabs, idx
    
    
    
    @torch.no_grad()
    def eval_metrics_newsc(self, dataloader, model):
        """
        both obj and constraint gap

        :param dataloader:
        :param model:
        :return:
        """
        model.eval()

        cons_gap = []
        objs_gap = []
        objs_nocon_gap = []
        cons_nocon_gap = []
        objs_nocon_gap_noabs = []
        objs_gap_noabs = []
        
        ttt = 0
        idxs = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            
            tt = time.time()
            vals, _ = model(data)
            ttt += time.time()-tt
            
            con_gap = self.get_constraint_violation(vals, data)
            obj_gap = self.get_obj_metric(data, vals, hard_non_negative=True)
            print('con_gap, obj_gap, vals shape', con_gap.shape, obj_gap.shape, vals.shape)
           
        
            # scale back by its own violation
            value, idx = (con_gap[:, -1]).sort(descending=True)     # descending
            nozero = torch.nonzero(value)
            nozero_val = value[nozero]
            nozero_idx = idx[nozero]
            print('non0 value, idx', nozero_val, nozero_idx)

            no_repeat = []
            for v in range(nozero_val.shape[0]):
                scale_back_idx = data['A_col'][torch.where(data['A_row']==nozero_idx[v])]
                for sb in scale_back_idx:
                    if sb in no_repeat:
                        print('pass')
                        pass
                    else:
                        print('vals orin', vals[sb, -1]) 
                        vals[sb, -1] = vals[sb, -1] * (1/(1+nozero_val[v]))
                        print('vals after sb', vals[sb, -1]) 
                    no_repeat.append(sb)
                    

            #print(no_repeat)
            
            
            con_nocon_gap = self.get_constraint_violation(vals, data)
            obj_nocon_gap = self.get_obj_metric(data, vals, hard_non_negative=True)
            
            cons_gap.append(np.abs(con_gap.detach().cpu().numpy()))
            objs_gap.append(np.abs(obj_gap.detach().cpu().numpy()))
            objs_nocon_gap.append(np.abs(obj_nocon_gap.detach().cpu().numpy()))
            cons_nocon_gap.append(np.abs(con_nocon_gap.detach().cpu().numpy()))
            
            objs_nocon_gap_noabs.append(obj_nocon_gap.detach().cpu().numpy())
            objs_gap_noabs.append(obj_gap.detach().cpu().numpy())
            if i == 9: 
                break
            
        print('time used:', ttt)
            
        objs_gap = np.concatenate(objs_gap, axis=0)
        objs_nocon_gap = np.concatenate(objs_nocon_gap, axis=0)
        cons_gap = np.concatenate(cons_gap, axis=0)
        cons_nocon_gap = np.concatenate(cons_nocon_gap, axis=0)
        
        objs_gap_noabs = np.concatenate(objs_gap_noabs, axis=0)
        objs_nocon_gap_noabs = np.concatenate(objs_nocon_gap_noabs, axis=0)
        

        return objs_gap, objs_nocon_gap, cons_gap, cons_nocon_gap, objs_gap_noabs, objs_nocon_gap_noabs, idx
    
    
    
    @torch.no_grad()
    def eval_metrics_newnewsc(self, dataloader, model):
        """
        both obj and constraint gap

        :param dataloader:
        :param model:
        :return:
        """
        model.eval()

        cons_gap = []
        objs_gap = []
        objs_nocon_gap = []
        cons_nocon_gap = []
        objs_nocon_gap_noabs = []
        objs_gap_noabs = []
        
        ttt = 0
        idxs = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            
            tt = time.time()
            vals, _ = model(data)
            ttt += time.time()-tt
            
            con_gap = self.get_constraint_violation(vals, data)
            obj_gap = self.get_obj_metric(data, vals, hard_non_negative=True)
            print('con_gap, obj_gap, vals shape', con_gap.shape, obj_gap.shape, vals.shape)
           
        
            # scale back by its own violation
            value, idx = (con_gap[:, -1]).sort(descending=True)     # descending
            nozero = torch.nonzero(value)
            nozero_val = value[nozero]
            nozero_idx = idx[nozero]
            print('non0 value, idx', nozero_val, nozero_idx)
            
            no_repeat = []
            for v in range(nozero_val.shape[0]):
                scale_back_idx = data['A_col'][torch.where(data['A_row']==nozero_idx[v])]
                for sb in scale_back_idx:
                    if sb in no_repeat:
                        print('pass')
                        pass
                    else:
                        vals[sb, -1] = vals[sb, -1] * (1/(1+nozero_val[v]))
                    no_repeat.append(sb)
            #print(no_repeat)
            
            
            con_nocon_gap = self.get_constraint_violation(vals, data)
            obj_nocon_gap = self.get_obj_metric(data, vals, hard_non_negative=True)
            print('check if 0 in con', con_nocon_gap[:, -1][torch.nonzero(con_nocon_gap[:, -1])])
            
            cons_gap.append(np.abs(con_gap.detach().cpu().numpy()))
            objs_gap.append(np.abs(obj_gap.detach().cpu().numpy()))
            objs_nocon_gap.append(np.abs(obj_nocon_gap.detach().cpu().numpy()))
            cons_nocon_gap.append(np.abs(con_nocon_gap.detach().cpu().numpy()))
            
            objs_nocon_gap_noabs.append(obj_nocon_gap.detach().cpu().numpy())
            objs_gap_noabs.append(obj_gap.detach().cpu().numpy())
            
        print('time used:', ttt)
            
        objs_gap = np.concatenate(objs_gap, axis=0)
        objs_nocon_gap = np.concatenate(objs_nocon_gap, axis=0)
        cons_gap = np.concatenate(cons_gap, axis=0)
        cons_nocon_gap = np.concatenate(cons_nocon_gap, axis=0)
        
        objs_gap_noabs = np.concatenate(objs_gap_noabs, axis=0)
        objs_nocon_gap_noabs = np.concatenate(objs_nocon_gap_noabs, axis=0)
        

        return objs_gap, objs_nocon_gap, cons_gap, cons_nocon_gap, objs_gap_noabs, objs_nocon_gap_noabs, idx
   


    
    @torch.no_grad()
    def eval_metrics_exact(self, dataloader, model):
        """
        both obj and constraint gap

        :param dataloader:
        :param model:
        :return:
        """
        model.eval()
        
        cons_gap = []
        objs_gap = []
        objs_nocon_gap = []
        cons_nocon_gap = []
        objs_nocon_gap_noabs = []
        objs_gap_noabs = []
        
        ttt = 0
        tttt = 0
        idxs = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            
            # pure tesing time
            tt = time.time()
            vals, _ = model(data)
            ttt += time.time()-tt
            
            non_nan_mask = ~torch.isnan(data.gt_primals)                                   # remove 'nan' 
            val_true = data.gt_primals[non_nan_mask].reshape(data.gt_primals.shape[0], -1) # shape: (# of vars, # of true iter)
            val_trimed = vals[:, 0:val_true.shape[1]]
            
            
            con_gap = self.get_constraint_violation_exact(val_trimed, data)
            obj_gap = self.get_obj_metric_exact(data, val_true, val_trimed, hard_non_negative=True)
            print('--------------------------------------')
            print('con_gap, obj_gap, vals shape', con_gap.shape, obj_gap.shape, val_trimed.shape)
           
        
            # scale back by its own violation
            value, idx = (con_gap[:, -1]).sort(descending=True)     # descending
            nozero = torch.nonzero(value)
            nozero_val = value[nozero]
            nozero_idx = idx[nozero]
            print('non0 value, idx:', nozero_val, nozero_idx)

            no_repeat = []
            if nozero.numel() == 0:
                print('No need for sc')
                con_nocon_gap = con_gap
                obj_nocon_gap = obj_gap
            else: 
                for v in range(nozero_val.shape[0]):
                    scale_back_idx = data['A_col'][torch.where(data['A_row']==nozero_idx[v])]
                    for sb in scale_back_idx:
                        if sb in no_repeat:
                            print('pass')
                            pass
                        else:
                            val_trimed[sb, -1] = val_trimed[sb, -1] * (1/(1+nozero_val[v]))
                        no_repeat.append(sb)
                con_nocon_gap = self.get_constraint_violation_exact(val_trimed, data)
                obj_nocon_gap = self.get_obj_metric_exact(data, val_true, val_trimed, hard_non_negative=True)
                print('check if 0 in con', con_nocon_gap[:, -1][torch.nonzero(con_nocon_gap[:, -1])])
            # testing time + scale back        
            tttt += time.time()-tt
            
            cons_gap.append(np.abs(con_gap.detach().cpu().numpy()))
            objs_gap.append(np.abs(obj_gap.detach().cpu().numpy()))
            objs_nocon_gap.append(np.abs(obj_nocon_gap.detach().cpu().numpy()))
            cons_nocon_gap.append(np.abs(con_nocon_gap.detach().cpu().numpy()))
            
            objs_nocon_gap_noabs.append(obj_nocon_gap.detach().cpu().numpy())
            objs_gap_noabs.append(obj_gap.detach().cpu().numpy())
            
            if i == 9:
                break
            
        print('tesing time used:', ttt)
        print('tesing time and scale back used:', ttt)
            
#         objs_gap = np.concatenate(objs_gap, axis=0)
#         objs_nocon_gap = np.concatenate(objs_nocon_gap, axis=0)
#         cons_gap = np.concatenate(cons_gap, axis=0)
#         cons_nocon_gap = np.concatenate(cons_nocon_gap, axis=0)
        
#         objs_gap_noabs = np.concatenate(objs_gap_noabs, axis=0)
#         objs_nocon_gap_noabs = np.concatenate(objs_nocon_gap_noabs, axis=0)
        

        return objs_gap, objs_nocon_gap, cons_gap, cons_nocon_gap, objs_gap_noabs, objs_nocon_gap_noabs
    
    
    
    

    @torch.no_grad()
    def eval_baseline(self, dataloader, model, T):
        model.eval()

        obj_gaps = []
        constraint_gaps = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            val_con_repeats = model(torch.ones(1, dtype=torch.float, device=self.device) * T,
                                    data)

            vals, cons = torch.split(val_con_repeats,
                                     torch.hstack([data.num_val_nodes.sum(),
                                                   data.num_con_nodes.sum()]).tolist(), dim=0)

            obj_gaps.append(self.get_obj_metric(data, vals[:, None], True).abs().cpu().numpy())
            constraint_gaps.append(self.get_constraint_violation(vals[:, None], data).abs().cpu().numpy())

        obj_gaps = np.concatenate(obj_gaps, axis=0).squeeze()
        constraint_gaps = np.concatenate(constraint_gaps, axis=0).squeeze()

        return obj_gaps, constraint_gaps
