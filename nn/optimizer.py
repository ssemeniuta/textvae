import theano
import theano.tensor as T
import numpy
import time
import datetime
import os
import shutil
import utils


class Optimizer:
    def __init__(self, model, train_db, valid_db, param_updates, grad_clip=None, disconnected_inputs='raise',
                 restore=False, folder_per_exp=False, name=None, print_info=False):
        self.model = model
        self.train_db = train_db
        self.valid_db = valid_db
        self.param_updates = param_updates
        self.params = model.params
        self.print_info = print_info
        self.restore = restore

        self.name = name
        self.train_log_f = None
        self.valid_log_f = None
        self.info_f = None
        self.opt_folder = None
        self.make_files(folder_per_exp)

        train_costs, train_updates = model.build(train=True)

        self.orig_costs_len = len(train_costs)

        print "computing grads...",
        t = time.time()
        self.grads = T.grad(train_costs[0], self.params, disconnected_inputs=disconnected_inputs)
        print "took %f seconds" % (time.time() - t)

        if print_info:
            params_l2 = T.sqrt(sum([T.sum(p ** 2) for p in self.params]))
            train_costs.append(params_l2)
            grad_l2 = T.sqrt(sum([T.sum(g ** 2) for g in self.grads]))
            train_costs.append(grad_l2)

        if grad_clip:
            self.grads = grad_clip(self.grads)
            if print_info:
                clip_grad_l2 = T.sqrt(sum([T.sum(g ** 2) for g in self.grads]))
                train_costs.append(clip_grad_l2)

        updates = param_updates(self.params, self.grads)
        updates.update(train_updates)
        try:
            if print_info:
                info = param_updates.additional_info()
                train_costs.extend(info)
        except AttributeError:
            pass
        print "compiling train fn...",
        t = time.time()
        self.train_net = theano.function([train_db.index],
                                         train_costs,
                                         givens=train_db.givens(model.input, model.target),
                                         updates=updates)
        print "took %f seconds" % (time.time() - t)

        print "compiling valid fn...",
        t = time.time()
        test_costs, test_updates = model.build(train=False)
        self.test_net = theano.function([self.valid_db.index],
                                        test_costs,
                                        givens=self.valid_db.givens(model.input, model.target),
                                        updates=test_updates)
        print "took %f seconds" % (time.time() - t)

        if restore:
            self.restore_state()

    def save_state(self, cur_epoch, total_iteration, best_cost):
        if self.name is not None:
            model_filename = "%s/model.flt" % self.opt_folder
            prev_model_filename = "%s/prev_model.flt" % self.opt_folder
            if os.path.exists(model_filename):
                shutil.move(model_filename, prev_model_filename)
            self.model.dump(model_filename)

            vars_filename = "%s/vars.flt" % self.opt_folder
            prev_vars_filename = "%s/prev_vars.flt" % self.opt_folder
            if os.path.exists(vars_filename):
                shutil.move(vars_filename, prev_vars_filename)
            utils.dump_update_vars(self.param_updates, vars_filename)

            utils.save_json("%s/opt_state.json" % self.opt_folder, {
                "cur_epoch": cur_epoch,
                "total_iteration": total_iteration,
                "best_cost": float(best_cost)
            })

    def restore_state(self):
        if self.name is not None:
            model_filename = "%s/model.flt" % self.opt_folder
            print "loading %s..." % model_filename,
            if os.path.exists(model_filename):
                self.model.load(model_filename, silent=True)
            else:
                raise Exception("can not restore model from %s, file not found" % model_filename)
            print "ok"

            vars_filename = "%s/vars.flt" % self.opt_folder
            print "loading %s..." % vars_filename,
            if os.path.exists(vars_filename):
                utils.load_update_vars(self.param_updates, vars_filename)
            else:
                raise Exception("can not restore vars from %s, file not found" % vars_filename)
            print "ok, lr = %f" % self.param_updates.lr.get_value()

    def make_files(self, folder_per_exp):
        if self.print_info:
            assert self.name is not None, "name should not be None if print_norms is True"
        if self.name is not None:
            if folder_per_exp:
                timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d-%H-%M-%S')
                self.opt_folder = "exp/%s/%s" % (self.name, timestamp)
            else:
                self.opt_folder = "exp/%s" % self.name

            if os.path.exists(self.opt_folder) is False:
                os.makedirs(self.opt_folder)
            print "current folder %s" % self.opt_folder

            mode = "a" if self.restore else "w"
            self.train_log_f = open("%s/log.train.csv" % self.opt_folder, mode, 0)
            self.valid_log_f = open("%s/log.valid.csv" % self.opt_folder, mode, 0)
            if self.print_info:
                self.info_f = open(self.opt_folder + "/info.csv", "w", buffering=0)

    def get_valid_costs(self, db):
        test_costs = []

        self.model.reset()
        for idx in db.indices():
            c = self.test_net(idx)
            c = c[0:self.orig_costs_len]
            test_costs.append(c)
        self.model.reset()

        return numpy.average(test_costs, axis=0)

    def print_to_log(self, phase, iteration, costs):
        logs = {
            "train": self.train_log_f,
            "valid": self.valid_log_f
        }
        if logs[phase] is not None:
            msg = '%i' % iteration
            for v in costs:
                msg += ',%f' % v
            print >>logs[phase], msg

    def train(self, epochs, output_frequency=10, validation_frequency=-1, callback=None,
              lr_decay=1, decay_after=0, decay_every=1, decay_schedule_in_iters=False):
        if lr_decay >= 1:
            lr_decay = 1. / lr_decay
        total_iteration = 0
        cur_epoch = 0
        best_cost = float('+inf')
        if decay_schedule_in_iters is False:
            decay_after *= self.train_db.total_batches()
            decay_every *= self.train_db.total_batches()
        if self.restore:
            cur_epoch, total_iteration, best_cost = utils.read_json_as_tuple("%s/opt_state.json" % self.opt_folder, ["cur_epoch", "total_iteration", "best_cost"])
        for epoch in xrange(cur_epoch, epochs):
            cost = []
            iteration = 0
            self.model.reset()
            t = time.time()
            for idx in self.train_db.indices():
                iteration += 1
                total_iteration += 1

                net_outs = self.train_net(idx)

                cost.append(net_outs[:self.orig_costs_len])

                if self.print_info:
                    msg = "%d" % total_iteration
                    for i in xrange(self.orig_costs_len, len(net_outs)):
                        msg += ",%f" % net_outs[i]
                    print >>self.info_f, msg

                done = float(iteration) / self.train_db.total_batches()
                if (iteration % output_frequency == 0) or (done == 1.):
                    cost = numpy.mean(cost, axis=0)
                    print '\repoch %03i, %f done, cost = %s, took %f sec. ' % (epoch+1, done, cost, time.time() - t),
                    self.print_to_log("train", total_iteration, cost)
                    cost = []

                if (done == 1.) or ((iteration % validation_frequency == 0) and (validation_frequency != -1)):
                    print "validation...",
                    cost = self.get_valid_costs(self.valid_db)

                    print "cost =", cost
                    self.print_to_log("valid", total_iteration, cost)

                    best = False
                    if callback is not None:
                        try:
                            best = callback(cost)
                        except TypeError:
                            best = callback()

                    if done == 1.:
                        epoch += 1
                    self.save_state(epoch, total_iteration, best_cost)
                    if best:
                        shutil.copy('%s/model.flt' % self.opt_folder, '%s/best.flt' % self.opt_folder)

                    cost = []

                if total_iteration >= decay_after:
                    if (total_iteration - decay_after) % decay_every == 0:
                        new_lr = (self.param_updates.lr.get_value() * lr_decay).astype(theano.config.floatX)
                        if lr_decay != 1.:
                            print "new learning rate = %f" % new_lr
                        self.param_updates.lr.set_value(new_lr)
