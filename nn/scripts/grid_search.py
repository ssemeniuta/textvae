import argparse
import subprocess
import os
import os.path
import time
import json
import os
import os.path


class GridSearch(object):

    def __init__(self, config="grid.json"):
        if os.path.exists("exp/gridlogs") is False:
            os.makedirs("exp/gridlogs")
        with open(config) as f:
            s = f.read()
        conf = json.JSONDecoder().decode(s)
        self.filename = conf["filename"]
        self.gpus = conf["gpus"]
        self.prefix = conf["prefix"] if "prefix" in conf else "-"
        self.processes = [None for _ in xrange(len(self.gpus))]
        self.runs = self.get_runs(conf["grid"])
        self.launched_runs = 0
        self.finished_runs = 0
        print "%d total runs on gpus %s:" % (len(self.runs), self.gpus)
        for i in xrange(len(self.runs)):
            print self.runs[i]

    def get_runs(self, grid, idx=0):
        param = grid[idx]
        runs = []
        for i in xrange(len(param["values"])):
            run = {param["name"]: param["values"][i]}
            if idx < len(grid) - 1:
                inner_runs = self.get_runs(grid, idx+1)
                for ir in inner_runs:
                    ir.update(run)
                    runs.append(ir)
            else:
                runs.append(run)
        return runs

    def run(self):
        while self.finished_runs < len(self.runs):
            idx = self.wait()
            if self.launched_runs < len(self.runs):
                self.launch_process(idx, self.runs[self.launched_runs])
                self.launched_runs += 1

    def wait(self):
        while True:
            for i in xrange(len(self.gpus)):
                if self.processes[i] is None:
                    if self.launched_runs < len(self.runs):
                        return i
                    continue
                if self.processes[i].poll() is not None:
                    self.processes[i] = None
                    self.finished_runs += 1
                    print "%d runs finished" % self.finished_runs
                    return i
            time.sleep(10)

    def launch_process(self, idx, params):
        assert self.processes[idx] is None
        env = os.environ.copy()
        env["THEANO_FLAGS"] = "device=gpu%d" % self.gpus[idx]
        command = self.make_command(params)
        name = ""
        for p in params:
            name += "%s_%s." % (p, params[p])
        name = name[:-1]
        log = open("exp/gridlogs/%s.txt" % name, "w")
        p = subprocess.Popen(command, env=env, stderr=log, stdout=log)
        self.processes[idx] = p

    def make_command(self, params):
        command = ["python", "-u", self.filename]
        for p in params:
            command.append("%s%s" % (self.prefix, p))
            command.append(str(params[p]))
        print command
        return command


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-grid", default="defs/grid.json")
    args = parser.parse_args()
    grid = GridSearch(args.grid)
    grid.run()
