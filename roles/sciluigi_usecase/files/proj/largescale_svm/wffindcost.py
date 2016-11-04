
# coding: utf-8

# # Reproduce SciLuigi Case Study Workflow
#
# The code for this virtual machine is available [here](https://github.com/pharmbio/bioimg-sciluigi-casestudy), and the direct link to the code for this notebook is available [here](https://github.com/pharmbio/bioimg-sciluigi-casestudy/blob/master/roles/sciluigi_usecase/files/proj/largescale_svm/wffindcost.ipynb).
#
# ## How to run
#
# - To run the workflow, click: "Cell > Run All" in the menu above!
#   - Note that running the full workflow takes a long time. On a Intel i5 dual core laptop, it could take up to ~45 minutes to finish.
# - For a visualization of the progress, see the "Luigi Task Visualizer" browser tab.
#   - If it is not open, you can also access it via [this link](http://localhost:8082/static/visualiser/index.html#)
# - To see the dependency graph for the CrossValidate workflow in the Luigi Task Visualizer interface:
#   - Click the "CrossValidateWorkflow" task in the left menu
#   - Then click the little blue dependency graph icon in the "Actions" column, to the far right on the page.
#   - **Note:** You might need to refresh the browser (Ctrl + R, or Cmd + R), to see the latest state of the workflow.
#
# ### Caveats
#
# - Please note that in order to re-run the workflow full, you need to go "Kernel > Restart", or "Kernel > Restart & Clear all output".
#   - This is because of how Luigi works, by defining tasks as classes, and that if reloading a cell in Jupyter, that would mean re-declaring an already declared class.
#   - Alternatively, one can just re-start any cells containing only execution code (no class definitions).
#
#
# ## Set up imports and logging

# In[ ]:

from cheminf_components import *
import logging
import luigi
import sciluigi
import time

log = logging.getLogger('sciluigi-interface')
log.setLevel(logging.WARN) # So as not to flood the Jupyter cells with output for the large number of tasks


# ## Define the workflow
#
# The workflow definition is defined in a subclass of `sciluigi.WorkflowTask`, in the `workflow()` special function, which returns the most downstream task in the workflow.

# In[ ]:

class CrossValidateWorkflow(sciluigi.WorkflowTask):
    '''
    Find the optimal SVM cost values via a grid-search, with cross-validation
    '''

    # PARAMETERS
    dataset_name = luigi.Parameter(default='testrun_dataset')
    run_id = luigi.Parameter('test_run_001')
    replicate_id = luigi.Parameter('')
    replicate_ids = luigi.Parameter(default='r1,r2,r3')
    folds_count = luigi.IntParameter(default=10)
    min_height = luigi.Parameter(default='1')
    max_height = luigi.Parameter(default='3')
    test_size = luigi.Parameter('1000')
    train_sizes = luigi.Parameter(default='500,1000,2000,4000,8000')
    lin_type = luigi.Parameter(default='12') # 12, See: https://www.csie.ntu.edu.tw/~cjlin/liblinear/FAQ.html
    randomdatasize_mb = luigi.IntParameter(default='10')
    runmode = 'local'
    slurm_project = 'N/A'

    def workflow(self):
        if self.runmode == 'local':
            runmode = sciluigi.RUNMODE_LOCAL
        elif self.runmode == 'hpc':
            runmode = sciluigi.RUNMODE_HPC
        elif self.runmode == 'mpi':
            runmode = sciluigi.RUNMODE_MPI
        else:
            raise Exception('Runmode is none of local, hpc, nor mpi. Please fix and try again!')

        # ----------------------------------------------------------------
        mmtestdata = self.new_task('mmtestdata', ExistingSmiles,
                replicate_id='na',
                dataset_name=self.dataset_name)
        tasks = {}
        lowest_rmsds = []
        mainwfruns = []
        if self.replicate_id != '':
            replicate_ids = [self.replicate_id]
        else:
            replicate_ids = [i for i in self.replicate_ids.split(',')]
        for replicate_id in replicate_ids:
            tasks[replicate_id] = {}
            gensign = self.new_task('gensign_%s' % replicate_id, GenerateSignaturesFilterSubstances,
                    replicate_id=replicate_id,
                    min_height = self.min_height,
                    max_height = self.max_height,
                    slurminfo = sciluigi.SlurmInfo(
                        runmode=runmode,
                        project=self.slurm_project,
                        partition='core',
                        cores='8',
                        time='1:00:00',
                        jobname='mmgensign',
                        threads='8'
                    ))
            gensign.in_smiles = mmtestdata.out_smiles
            # ----------------------------------------------------------------
            create_unique_run_copy = self.new_task('create_unique_run_copy_%s' % self.run_id,
                    CreateRunCopy,
                    run_id = self.run_id)
            create_unique_run_copy.in_file = gensign.out_signatures
            # ----------------------------------------------------------------
            replcopy = self.new_task('replcopy_%s' % replicate_id, CreateReplicateCopy,
                    replicate_id=replicate_id)
            replcopy.in_file = create_unique_run_copy.out_copy
            # ----------------------------------------------------------------
            for train_size in [i for i in self.train_sizes.split(',')]:
                samplett = self.new_task('sampletraintest_%s_%s' % (train_size, replicate_id), SampleTrainAndTest,
                        replicate_id=replicate_id,
                        sampling_method='random',
                        test_size=self.test_size,
                        train_size=train_size,
                        slurminfo = sciluigi.SlurmInfo(
                            runmode=runmode,
                            project='b2013262',
                            partition='core',
                            cores='2',
                            time='1:00:00',
                            jobname='mmsampletraintest_%s_%s' % (train_size, replicate_id),
                            threads='1'
                        ))
                samplett.in_signatures = replcopy.out_copy
                # ----------------------------------------------------------------
                sprstrain = self.new_task('sparsetrain_%s_%s' % (train_size, replicate_id), CreateSparseTrainDataset,
                        replicate_id=replicate_id,
                        slurminfo = sciluigi.SlurmInfo(
                            runmode=runmode,
                            project=self.slurm_project,
                            partition='core',
                            cores='8',
                            time='1-00:00:00', # Took ~16hrs for acd_logd, size: rest(train) - 50000(test)
                            jobname='mmsparsetrain_%s_%s' % (train_size, replicate_id),
                            threads='8'
                        ))
                sprstrain.in_traindata = samplett.out_traindata
                # ----------------------------------------------------------------
                gunzip = self.new_task('gunzip_sparsetrain_%s_%s' % (train_size, replicate_id), UnGzipFile,
                        slurminfo = sciluigi.SlurmInfo(
                            runmode=runmode,
                            project=self.slurm_project,
                            partition='core',
                            cores='1',
                            time='1:00:00',
                            jobname='gunzip_sparsetrain_%s_%s' % (train_size, replicate_id),
                            threads='1'
                        ))
                gunzip.in_gzipped = sprstrain.out_sparse_traindata
                # ----------------------------------------------------------------
                cntlines = self.new_task('countlines_%s_%s' % (train_size, replicate_id), CountLines,
                        slurminfo = sciluigi.SlurmInfo(
                            runmode=runmode,
                            project=self.slurm_project,
                            partition='core',
                            cores='1',
                            time='15:00',
                            jobname='gunzip_sparsetrain_%s_%s' % (train_size, replicate_id),
                            threads='1'
                        ))
                cntlines.in_file = gunzip.out_ungzipped
                # ----------------------------------------------------------------
                genrandomdata= self.new_task('genrandomdata_%s_%s' % (train_size, replicate_id), CreateRandomData,
                        size_mb=self.randomdatasize_mb,
                        replicate_id=replicate_id,
                        slurminfo = sciluigi.SlurmInfo(
                            runmode=runmode,
                            project=self.slurm_project,
                            partition='core',
                            cores='1',
                            time='1:00:00',
                            jobname='genrandomdata_%s_%s' % (train_size, replicate_id),
                            threads='1'
                        ))
                genrandomdata.in_basepath = gunzip.out_ungzipped
                # ----------------------------------------------------------------
                shufflelines = self.new_task('shufflelines_%s_%s' % (train_size, replicate_id), ShuffleLines,
                        slurminfo = sciluigi.SlurmInfo(
                            runmode=runmode,
                            project=self.slurm_project,
                            partition='core',
                            cores='1',
                            time='15:00',
                            jobname='shufflelines_%s_%s' % (train_size, replicate_id),
                            threads='1'
                        ))
                shufflelines.in_randomdata = genrandomdata.out_random
                shufflelines.in_file = gunzip.out_ungzipped
                # ----------------------------------------------------------------

                costseq = ['0.0001', '0.0005', '0.001', '0.005', '0.01', '0.05', '0.1', '0.25', '0.5', '0.75', '1', '2', '3', '4', '5' ] + [str(int(10**p)) for p in xrange(1,12)]
                # Branch the workflow into one branch per fold
                for fold_idx in xrange(self.folds_count):
                    tasks[replicate_id][fold_idx] = {}
                    # Init tasks
                    create_folds = self.new_task('create_fold%02d_%s_%s' % (fold_idx, train_size, replicate_id), CreateFolds,
                            fold_index = fold_idx,
                            folds_count = self.folds_count,
                            seed = 0.637,
                            slurminfo = sciluigi.SlurmInfo(
                                runmode=runmode,
                                project=self.slurm_project,
                                partition='core',
                                cores='1',
                                time='1:00:00',
                                jobname='create_fold%02d_%s_%s' % (fold_idx, train_size, replicate_id),
                                threads='1'
                            ))
                    for cost in costseq:
                        tasks[replicate_id][fold_idx][cost] = {}
                        create_folds.in_dataset = shufflelines.out_shuffled
                        create_folds.in_linecount = cntlines.out_linecount
                        # -------------------------------------------------
                        train_lin = self.new_task('trainlin_fold_%d_cost_%s_%s_%s' % (fold_idx, cost, train_size, replicate_id), TrainLinearModel,
                                replicate_id = replicate_id,
                                lin_type = self.lin_type,
                                lin_cost = cost,
                                slurminfo = sciluigi.SlurmInfo(
                                    runmode=runmode,
                                    project=self.slurm_project,
                                    partition='core',
                                    cores='1',
                                    time='4-00:00:00',
                                    jobname='trnlin_f%02d_c%s_%s_%s' % (fold_idx, cost, train_size, replicate_id),
                                    threads='1'
                                ))
                        train_lin.in_traindata = create_folds.out_traindata
                        # -------------------------------------------------
                        pred_lin = self.new_task('predlin_fold_%d_cost_%s_%s_%s' % (fold_idx, cost, train_size, replicate_id), PredictLinearModel,
                                replicate_id = replicate_id,
                                slurminfo = sciluigi.SlurmInfo(
                                    runmode=runmode,
                                    project=self.slurm_project,
                                    partition='core',
                                    cores='1',
                                    time='8:00:00',
                                    jobname='predlin_f%02d_c%s_%s_%s' % (fold_idx, cost, train_size, replicate_id),
                                    threads='1'
                                ))
                        pred_lin.in_model = train_lin.out_model
                        pred_lin.in_sparse_testdata = create_folds.out_testdata
                        # -------------------------------------------------
                        assess_lin = self.new_task('assesslin_fold_%d_cost_%s_%s_%s' % (fold_idx, cost, train_size, replicate_id), AssessLinearRMSD,
                                lin_cost = cost,
                                slurminfo = sciluigi.SlurmInfo(
                                    runmode=runmode,
                                    project=self.slurm_project,
                                    partition='core',
                                    cores='1',
                                    time='15:00',
                                    jobname='assesslin_f%02d_c%s_%s_%s' % (fold_idx, cost, train_size, replicate_id),
                                    threads='1'
                                ))
                        assess_lin.in_model = train_lin.out_model
                        assess_lin.in_sparse_testdata = create_folds.out_testdata
                        assess_lin.in_prediction = pred_lin.out_prediction
                        # -------------------------------------------------
                        tasks[replicate_id][fold_idx][cost] = {}
                        tasks[replicate_id][fold_idx][cost]['create_folds'] = create_folds
                        tasks[replicate_id][fold_idx][cost]['train_linear'] = train_lin
                        tasks[replicate_id][fold_idx][cost]['predict_linear'] = pred_lin
                        tasks[replicate_id][fold_idx][cost]['assess_linear'] = assess_lin

                # Tasks for calculating average RMSD and finding the cost with lowest RMSD
                avgrmsd_tasks = {}
                for cost in costseq:
                    # Calculate the average RMSD for each cost value
                    average_rmsd = self.new_task('average_rmsd_cost_%s_%s_%s' % (cost, train_size, replicate_id), CalcAverageRMSDForCost,
                            lin_cost=cost)
                    average_rmsd.in_assessments = [tasks[replicate_id][fold_idx][cost]['assess_linear'].out_assessment for fold_idx in xrange(self.folds_count)]
                    avgrmsd_tasks[cost] = average_rmsd
                # --------------------------------------------------------------------------------
                sel_lowest_rmsd = self.new_task('select_lowest_rmsd_%s_%s' % (train_size, replicate_id), SelectLowestRMSD)
                sel_lowest_rmsd.in_values = [average_rmsd.out_rmsdavg for average_rmsd in avgrmsd_tasks.values()]
                # --------------------------------------------------------------------------------
                run_id = 'mainwfrun_liblinear_%s_tst%s_trn%s_%s' % (self.dataset_name, self.test_size, train_size, replicate_id)
                mainwfrun = self.new_task('mainwfrun_%s_%s' % (train_size, replicate_id), MainWorkflowRunner,
                        dataset_name=self.dataset_name,
                        run_id=run_id,
                        replicate_id=replicate_id,
                        sampling_method='random',
                        train_method='liblinear',
                        train_size=train_size,
                        test_size=self.test_size,
                        lin_type=self.lin_type,
                        slurm_project=self.slurm_project,
                        parallel_lin_train=False,
                        runmode=self.runmode)
                mainwfrun.in_lowestrmsd = sel_lowest_rmsd.out_lowest
                # --------------------------------------------------------------------------------
                # Collect one lowest rmsd per train size
                lowest_rmsds.append(sel_lowest_rmsd)

                mainwfruns.append(mainwfrun)

        # --------------------------------------------------------------------------------
        mergedreport = self.new_task('merged_report_%s_%s' % (self.dataset_name, self.run_id), MergedDataReport,
                run_id = self.run_id)
        mergedreport.in_reports = [t.out_report for t in mainwfruns]

        return mergedreport

# ================================================================================

class MainWorkflowRunner(sciluigi.Task):
    # Parameters
    dataset_name = luigi.Parameter()
    run_id = luigi.Parameter()
    replicate_id =luigi.Parameter()
    sampling_method = luigi.Parameter()
    train_method = luigi.Parameter()
    train_size = luigi.Parameter()
    test_size = luigi.Parameter()
    lin_type = luigi.Parameter()
    slurm_project = luigi.Parameter()
    parallel_lin_train = luigi.BoolParameter()
    runmode = luigi.Parameter()

    # In-ports (defined as fields accepting sciluigi.TargetInfo objects)
    in_lowestrmsd = None

    # Out-ports
    def out_done(self):
        return sciluigi.TargetInfo(self, self.in_lowestrmsd().path + '.mainwf_done')
    def out_report(self):
        outf_path = 'data/' + self.run_id + '/testrun_dataset_liblinear_datareport.csv'
        return sciluigi.TargetInfo(self, outf_path) # We manually re-create the filename that this should have

    # Task implementation
    def run(self):
        with self.in_lowestrmsd().open() as infile:
            records = sciluigi.recordfile_to_dict(infile)
            lowest_cost = records['lowest_cost']
        self.ex('python wfmm.py' +
                ' --dataset-name=%s' % self.dataset_name +
                ' --run-id=%s' % self.run_id +
                ' --replicate-id=%s' % self.replicate_id +
                ' --sampling-method=%s' % self.sampling_method +
                ' --train-method=%s' % self.train_method +
                ' --train-size=%s' % self.train_size +
                ' --test-size=%s' % self.test_size +
                ' --lin-type=%s' % self.lin_type +
                ' --lin-cost=%s' % lowest_cost +
                ' --slurm-project=%s' % self.slurm_project +
                ' --runmode=%s' % self.runmode)
        with self.out_done().open('w') as donefile:
            donefile.write('Done!\n')


# ## Execute the workflow
#
# Execute the workflow locally (using the luigi daemon which runs in the background), starting with the `CrossValidateWorkflow` workflow class.

# In[ ]:

print time.strftime('%Y-%m-%d %H:%M:%S: ') + 'Workflow started ...'
sciluigi.run(cmdline_args=['--scheduler-host=localhost', '--workers=4'], main_task_cls=CrossValidateWorkflow)
print time.strftime('%Y-%m-%d %H:%M:%S: ') + 'Workflow finished!'


# ## Parse result data from workflow into python dicts
#
# This step does not produce any output, but is done as a preparation for the subsequent printing of values, and plotting.

# In[ ]:

import csv
from matplotlib.pyplot import *

merged_report_filepath = 'data/test_run_001_merged_report.csv'
replicate_ids = ['r1','r2','r3']
rowdicts = []


# Collect data in one dict per row in the csv file
with open(merged_report_filepath) as infile:
    csvrd = csv.reader(infile, delimiter=',')
    for rid, row in enumerate(csvrd):
        if rid == 0:
            headerrow = row
        else:
            rowdicts.append({headerrow[i]:v for i, v in enumerate(row)})

# Collect the training sizes
train_sizes = []
for r  in rowdicts:
    if r['replicate_id'] == 'r1':
        train_sizes.append(r['train_size'])

# Collect the training times, RMSD- and (LIBLINEAR) Cost values
train_times = {}
rmsd_values = {}
cost_values = {}
for repl_id in replicate_ids:
    train_times[repl_id] = []
    rmsd_values[repl_id] = []
    cost_values[repl_id] = []
    for r in rowdicts:
        if r['replicate_id'] == repl_id:
            train_times[repl_id].append(r['train_time_sec'])
            rmsd_values[repl_id].append(r['rmsd'])
            cost_values[repl_id].append(r['lin_cost'])

# Calculate average values for the training time
train_times_avg = []
for i in range(0, len(train_times['r1'])):
    train_times_avg.append(0.0)
    for repl_id in replicate_ids:
        train_times_avg[i] += float(train_times[repl_id][i])
    train_times_avg[i] =  train_times_avg[i] / float(len(replicate_ids))

# Calculate average values for the RMSD values
rmsd_values_avg = []
for i in range(0, len(rmsd_values['r1'])):
    rmsd_values_avg.append(0.0)
    for repl_id in replicate_ids:
        rmsd_values_avg[i] += float(rmsd_values[repl_id][i])
    rmsd_values_avg[i] =  rmsd_values_avg[i] / float(len(replicate_ids))


# ## Print values (Train sizes, train times and RMSD)

# In[ ]:

print "-"*60
print 'Train sizes:        ' + ', '.join(train_sizes) + ' molecules'
print ''
print 'RMSD values: '
for rid in range(1,4):
    print '      Replicate %d: ' % rid + ', '.join(['%.2f' % float(v) for v in rmsd_values['r%d' % rid]])
print ''
print 'Train times: '
for rid in range(1,4):
    print '      Replicate %d: ' % rid + ', '.join(train_times['r%d' % rid]) + ' seconds'
print ''
print 'Cost values: '
for rid in range(1,4):
    print '      Replicate %d: ' % rid + ', '.join(cost_values['r%d' % rid])
print ''
print 'RMSD values (avg): ' + ', '.join(['%.2f' % x for x in rmsd_values_avg])
print 'Train times (avg): ' + ', '.join(['%.2f' % x for x in train_times_avg]) + ' seconds'
print "-"*60


# ## Plot train time and RMSD against training size

# In[ ]:

# Initialize plotting figure
fig = figure()

# Set up subplot for RMSD values
subpl1 = fig.add_subplot(1,1,1)
# x-axis
xticks = [500,1000,2000,4000,8000]
subpl1.set_xscale('log')
subpl1.set_xlim([500,8000])
subpl1.set_xticks(ticks=xticks)
subpl1.set_xticklabels([str(l) for l in xticks])
subpl1.set_xlabel('Training set size (number of molecules)')
# y-axis
subpl1.set_ylim([0,1])
subpl1.set_ylabel('RMSD for test prediction')
# plot
subpl1.plot(train_sizes,
     rmsd_values_avg,
     label='RMSD for test prediction',
     marker='.',
     color='k',
     linestyle='-')

# Set up subplot for training times
subpl2 = subpl1.twinx()
# y-axis
yticks = [0.01,0.02,0.03,0.05,0.1,0.2,0.3,0.4,0.5]
subpl2.set_ylim([0.01,0.5])
subpl2.set_yscale('log')
subpl2.set_yticks(ticks=yticks)
subpl2.set_yticklabels([str(int(l*1000)) for l in yticks])
subpl2.set_ylabel('Training time (milliseconds)')
subpl2.tick_params(axis='y', colors='r')
subpl2.yaxis.label.set_color('r')
subpl2.spines['right'].set_color('red')
# plot
subpl2.plot(train_sizes,
     train_times_avg,
     label='Training time (seconds)',
     marker='.',
     color='r',
     linestyle='-')

subpl1.legend(loc='upper left', fontsize=9)
subpl2.legend(bbox_to_anchor=(0, 0.9), loc='upper left', fontsize=9)

show() # Display the plot

