from cheminf_components import *
import logging
import luigi
import sciluigi as sl
import time

# ====================================================================================================
#  New components for Cross-validation - May 8, 2015
# ====================================================================================================

log = logging.getLogger('sciluigi-interface')

class CrossValidate(sl.WorkflowTask):
    '''
    For now, a sketch on how to implement Cross-Validation as a sub-workflow components
    '''

    # PARAMETERS
    dataset_name = luigi.Parameter(default='testrun_dataset')
    run_id = luigi.Parameter('test_run_001')
    replicate_id = luigi.Parameter(default=None)
    replicate_ids = luigi.Parameter(default='r1,r2,r3')
    folds_count = luigi.IntParameter(default=10)
    min_height = luigi.Parameter(default=1)
    max_height = luigi.Parameter(default=3)
    test_size = luigi.Parameter(1000)
    train_sizes = luigi.Parameter(default='5000')
    lin_type = luigi.Parameter(default='12') # 12, See: https://www.csie.ntu.edu.tw/~cjlin/liblinear/FAQ.html
    randomdatasize_mb = luigi.IntParameter(default=10)
    runmode = 'local'
    slurm_project = 'N/A'

    def workflow(self):
        if self.runmode == 'local':
            runmode = sl.RUNMODE_LOCAL
        elif self.runmode == 'hpc':
            runmode = sl.RUNMODE_HPC
        elif self.runmode == 'mpi':
            runmode = sl.RUNMODE_MPI
        else:
            raise Exception('Runmode is none of local, hpc, nor mpi. Please fix and try again!')

        # ----------------------------------------------------------------
        mmtestdata = self.new_task('mmtestdata', ExistingSmiles,
                replicate_id='na',
                dataset_name=self.dataset_name)
        tasks = {}
        lowest_rmsds = []
        mainwfruns = []
        if self.replicate_id is not None:
            replicate_ids = [self.replicate_id]
        else:
            replicate_ids = [i for i in self.replicate_ids.split(',')]
        for replicate_id in replicate_ids:
            tasks[replicate_id] = {}
            gensign = self.new_task('gensign_%s' % replicate_id, GenerateSignaturesFilterSubstances,
                    replicate_id=replicate_id,
                    min_height = self.min_height,
                    max_height = self.max_height,
                    slurminfo = sl.SlurmInfo(
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
                        slurminfo = sl.SlurmInfo(
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
                        slurminfo = sl.SlurmInfo(
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
                        slurminfo = sl.SlurmInfo(
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
                        slurminfo = sl.SlurmInfo(
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
                        slurminfo = sl.SlurmInfo(
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
                        slurminfo = sl.SlurmInfo(
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
                            slurminfo = sl.SlurmInfo(
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
                                slurminfo = sl.SlurmInfo(
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
                                slurminfo = sl.SlurmInfo(
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
                                slurminfo = sl.SlurmInfo(
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

                sel_lowest_rmsd = self.new_task('select_lowest_rmsd_%s_%s' % (train_size, replicate_id), SelectLowestRMSD)
                sel_lowest_rmsd.in_values = [average_rmsd.out_rmsdavg for average_rmsd in avgrmsd_tasks.values()]

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

                # Collect one lowest rmsd per train size
                lowest_rmsds.append(sel_lowest_rmsd)

                mainwfruns.append(mainwfrun)

        return mainwfruns

# ================================================================================

class MainWorkflowRunner(sl.Task):
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
    parallel_lin_train = luigi.BooleanParameter()
    runmode = luigi.Parameter()
    # In-ports
    in_lowestrmsd = None
    # Out-ports
    def out_done(self):
        return sl.TargetInfo(self, self.in_lowestrmsd().path + '.mainwf_done')
    # Task action
    def run(self):
        with self.in_lowestrmsd().open() as infile:
            records = sl.recordfile_to_dict(infile)
            lowest_cost = records['lowest_cost']
        self.ex('python wfmm.py MMWorkflow' +
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

# ================================================================================

if __name__ == '__main__':
    sl.run_local(main_task_cls=CrossValidate)
