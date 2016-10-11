from cheminf_components import *
import luigi
import sciluigi as sl
import time

# ================================================================================

TRAINMETHOD_LIBLINEAR = 'liblinear'
TRAINMETHOD_SVMRBF = 'svmrbf'

class MMWorkflow(sl.WorkflowTask):
    '''
    This class runs the MM Workflow using LibLinear
    as the method for doing machine learning
    '''

    # WORKFLOW PARAMETERS
    dataset_name = luigi.Parameter(default='testrun_dataset')
    run_id = luigi.Parameter(default='test_run_001')
    replicate_id = luigi.Parameter(default=None)
    replicate_ids = luigi.Parameter(default='r1,r2,r3')
    sampling_method = luigi.Parameter(default='random')
    train_method = luigi.Parameter(default=TRAINMETHOD_LIBLINEAR) # TRAINMETHOD_LIBLINEAR or TRAINMETHOD_SVMRBF
    train_size = luigi.Parameter(default=5000)
    train_sizes = luigi.Parameter(default=None)
    test_size = luigi.Parameter(default=1000)

    lin_type = luigi.Parameter(default='12')
    lin_cost = luigi.Parameter(default='100')

    svm_gamma = luigi.Parameter(default='0.001')
    svm_cost = luigi.Parameter(default='100')
    svm_type = luigi.Parameter(default='3')
    svm_kernel_type = luigi.Parameter('2')

    parallel_lin_train = luigi.BooleanParameter()
    parallel_svm_train = luigi.BooleanParameter()
    runmode = luigi.Parameter(default='local')
    
    slurm_project = luigi.Parameter(default='N/A')

    #folds_count = luigi.Parameter()

    def workflow(self):
        if self.runmode == 'local':
            runmode = sl.RUNMODE_LOCAL
        elif self.runmode == 'hpc':
            runmode = sl.RUNMODE_HPC
        elif self.runmode == 'mpi':
            runmode = sl.RUNMODE_MPI
        else:
            raise Exception('Runmode is none of local, hpc, nor mpi. Please fix and try again!')

        datareport_rows = []
        plotmerged_tasks = []

        coloring_train_data_task = None
        coloring_train_task = None

        if self.replicate_id is not None:
            replicate_ids = [self.replicate_id]
        elif self.replicate_ids is not None:
            replicate_ids = [i for i in self.replicate_ids.split(',')]
        for replicate_id in replicate_ids:
            if self.train_size is not None:
                train_sizes = [self.train_size]
            elif self.train_sizes is not None:
                train_sizes = [i for i in self.train_sizes.split(',')]
            for train_size in train_sizes:
                # --------------------------------------------------------------------------------
                existing_smiles = self.new_task('existing_smiles', ExistingSmiles,
                        dataset_name = self.dataset_name)
                # --------------------------------------------------------------------------------
                gen_sign_filter_subst = self.new_task('gen_sign_filter_subst', GenerateSignaturesFilterSubstances,
                        min_height = 1,
                        max_height = 3,
                        dataset_name = self.dataset_name,
                        silent_mode=False,
                        slurminfo = sl.SlurmInfo(
                            runmode=runmode,
                            project=self.slurm_project,
                            partition='core',
                            cores='8',
                            time='1:00:00',
                            jobname='MMGenSign',
                            threads='8',
                        ))
                gen_sign_filter_subst.in_smiles = existing_smiles.out_smiles
                # --------------------------------------------------------------------------------
                create_unique_run_copy = self.new_task('create_unique_run_copy_%s' % self.run_id,
                        CreateRunCopy,
                        run_id = self.run_id)
                create_unique_run_copy.in_file = gen_sign_filter_subst.out_signatures
                # --------------------------------------------------------------------------------
                create_unique_sign_copy = self.new_task('create_unique_sign_copy_%s' % replicate_id, CreateReplicateCopy,
                        replicate_id = replicate_id)
                create_unique_sign_copy.in_file = create_unique_run_copy.out_copy
                # --------------------------------------------------------------------------------
                sample_train_and_test = self.new_task('sample_trn%s_tst%s_c%s_%s' % (train_size, self.test_size, self.lin_cost, replicate_id), SampleTrainAndTest,
                        test_size = self.test_size,
                        train_size = train_size,
                        sampling_method = self.sampling_method,
                        dataset_name = self.dataset_name,
                        replicate_id = replicate_id,
                        slurminfo = sl.SlurmInfo(
                            runmode=runmode,
                            project=self.slurm_project,
                            partition='core',
                            cores='2',
                            time='1:00:00',
                            jobname='sample_trn%s_tst%s_c%s_%s' % (train_size, self.test_size, self.lin_cost, replicate_id),
                            threads='1'
                        ))
                sample_train_and_test.in_signatures = create_unique_sign_copy.out_copy
                # --------------------------------------------------------------------------------
                create_sparse_train_dataset = self.new_task('create_sparse_traindata_trn%s_tst%s_c%s_%s' % (train_size, self.test_size, self.lin_cost, replicate_id), CreateSparseTrainDataset,
                        dataset_name = self.dataset_name,
                        replicate_id = replicate_id,
                        slurminfo = sl.SlurmInfo(
                            runmode=runmode,
                            project=self.slurm_project,
                            partition='core',
                            cores='8',
                            time='1-00:00:00',
                            jobname='MMCreateSparseTrain',
                            threads='8'
                        ))
                create_sparse_train_dataset.in_traindata = sample_train_and_test.out_traindata
                # ------------------------------------------------------------------------
                create_sparse_test_dataset = self.new_task('create_sparse_testdata_trn%s_tst%s_c%s_%s' % (train_size, self.test_size, self.lin_cost, replicate_id), CreateSparseTestDataset,
                        dataset_name = self.dataset_name,
                        replicate_id = replicate_id,
                        slurminfo = sl.SlurmInfo(
                            runmode=runmode,
                            project=self.slurm_project,
                            partition='core',
                            cores='8',
                            time='1-00:00:00',
                            jobname='sparse_trn%s_tst%s_c%s' % (train_size, self.test_size, self.lin_cost),
                            threads='8'
                        ))
                create_sparse_test_dataset.in_testdata = sample_train_and_test.out_testdata
                create_sparse_test_dataset.in_signatures = create_sparse_train_dataset.out_signatures
                # ------------------------------------------------------------------------
                ungzip_testdata = self.new_task('ungzip_testdata_trn%s_tst%s_c%s_%s' % (train_size, self.test_size, self.lin_cost, replicate_id), UnGzipFile,
                        slurminfo = sl.SlurmInfo(
                            runmode=runmode,
                            project=self.slurm_project,
                            partition='core',
                            cores='1',
                            time='1:00:00',
                            jobname='ungziptest_trn%s_tst%s_c%s' % (train_size, self.test_size, self.lin_cost),
                            threads='1'
                        ))
                ungzip_testdata.in_gzipped = create_sparse_test_dataset.out_sparse_testdata
                # ------------------------------------------------------------------------
                ungzip_traindata = self.new_task('ungzip_traindata_trn%s_tst%s_c%s_%s' % (train_size, self.test_size, self.lin_cost, replicate_id), UnGzipFile,
                        slurminfo = sl.SlurmInfo(
                            runmode=runmode,
                            project=self.slurm_project,
                            partition='core',
                            cores='1',
                            time='1:00:00',
                            jobname='ungziptrain_trn%s_tst%s_c%s' % (train_size, self.test_size, self.lin_cost),
                            threads='1'
                        ))
                ungzip_traindata.in_gzipped = create_sparse_train_dataset.out_sparse_traindata
                # ========================================================================
                # START: ALTERNATIVE TRAINING METHODS
                # ========================================================================
                if self.train_method == TRAINMETHOD_LIBLINEAR:
                    train_model = self.new_task('train_lin_trn%s_tst%s_c%s_%s' % (train_size, self.test_size, self.lin_cost, replicate_id), TrainLinearModel,
                            replicate_id = replicate_id,
                            dataset_name = self.dataset_name,
                            train_size = train_size,
                            test_size = self.test_size,
                            lin_type = self.lin_type,
                            lin_cost = self.lin_cost,
                            slurminfo = sl.SlurmInfo(
                                runmode=runmode,
                                project=self.slurm_project,
                                partition='core',
                                cores='1',
                                time='4-00:00:00',
                                jobname='trainlin_trn%s_tst%s_c%s' % (train_size, self.test_size, self.lin_cost),
                                threads='1'
                            ))
                    train_model.in_traindata = ungzip_traindata.out_ungzipped
                    # ------------------------------------------------------------------------
                    predict = self.new_task('predict_lin_trn%s_tst%s_c%s_%s' % (train_size, self.test_size, self.lin_cost, replicate_id), PredictLinearModel,
                            dataset_name = self.dataset_name,
                            replicate_id = replicate_id,
                            slurminfo = sl.SlurmInfo(
                                runmode=runmode,
                                project=self.slurm_project,
                                partition='core',
                                cores='1',
                                time='4:00:00',
                                jobname='predlin_trn%s_tst%s_c%s' % (train_size, self.test_size, self.lin_cost),
                                threads='1'
                            ))
                    predict.in_model = train_model.out_model
                    predict.in_sparse_testdata = ungzip_testdata.out_ungzipped
                    # ------------------------------------------------------------------------
                    assess_model = self.new_task('assess_lin_trn%s_tst%s_c%s_%s' % (train_size, self.test_size, self.lin_cost, replicate_id), AssessLinearRMSD,
                            dataset_name = self.dataset_name,
                            replicate_id = replicate_id,
                            lin_cost = self.lin_cost,
                            slurminfo = sl.SlurmInfo(
                                runmode=runmode,
                                project=self.slurm_project,
                                partition='core',
                                cores='1',
                                time='15:00',
                                jobname='assesslin_trn%s_tst%s_c%s' % (train_size, self.test_size, self.lin_cost),
                                threads='1'
                            ))
                # ========================================================================
                elif self.train_method == TRAINMETHOD_SVMRBF:
                # ========================================================================
                    if self.dataset_name == 'acd_logd':
                        if train_size == 'rest':
                            runtime = '10-00:00:00'
                        elif train_size == '20000':
                            runtime = '7-00:00:00'
                        else:
                            runtime = '4-00:00:00'
                    else:
                        runtime = '4-00:00:00'
                    if runmode in [sl.RUNMODE_HPC, sl.RUNMODE_MPI]:
                        runmode_train = sl.RUNMODE_MPI
                    else:
                        runmode_train = sl.RUNMODE_LOCAL
                    train_model = self.new_task('train_svm_trn%s_tst%s_g%s_c%s_%s' % (train_size, self.test_size, self.svm_gamma, self.svm_cost, replicate_id), 
                            TrainSVMModel,
                            replicate_id = replicate_id,
                            dataset_name = self.dataset_name,
                            train_size = train_size,
                            svm_gamma = '0.001',
                            svm_cost = '100',
                            svm_type = '3',
                            svm_kernel_type = '2',
                            parallel_train = self.parallel_svm_train,
                            slurminfo = sl.SlurmInfo(
                                runmode=runmode_train,
                                project=self.slurm_project,
                                partition='node',
                                cores='64',
                                time=runtime,
                                jobname='trainsvm_tr%s_ts%s_g%s_c%s' % (train_size, self.test_size, self.svm_gamma, self.svm_cost),
                                threads='64' # Not used!
                            ))
                    train_model.in_traindata = ungzip_traindata.out_ungzipped
                    # ------------------------------------------------------------------------
                    # Save task creating train set for coloring
                    # ------------------------------------------------------------------------
                    if train_size == '80000' or train_size == '2048': # The 2048 is used in the test script only!
                        coloring_train_data_task = ungzip_traindata
                        coloring_train_task = train_model
                    # ------------------------------------------------------------------------
                    predict = self.new_task('predict_svm_trn%s_tst%s_g%s_c%s_%s' % (train_size, self.test_size, self.svm_gamma, self.svm_cost, replicate_id),
                            PredictSVMModel,
                            dataset_name = self.dataset_name,
                            replicate_id = replicate_id,
                            slurminfo = sl.SlurmInfo(
                                runmode=runmode,
                                project=self.slurm_project,
                                partition='core',
                                cores='1',
                                time='4:00:00',
                                jobname='predlin_trn%s_tst%s_c%s' % (train_size, self.test_size, self.lin_cost),
                                threads='1'
                            ))
                    predict.in_svmmodel = train_model.out_model
                    predict.in_sparse_testdata = ungzip_testdata.out_ungzipped
                    # ------------------------------------------------------------------------
                    assess_model = self.new_task('assess_lin_trn%s_tst%s_c%s_%s' % (train_size, self.test_size, self.lin_cost, replicate_id),
                            AssessSVMRMSD,
                            dataset_name = self.dataset_name,
                            replicate_id = replicate_id,
                            svm_cost = self.svm_cost,
                            svm_gamma = self.svm_gamma,
                            svm_type = self.svm_type,
                            svm_kernel_type = self.svm_kernel_type,
                            slurminfo = sl.SlurmInfo(
                                runmode=runmode,
                                project=self.slurm_project,
                                partition='core',
                                cores='1',
                                time='15:00',
                                jobname='assesslin_trn%s_tst%s_c%s' % (train_size, self.test_size, self.lin_cost),
                                threads='1'
                            ))
                # ========================================================================
                # END: ALTERNATIVE TRAINING METHODS
                # ========================================================================


                assess_model.in_prediction = predict.out_prediction
                assess_model.in_model = ungzip_traindata.out_ungzipped
                assess_model.in_sparse_testdata = ungzip_testdata.out_ungzipped
                # ------------------------------------------------------------------------
                count_trainsize_filtered = self.new_task('count_trainsize_filtered_svmrbf_%s_trn%s_tst%s_%s' % (self.dataset_name, train_size, self.test_size, replicate_id),
                        CountLines,
                        ungzip=True,
                        slurminfo = sl.SlurmInfo(
                            runmode=runmode,
                            project=self.slurm_project,
                            partition='core',
                            cores='1',
                            time='15:00',
                            jobname='count_trainsize_filtered_svmrbf_%s_trn%s_tst%s_%s' % (self.dataset_name, train_size, self.test_size, replicate_id),
                            threads='1'
                        ))
                count_trainsize_filtered.in_file = sample_train_and_test.out_traindata
                # ------------------------------------------------------------------------
                mergedorigpred = self.new_task('merge_orig_pred_%s_trn%s_tst%s_%s' % (self.dataset_name, train_size, self.test_size, replicate_id),
                        MergeOrigAndPredValues)
                mergedorigpred.in_original_dataset = ungzip_testdata.out_ungzipped
                mergedorigpred.in_predicted_dataset = predict.out_prediction
                # ------------------------------------------------------------------------
                plotmerged = self.new_task('plotmerged_orig_pred_%s_trn%s_tst%s_%s' % (self.dataset_name, train_size, self.test_size, replicate_id),
                        PlotCSV,
                        xmin='-20',
                        xmax='20',
                        ymin='-20',
                        ymax='20')

                plotmerged.in_csv = mergedorigpred.out_merged
                plotmerged_tasks.append(plotmerged)
                # ------------------------------------------------------------------------
                collect_datarow = self.new_task('collect_datarow_svmrbf_%s_trn%s_tst%s_%s' % (self.dataset_name, train_size, self.test_size, replicate_id),
                        CollectDataReportRow,
                        dataset_name = self.dataset_name,
                        train_method = self.train_method,
                        train_size = train_size,
                        replicate_id = replicate_id,
                        lin_cost = self.lin_cost,
                        slurminfo = sl.SlurmInfo(
                            runmode=runmode,
                            project=self.slurm_project,
                            partition='core',
                            cores='1',
                            time='15:00',
                            jobname='datarow_svmrbf_%s_trn%s_tst%s_%s' % (self.dataset_name, train_size, self.test_size, replicate_id),
                            threads='1'
                        ))
                collect_datarow.in_rmsd = assess_model.out_assessment
                collect_datarow.in_traintime = train_model.out_traintime
                collect_datarow.in_trainsize_filtered = count_trainsize_filtered.out_linecount
                # ------------------------------------------------------------------------
                datareport_rows.append(collect_datarow)
        datareport = self.new_task('datareport_%s_%s' % (self.dataset_name, self.train_method),
                CollectDataReport,
                dataset_name = self.dataset_name,
                train_method = self.train_method)
        datareport.in_datareport_rows = [dr.out_datareport_row for dr in datareport_rows]
        # ------------------------------------------------------------------------

        # ========================================================================
        # START: CALCULATECOLORING INDEX VALUES
        # ========================================================================
        if coloring_train_data_task is not None and coloring_train_task is not None:
            predict_train = self.new_task('predict_train',
                    PredictSVMModel,
                    dataset_name = self.dataset_name,
                    replicate_id = replicate_id,
                    slurminfo = sl.SlurmInfo(
                        runmode=runmode,
                        project=self.slurm_project,
                        partition='core',
                        cores='1',
                        time='4:00:00',
                        jobname='predict_train',
                        threads='1'
                    ))
            predict_train.in_svmmodel = coloring_train_task.out_model
            predict_train.in_sparse_testdata = coloring_train_data_task.out_ungzipped
            # ------------------------------------------------------------------------
            select_idx10 = self.new_task('select_idx10',
                    SelectPercentIndexValue,
                    percent_index=10)
            select_idx10.in_prediction = predict_train.out_prediction
            # ------------------------------------------------------------------------
            select_idx90 = self.new_task('select_idx90',
                    SelectPercentIndexValue,
                    percent_index=90)
            select_idx90.in_prediction = predict_train.out_prediction
        # ------------------------------------------------------------------------
        # return [select_idx10, select_idx90]
        # ========================================================================
        # END: CALCULATECOLORING INDEX VALUES
        # ========================================================================

        return [plotmerged_tasks, datareport]
        #return mergedorigpred

# ====================================================================================================

if __name__ == '__main__':
    sl.run_local(main_task_cls=MMWorkflow)
