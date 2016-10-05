# ====================================================================================================
#  New components for Cross-validation - May 8, 2015
# ----------------------------------------------------------------------------------------------------
#  NOTE, Aug 19 2015: This file is deprecated, and will be replaced
#                     with new code from an experiment in the mmproj repo!
#                     It is kept only for reference when developing
#                     the new components!
# ====================================================================================================

## ====================================================================================================
#
#class SplitDataset(DependencyMetaTask, TaskHelpers, AuditTrailMixin):
#
#    # INPUT TARGETS
#    dataset_target = luigi.Parameter()
#
#    # TASK PARAMETERS
#    splits_count = luigi.Parameter()
#
#    def output(self):
#        return { 'split_%d' % i: luigi.LocalTarget(self.get_input('dataset_target').path + '.split_%d' % i) for i in xrange(int(self.splits_count)) }
#
#    # CONVENIENCE METHODS
#    def count_lines(self, filename):
#        status, output = self.lx(['cat', filename, '|', 'wc -l'])
#        return int(output)
#
#    def remove_dict_key(self, orig_dict, key):
#        new_dict = dict(orig_dict)
#        del new_dict[key]
#        return new_dict
#
#    def pick_lines(self, dataset, line_nos):
#        return [line for i, line in enumerate(dataset) if i in line_nos]
#
#    def run(self):
#        linecnt = self.count_lines(self.get_input('dataset_target').path)
#        line_nos = [i for i in xrange(1, linecnt)]
#        random.shuffle(line_nos)
#
#        splits_as_linenos = {}
#
#        set_size = len(line_nos) // int(self.splits_count)
#
#        # Split into splits, in terms of line numbers
#        for i in xrange(int(self.splits_count)):
#            splits_as_linenos[i] = line_nos[i * set_size : (i+1) * set_size]
#
#        for i, split_id in enumerate(self.output()):
#            with self.get_input('dataset_target').open() as infile, self.output()[split_id].open('w') as outfile:
#                lines = self.pick_lines(infile, splits_as_linenos[i])
#                outfile.writelines(lines)
#
## ====================================================================================================
#
#class CreateFolds(DependencyMetaTask, TaskHelpers, AuditTrailMixin):
#
#    # INPUT TARGETS
#    dataset_target = luigi.Parameter()
#
#    # TASK PARAMETERS
#    folds_count = luigi.Parameter()
#
#    def output(self):
#        return { 'folds' : { i: { 'train': luigi.LocalTarget(self.get_input('dataset_target').path + '.fold_%d_train' % i),
#                                  'test':  luigi.LocalTarget(self.get_input('dataset_target').path + '.fold_%d_test' % i) }
#                                  for i in xrange(int(self.folds_count)) } }
#
#    # CONVENIENCE METHODS
#    def count_lines(self, filename):
#        status, output = self.lx(['cat', filename, '|', 'wc -l'])
#        return int(output)
#
#    def remove_dict_key(self, orig_dict, key):
#        new_dict = dict(orig_dict)
#        del new_dict[key]
#        return new_dict
#
#    def pick_lines(self, dataset, line_nos):
#        return [line for i, line in enumerate(dataset) if i in line_nos]
#
#    def pick_lines_inverted(self, dataset, line_nos):
#        return [line for i, line in enumerate(dataset) if i not in line_nos]
#
#    def run(self):
#        linecnt = self.count_lines(self.get_input('dataset_target').path)
#        line_nos = [i for i in xrange(1, linecnt)]
#        random.shuffle(line_nos)
#
#        splits_as_linenos = {}
#
#        set_size = len(line_nos) // int(self.folds_count)
#
#        # Split into splits, in terms of line numbers
#        for i in xrange(int(self.folds_count)):
#            splits_as_linenos[i] = line_nos[i * set_size : (i+1) * set_size]
#
#        for i, targets in self.output()['folds'].iteritems():
#            with targets['train'].open('w') as trainfile, targets['test'].open('w') as testfile:
#                with self.get_input('dataset_target').open() as infile:
#                    test_lines = self.pick_lines(infile, splits_as_linenos[i])
#                    testfile.writelines(test_lines)
#                with self.get_input('dataset_target').open() as infile:
#                    train_lines = self.pick_lines_inverted(infile, splits_as_linenos[i])
#                    trainfile.writelines(train_lines)

#class CrossValSplitIntoFolds(DependencyMetaTask, TaskHelpers, AuditTrailMixin):
#
#    # INPUT TARGETS
#    dataset_target = luigi.Parameter()
#
#    # TASK PARAMETERS
#    folds_count = luigi.Parameter()
#
#    def output(self):
#        return { 'split_%d' % i: luigi.LocalTarget(self.get_input('dataset_target').path + '.split_%d' % i) for i in xrange(int(self.splits_count)) }
#
#    # CONVENIENCE METHODS
#    def count_lines(self, filename):
#        status, output = self.lx(['cat', filename, '|', 'wc -l'])
#        return int(output)
#
#    def remove_dict_key(self, orig_dict, key):
#        new_dict = dict(orig_dict)
#        del new_dict[key]
#        return new_dict
#
#    def pick_lines(self, dataset, line_nos):
#        return [line for i, line in enumerate(dataset) if i in line_nos]
#
#    def run(self):
#        linecnt = self.count_lines(self.get_input('dataset_target').path)
#        line_nos = [i for i in xrange(1, linecnt)]
#        random.shuffle(line_nos)
#
#        splits_as_linenos = {}
#
#        set_size = len(line_nos) // int(self.folds_count)
#
#        # Split into splits, in terms of line numbers
#        for i in xrange(int(self.folds_count)):
#            splits_as_linenos[i] = line_nos[i * set_size : (i+1) * set_size]
#
#        for i, split_id in enumerate(self.output()):
#            with self.get_input('dataset_target').open() as infile, self.output()[split_id].open('w') as outfile:
#                lines = self.pick_lines(infile, splits_as_linenos[i])
#                outfile.writelines(lines)
#
#
#class CrossValidate(DependencyMetaTask, TaskHelpers, AuditTrailMixin):
#    '''
#    For now, a sketch on how to implement Cross-Validation as a sub-workflow components
#    '''
#
#    # TARGETS
#    dataset_target = luigi.Parameter()
#
#    # PARAMETERS
#    folds_count = luigi.Parameter()
#    replicate_id = luigi.Parameter()
#
#    def requires(self):
#
#        # Create the initial component that splits the initial dataset into
#        # k equal splits, or folds ...
#        split_dataset = CrossValSplitIntoFolds(
#                dataset_target = self.dataset_target,
#                folds_count=self.folds_count
#            )
#
#        # Branch the workflow into one branch per fold
#        fold_tasks = {}
#        for fc in self.folds_count:
#            fold_tasks[fc] = {}
#
#            # A task that will merge all folds except the one left out for testing,
#            # ... into a training data set, and just pass on the one left out, as
#            # the test data set.
#            fold_tasks[fc]['merge'] = CreateTestTrainDatasets(
#                split_dataset_target =
#                    { 'upstream' : { 'task' : split_dataset,
#                                     'port' : 'splits' } },
#                replicate_id =  self.replicate_id
#            )
#            # Plugging in the 'generic' train components, for SVM/LibLinear, here
#            fold_tasks[fc]['train'] = TrainSVMModel(
#                train_dataset_target =
#                    { 'upstream' : { 'task' : fold_tasks[fc]['merge'],
#                                     'port' : 'train_dataset' } },
#                replicate_id = self.replicate_id
#            )
#            # Plugging in the 'generic' predict components, for SVM/LibLinear, here
#            fold_tasks[fc]['predict'] = PredictSVMModel(
#                svmmodel_target =
#                    { 'upstream' : { 'task' : fold_tasks[fc]['train'],
#                                     'port' : ... } },
#                sparse_test_dataset_target = ,
#                replicate_id = self.replicate_id
#            )
#
#        # Collect the prediction targets from the branches above, into one dict, to feed
#        # into the specialized assess component below
#        predict_targets = { fc : fold_tasks[fc]['predict'] for fc in self.folds_count }
#
#        assess_crossval_predicts = CrossValAssessPredictions(
#                prediction_targets = predict_targets,
#                folds_count = self.folds_count
#            )
#        return assess_crossval_predicts.output()['report']
#
#    def output(self):
#        return self.input()
#
#    def run(self):
#        pass # Don't do anything ... everything happens in the requires() method
