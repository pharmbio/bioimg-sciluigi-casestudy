import ast
import commands
import csv
import datetime
import gzip
import hashlib
import logging
import luigi
import math
import os
import random
import re
import sciluigi as sl
import shutil
import string
import sys
import textwrap
import time
import uuid
from ConfigParser import ConfigParser

# ====================================================================================================

log = logging.getLogger('sciluigi-interface')

# ====================================================================================================

class JVMHelpers(sl.Task):
    '''Mixin with helper methods for starting and keeping alive a JVM, using jpype'''
    def start_jvm(self, jvm_path, class_path):
        import jpype
        jpype.startJVM(jvm_path,
                       '-ea',
                       '-Djava.class.path=%s' % class_path)

    def close_jvm(self):
        import jpype
        jpype.shutdownJVM()

# ====================================================================================================

class DBHelpers(sl.Task):
    '''Mixin with helper methods for connecting to databases'''
    def connect_mysql(self, db_host, db_name, db_user, db_passwd):
        import MySQLdb as mydb
        connection = mydb.connect(host=db_host,
                         db=db_name,
                         user=db_user,
                         passwd=db_passwd)
        return connection.cursor()

    def connect_sqlite(self, db_filename):
        import pysqlite2.dbapi2 as sqlite
        connection = sqlite.connect(db_filename)
        return connection.cursor()

# ====================================================================================================

class ExistingFile(sl.ExternalTask):
    '''External task for getting hand on existing files'''

    # PARAMETERS
    file_path = luigi.Parameter()

    # TARGETS
    def out_file(self):
        dirpath = os.path.abspath('.')
        filepath = os.path.join(dirpath, self.file_path)
        return sl.TargetInfo(self, filepath)

# ====================================================================================================

class ExistingSmiles(sl.ExternalTask):
    '''External task for getting hand on existing smiles files'''

    # PARAMETERS
    dataset_name = luigi.Parameter()

    # TARGETS
    def out_smiles(self):
        datapath = os.path.abspath('./data')
        filename = self.dataset_name + '.smi'
        outfile_path = os.path.join(datapath, filename)
        smilestgt = sl.TargetInfo(self, outfile_path)
        return smilestgt

# ====================================================================================================

class Concatenate2Files(sl.Task):

    # INPUT TARGETS
    in_file1 = None
    in_file2 = None

    # TASK PARAMETERS
    skip_file1_header = luigi.BooleanParameter(default=True)
    skip_file2_header = luigi.BooleanParameter(default=True)

    def out_concatenated_file():
        return sl.TargetInfo(self, self.in_file1().path + '.concat')

    def run(self):
        inpath1 = self.in_file1().path
        inpath2 = self.in_file2().path
        outpath = self.out_concatenated_file().path
        with open(inpath1) as infile1, open(inpath2) as infile2, open(outpath, 'w') as outfile:
            # Write file 1, with or without header
            i = 1
            for line in infile1:
                if not (i == 1 and self.skip_file1_header):
                    outfile.write(line)
                i += 1

            # Write file 2, with or without header
            j = 1
            for line in infile2:
                if not (j == 1 and self.skip_file2_header):
                    outfile.write(line)
                j += 1

# ====================================================================================================

class GenerateSignaturesFilterSubstances(sl.SlurmTask):

    # TASK PARAMETERS
    min_height = luigi.IntParameter()
    max_height = luigi.IntParameter()
    silent_mode = luigi.BooleanParameter(default=True)

    # INPUT TARGETS
    in_smiles = None

    # DEFINE OUTPUTS
    def out_signatures(self):
        return sl.TargetInfo(self, self.in_smiles().path + '.h%d_%d.sign' % (self.min_height, self.max_height))

    # WHAT THE TASK DOES
    def run(self):
        self.ex(['java', '-jar', 'bin/GenerateSignatures.jar',
                '-inputfile', self.in_smiles().path,
                '-threads', self.slurminfo.threads,
                '-minheight', str(self.min_height),
                '-maxheight', str(self.max_height),
                '-outputfile', self.out_signatures().path,
                '-silent' if self.silent_mode else ''])
        self.ex_local(['touch', self.out_signatures().path])

# ====================================================================================================

class UnGzipFile(sl.SlurmTask):
    # TARGETS
    in_gzipped = None

    def out_ungzipped(self):
        return sl.TargetInfo(self, self.in_gzipped().path + '.ungz')

    def run(self):
        self.ex(['gunzip', '-c',
                  self.in_gzipped().path,
                  '>',
                  self.out_ungzipped().path])

# ====================================================================================================

class CreateRunCopy(sl.Task):

    # TASK PARAMETERS
    run_id = luigi.Parameter()

    # TARGETS
    in_file = None

    def out_copy(self):
        filedir = os.path.dirname(self.in_file().path)
        filename = os.path.basename(self.in_file().path)
        newdir = os.path.join(filedir, self.run_id)
        if not os.path.isdir(newdir):
            os.mkdir(newdir)
        return sl.TargetInfo(self, os.path.join(newdir, filename))

    def run(self):
        shutil.copy(self.in_file().path, self.out_copy().path)

# ====================================================================================================

class CreateReplicateCopy(sl.Task):

    # TASK PARAMETERS
    replicate_id = luigi.Parameter()

    # TARGETS
    in_file = None

    def out_copy(self):
        if self.in_file is None:
            raise Exception('In-port field in_file of CreateReplicateCopy is None')
        elif self.in_file() is None:
            raise Exception('In-port field in_file of CreateReplicateCopy return None')
        else:
            return sl.TargetInfo(self, self.in_file().path + '.' + self.replicate_id)

    def run(self):
        shutil.copy(self.in_file().path, self.out_copy().path)

# ====================================================================================================

class SampleTrainAndTest(sl.SlurmTask):

    # TASK PARAMETERS
    seed = luigi.Parameter(default=None)
    test_size = luigi.Parameter()
    train_size = luigi.Parameter()
    sampling_method = luigi.Parameter()
    replicate_id = luigi.Parameter()

    # INPORTS
    in_signatures = None

    # OUTPORTS
    def out_traindata(self):
        return sl.TargetInfo(self, self.get_basepath() + '_trn')
    def out_testdata(self):
        return sl.TargetInfo(self, self.get_basepath() + '_tst')
    def out_log(self):
        return sl.TargetInfo(self, self.get_basepath() + '_trn.log') # This is generated by the jar
    # OUTPORT Helper method
    def get_basepath(self):
        base_path = self.in_signatures().path + '.{test}_{train}_{method}'.format(
            test  = self.test_size.replace('%', 'proc'),
            train = self.train_size,
            method = self.sampling_method.replace('random', 'rand').replace('signature_count', 'signcnt'))
        return base_path

    # WHAT THE TASK DOES
    def run(self):
        test_temp_path  = self.out_testdata().path  + '.tmp'
        train_temp_path = self.out_traindata().path + '.tmp'

        jar_files = { 'random'          : 'SampleTrainingAndTest',
                      'signature_count' : 'SampleTrainingAndTestSizedBased' }
        jar_file = jar_files[self.sampling_method]

        cmd = ['java', '-jar', 'bin/' + jar_file + '.jar',
                     '-inputfile', self.in_signatures().path,
                     '-testfile', test_temp_path,
                     '-trainingfile', train_temp_path,
                     '-testsize', self.test_size,
                     '-trainingsize', self.train_size,
                     '-silent']
        if self.seed is not None and self.seed != 'None':
            cmd.extend(['-seed', self.seed])

        self.ex(cmd)

        # Restore temporary test and train files to their original file names
        shutil.move(test_temp_path,
                    self.out_testdata().path)
        shutil.move(train_temp_path,
                    self.out_traindata().path)
        shutil.move(self.out_traindata().path + '.tmp.log',
                    self.out_traindata().path + '.log')

# ====================================================================================================

class CreateSparseTrainDataset(sl.SlurmTask):

    # TASK PARAMETERS
    replicate_id = luigi.Parameter()

    # INPUT TARGETS
    in_traindata = None

    def out_sparse_traindata(self):
        return sl.TargetInfo(self, self.in_traindata().path + '.csr')

    def out_signatures(self):
        return sl.TargetInfo(self, self.in_traindata().path + '.signatures')

    def out_log(self):
        return sl.TargetInfo(self, self.in_traindata().path + '.csr.log')

    # WHAT THE TASK DOES
    def run(self):
        self.ex(['java', '-jar', 'bin/CreateSparseDataset.jar',
                '-inputfile', self.in_traindata().path,
                '-datasetfile', self.out_sparse_traindata().path,
                '-signaturesoutfile', self.out_signatures().path,
                '-silent'])

# ====================================================================================================

class CreateSparseTestDataset(sl.Task):

    # INPUT TARGETS
    in_testdata = None
    in_signatures = None

    # TASK PARAMETERS
    replicate_id = luigi.Parameter()
    java_path = luigi.Parameter

    # DEFINE OUTPUTS
    def out_sparse_testdata(self):
        return sl.TargetInfo(self, self.get_basepath()+ '.csr')
    def out_signatures(self):
        return sl.TargetInfo(self, self.get_basepath()+ '.signatures')
    def out_log(self):
        return sl.TargetInfo(self, self.get_basepath()+ '.csr.log')
    def get_basepath(self):
        return self.in_testdata().path

    # WHAT THE TASK DOES
    def run(self):
        self.ex(['java', '-jar', 'bin/CreateSparseDataset.jar',
                '-inputfile', self.in_testdata().path,
                '-signaturesinfile', self.in_signatures().path,
                '-datasetfile', self.out_sparse_testdata().path,
                '-signaturesoutfile', self.out_signatures().path,
                '-silent'])

# ====================================================================================================

class TrainSVMModel(sl.SlurmTask):

    # INPUT TARGETS
    in_traindata = None

    # TASK PARAMETERS
    replicate_id = luigi.Parameter()
    train_size = luigi.Parameter()
    svm_gamma = luigi.Parameter()
    svm_cost = luigi.Parameter()
    svm_type = luigi.Parameter()
    svm_kernel_type = luigi.Parameter()

    # Whether to run svm-train or pisvm-train when training
    parallel_train = luigi.BooleanParameter()

    # DEFINE OUTPUTS
    def out_model(self):
        return sl.TargetInfo(self, self.in_traindata().path + '.g{g}_c{c}_s{s}_t{t}.svm'.format(
            g = self.svm_gamma.replace('.', 'p'),
            c = self.svm_cost,
            s = self.svm_type,
            t = self.svm_kernel_type))

    def out_traintime(self):
        return sl.TargetInfo(self, self.out_model().path + '.extime')

    # WHAT THE TASK DOES
    def run(self):
        '''
        Determine pisvm parameters based on training set size
        Details from Ola and Marcus:

        size         o    q
        -------------------
        <1k:       100  100
        1k-5k:     512  256
        5k-40k    1024 1024
        >40k      2048 2048
        '''

        train_size = self.train_size
        if train_size == 'rest':
            o = 2048
            q = 2048
        else:
            trainsize_num = int(train_size)
            if trainsize_num < 100:
                o = 10
                q = 10
            elif 100 <= trainsize_num < 1000:
                o = 100
                q = 100
            elif 1000 <= trainsize_num < 5000:
                o = 512
                q = 256
            elif 5000 <= trainsize_num < 40000:
                o = 1024
                q = 1024
            elif 40000 <= trainsize_num:
                o = 2048
                q = 2048
            else:
                raise Exception('Trainingsize {s} is not "rest" nor a valid positive number!'.format(s = trainsize_num))

        # Set some file paths
        trainfile = self.in_traindata().path
        svmmodel_file = self.out_model().path

        # Select train command based on parameter
        if self.parallel_train:
            self.ex(['/usr/bin/time', '-f%e', '-o',
                    self.out_traintime().path,
                    'bin/pisvm-train',
                    '-o', str(o),
                    '-q', str(q),
                    '-s', self.svm_type,
                    '-t', self.svm_kernel_type,
                    '-g', self.svm_gamma,
                    '-c', self.svm_cost,
                    '-m', '2000',
                    self.in_traindata().path,
                    svmmodel_file,
                    '>',
                    '/dev/null']) # Needed, since there is no quiet mode in pisvm :/
        else:
            self.ex(['/usr/bin/time', '-f%e', '-o',
                self.out_traintime().path,
                'bin/svm-train',
                '-s', self.svm_type,
                '-t', self.svm_kernel_type,
                '-g', self.svm_gamma,
                '-c', self.svm_cost,
                '-m', '2000',
                '-q', # quiet mode
                self.in_traindata().path,
                svmmodel_file])

# ====================================================================================================

class TrainLinearModel(sl.SlurmTask):
    # INPUT TARGETS
    in_traindata = None

    # TASK PARAMETERS
    replicate_id = luigi.Parameter()
    lin_type = luigi.Parameter() # 0 (regression)
    lin_cost = luigi.Parameter() # 100
    # Let's wait with implementing these
    #lin_epsilon = luigi.Parameter()
    #lin_bias = luigi.Parameter()
    #lin_weight = luigi.Parameter()
    #lin_folds = luigi.Parameter()

    # Whether to run normal or distributed lib linear
    #parallel_train = luigi.BooleanParameter()

    # DEFINE OUTPUTS
    def out_model(self):
        return sl.TargetInfo(self, self.in_traindata().path + '.s{s}_c{c}.linmdl'.format(
            s = self.lin_type,
            c = self.lin_cost))

    def out_traintime(self):
        return sl.TargetInfo(self, self.out_model().path + '.extime')

    # WHAT THE TASK DOES
    def run(self):
        #self.ex(['distlin-train',
        self.ex(['/usr/bin/time', '-f%e', '-o',
            self.out_traintime().path,
            'bin/lin-train',
            '-s', self.lin_type,
            '-c', self.lin_cost,
            '-q', # quiet mode
            self.in_traindata().path,
            self.out_model().path])

# ====================================================================================================

class PredictSVMModel(sl.Task):
    # INPUT TARGETS
    in_svmmodel = None
    in_sparse_testdata = None
    replicate_id = luigi.Parameter()

    # TASK PARAMETERS
    testdata_gzipped = luigi.BooleanParameter(default=True)

    # DEFINE OUTPUTS
    def out_prediction(self):
        return sl.TargetInfo(self, self.in_svmmodel().path + '.pred')

    # WHAT THE TASK DOES
    def run(self):
        # Run prediction
        self.ex(['bin/svm-predict',
                self.in_sparse_testdata().path,
                self.in_svmmodel().path,
                self.out_prediction().path])

# ====================================================================================================

class PredictLinearModel(sl.Task):
    # INPUT TARGETS
    in_model = None
    in_sparse_testdata = None

    # TASK PARAMETERS
    replicate_id = luigi.Parameter()

    # DEFINE OUTPUTS
    def out_prediction(self):
        return sl.TargetInfo(self, self.in_model().path + '.pred')

    # WHAT THE TASK DOES
    def run(self):
        self.ex(['bin/lin-predict',
            self.in_sparse_testdata().path,
            self.in_model().path,
            self.out_prediction().path])

# ====================================================================================================

class AssessLinearRMSD(sl.Task): # TODO: Check with Jonalv whether RMSD is what we want to do?!!
    # Parameters
    lin_cost = luigi.Parameter()

    # INPUT TARGETS
    in_model = None
    in_sparse_testdata = None
    in_prediction = None

    # DEFINE OUTPUTS
    def out_assessment(self):
        return sl.TargetInfo(self, self.in_prediction().path + '.rmsd')

    # WHAT THE TASK DOES
    def run(self):
        with self.in_sparse_testdata().open() as testfile:
            with self.in_prediction().open() as predfile:
                squared_diffs = []
                for tline, pline in zip(testfile, predfile):
                    test = float(tline.split(' ')[0])
                    pred = float(pline)
                    squared_diff = (pred-test)**2
                    squared_diffs.append(squared_diff)
        rmsd = math.sqrt(sum(squared_diffs)/len(squared_diffs))
        rmsd_records = {'rmsd': rmsd,
                        'cost': self.lin_cost}
        with self.out_assessment().open('w') as assessfile:
            sl.util.dict_to_recordfile(assessfile, rmsd_records)

# ====================================================================================================

class AssessSVMRMSD(sl.Task):
    # Parameters
    svm_cost = luigi.Parameter()
    svm_gamma = luigi.Parameter()
    svm_type = luigi.Parameter()
    svm_kernel_type = luigi.Parameter()

    # INPUT TARGETS
    in_model = None
    in_sparse_testdata = None
    in_prediction = None

    # DEFINE OUTPUTS
    def out_assessment(self):
        return sl.TargetInfo(self, self.in_prediction().path + '.rmsd')

    # WHAT THE TASK DOES
    def run(self):
        with self.in_sparse_testdata().open() as testfile:
            with self.in_prediction().open() as predfile:
                squared_diffs = []
                for tline, pline in zip(testfile, predfile):
                    test = float(tline.split(' ')[0])
                    pred = float(pline)
                    squared_diff = (pred-test)**2
                    squared_diffs.append(squared_diff)
        rmsd = math.sqrt(sum(squared_diffs)/len(squared_diffs))
        rmsd_records = {'rmsd': rmsd,
                        'svm_cost': self.svm_cost,
                        'svm_gamma': self.svm_gamma,
                        'svm_type': self.svm_type,
                        'svm_kernel_type': self.svm_kernel_type}
        with self.out_assessment().open('w') as assessfile:
            sl.util.dict_to_recordfile(assessfile, rmsd_records)

# ====================================================================================================

class CollectDataReportRow(sl.Task):
    dataset_name = luigi.Parameter()
    train_method = luigi.Parameter()
    train_size = luigi.Parameter()
    replicate_id = luigi.Parameter()
    lin_cost = luigi.Parameter()

    in_rmsd = None
    in_traintime = None
    in_trainsize_filtered = None

    def out_datareport_row(self):
        outdir = os.path.dirname(self.in_rmsd().path)
        return sl.TargetInfo(self, os.path.join(outdir, '{ds}_{lm}_{ts}_{ri}_datarow.txt'.format(
                    ds=self.dataset_name,
                    lm=self.train_method,
                    ts=self.train_size,
                    ri=self.replicate_id
                )))

    def run(self):
        with self.in_rmsd().open() as rmsdfile:
            rmsddict = sl.recordfile_to_dict(rmsdfile)
            rmsd = rmsddict['rmsd']

        with self.in_traintime().open() as traintimefile:
            train_time_sec = traintimefile.read().rstrip('\n')

        with self.in_trainsize_filtered().open() as trainsizefile:
            train_size_filtered = trainsizefile.read().strip('\n')

        if self.lin_cost is not None:
            lin_cost = self.lin_cost
        else:
            lin_cost = 'NA'

        with self.out_datareport_row().open('w') as outfile:
            rdata = { 'dataset_name': self.dataset_name,
                      'train_method': self.train_method,
                      'train_size': self.train_size,
                      'train_size_filtered': train_size_filtered,
                      'replicate_id': self.replicate_id,
                      'rmsd': rmsd,
                      'train_time_sec': train_time_sec,
                      'lin_cost': lin_cost}
            sl.dict_to_recordfile(outfile, rdata)

# ====================================================================================================

class CollectDataReport(sl.Task):
    dataset_name = luigi.Parameter()
    train_method = luigi.Parameter()

    in_datareport_rows = None

    def out_datareport(self):
        outdir = os.path.dirname(self.in_datareport_rows[0]().path)
        return sl.TargetInfo(self, os.path.join(outdir, '{ds}_{tm}_datareport.csv'.format(
                    ds=self.dataset_name,
                    tm=self.train_method
               )))

    def run(self):
        with self.out_datareport().open('w') as outfile:
            csvwrt = csv.writer(outfile)
            # Write header
            csvwrt.writerow(['dataset_name',
                             'train_method',
                             'train_size',
                             'train_size_filtered',
                             'replicate_id',
                             'rmsd',
                             'train_time_sec',
                             'lin_cost'])
            # Write data rows
            for intargetinfofunc in self.in_datareport_rows:
                with intargetinfofunc().open() as infile:
                    r = sl.recordfile_to_dict(infile)
                    csvwrt.writerow([r['dataset_name'],
                                     r['train_method'],
                                     r['train_size'],
                                     r['train_size_filtered'],
                                     r['replicate_id'],
                                     r['rmsd'],
                                     r['train_time_sec'],
                                     r['lin_cost']])

# ====================================================================================================

class CalcAverageRMSDForCost(sl.Task): # TODO: Check with Jonalv whether RMSD is what we want to do?!!
    # Parameters
    lin_cost = luigi.Parameter()

    # Inputs
    in_assessments = None

    # output
    def out_rmsdavg(self):
        return sl.TargetInfo(self, self.in_assessments[0]().path + '.avg')

    def run(self):
        vals = []
        for invalfun in self.in_assessments:
            infile = invalfun().open()
            records = sl.util.recordfile_to_dict(infile)
            vals.append(float(records['rmsd']))
        rmsdavg = sum(vals)/len(vals)
        rmsdavg_records = {'rmsd_avg': rmsdavg,
                           'cost': self.lin_cost}
        with self.out_rmsdavg().open('w') as outfile:
            sl.util.dict_to_recordfile(outfile, rmsdavg_records)

# ====================================================================================================

class SelectLowestRMSD(sl.Task):
    # Inputs
    in_values = None

    # output
    def out_lowest(self):
        cost_part = '.c' + hashlib.md5('_'.join([v().task.lin_cost for v in self.in_values])).hexdigest()
        return sl.TargetInfo(self, self.in_values[0]().path + cost_part + '.min')

    def run(self):
        vals = []
        for invalfun in self.in_values:
            infile = invalfun().open()
            records = sl.util.recordfile_to_dict(infile)
            vals.append(records)

        lowest_rmsd = float(min(vals, key=lambda v: float(v['rmsd_avg']))['rmsd_avg'])
        vals_lowest_rmsd = [v for v in vals if float(v['rmsd_avg']) <= lowest_rmsd]
        val_lowest_rmsd_cost = min(vals_lowest_rmsd, key=lambda v: v['cost'])
        lowestrec = {'lowest_rmsd_avg': val_lowest_rmsd_cost['rmsd_avg'],
                     'lowest_cost': val_lowest_rmsd_cost['cost']}
        with self.out_lowest().open('w') as lowestfile:
            sl.util.dict_to_recordfile(lowestfile, lowestrec)

# ====================================================================================================

# *** DEPRECATED: Use AssessSVMRMSD instead! ***

#class AssessSVMRegression(sl.Task):
#
#    # INPUT TARGETS
#    in_svmmodel = None
#    in_sparse_testdata = None
#    in_prediction = None
#
#    # TASK PARAMETERS
#    replicate_id = luigi.Parameter()
#    testdata_gzipped = luigi.BooleanParameter(default=True)
#
#    # DEFINE OUTPUTS
#    def out_plot(self):
#        return sl.TargetInfo(self, self.in_svmmodel().path + '.pred.png')
#    def out_log(self):
#        return sl.TargetInfo(self, self.in_svmmodel().path + '.pred.log')
#
#    # WHAT THE TASK DOES
#    def run(self):
#        # Run Assess
#        self.ex(['/usr/bin/xvfb-run /sw/apps/R/x86_64/3.0.2/bin/Rscript assess/assess.r',
#                '-p', self.in_prediction().path,
#                '-t', self.in_sparse_testdata().path])

# ====================================================================================================

class CreateHtmlReport(sl.Task):

    # TASK PARAMETERS
    dataset_name = luigi.Parameter()
    train_size = luigi.Parameter()
    test_size = luigi.Parameter()
    svm_cost = luigi.Parameter()
    svm_gamma = luigi.Parameter()
    replicate_id = luigi.Parameter()
    accounted_project = luigi.Parameter()

    # INPUT TARGETS
    in_signatures = None
    in_sample_traintest_log = None
    in_sparse_testdata_log = None
    in_sparse_traindata_log = None
    in_traindata = None
    in_svmmodel = None
    in_assess_svm_log = None
    in_assess_svm_plot = None

    # DEFINE OUTPUTS
    def out_html_report(self):
        return sl.TargetInfo(self, self.in_assess_svm_log().path + '.report.html')

    def get_svm_rmsd(self):
        # Figure out svm_rmsd
        with self.in_assess_svm_log().open() as infile:
            csv_reader = csv.reader(infile)
            for row in csv_reader:
                if row[0] == 'RMSD':
                    svm_rmsd = float(row[1])
        return svm_rmsd


    def get_html_report_content_old(self):
        '''Create and return the content of the HTML report'''
        output_html = '<html><body style=\'font-family: arial, helvetica, sans-serif;\'><h1>Report for dataset {dataset}</h1>\n'.format(
                dataset=self.dataset_name)

        # Get hand on some log files that we need to create the HTML report
        log_files = {}
        log_files['Sample_Train_and_Test'] = self.in_sample_traintest_log().path
        log_files['Create_Sparse_Test_Dataset'] = self.in_sparse_testdata_log().path
        log_files['Create_Sparse_Train_Dataset'] = self.in_sparse_traindata_log().path
        log_files['Predict_SVM'] = self.in_assess_svm_log().path

        for name, path in log_files.iteritems():
            output_html += self.tag('h2', name.replace('_',' '))
            output_html += '<ul>\n'
            with open(path, 'r') as infile:
                tsv_reader = csv.reader(infile, delimiter=',')
                for row in tsv_reader:
                    output_html += '<li><strong>{k}:</strong> {v}</li>\n'.format(
                            k=row[0],
                            v=row[1])
            output_html += '</ul>\n'
        output_html += '<ul>\n'
        output_html += '</body></html>'
        return output_html

    def get_html_report_content(self):
        log_files = {}
        log_files['sample_train_and_test'] = self.in_sample_traintest_log().path
        log_files['create_sparse_test_dataset'] = self.in_sparse_testdata_log().path
        log_files['create_sparse_train_dataset'] = self.in_sparse_traindata_log().path
        log_files['predict_svm'] = self.in_assess_svm_log().path

        # Initialize an empty dict where to store information to show in the report
        report_info = {}

        # Loop over the log files to gather info in to dict structure
        for logfile_name, path in log_files.iteritems():
            report_info[logfile_name] = {}
            with open(path, 'r') as infile:
                tsv_reader = csv.reader(infile, delimiter=',')
                for key, val in tsv_reader:
                    report_info[logfile_name][key] = val

        html_head = textwrap.dedent('''
            <!DOCTYPE html PUBLIC '-//W3C//DTD HTML 4.01 Transitional//EN'>
            <html>
            <head>
              <title>Luigi HTML report / help file</title>
              <style type='text/css'>
                ol{margin:0;padding:0}
                body{font-family:'Arial'}
                li{font-size:11pt;}
                p{font-size:11pt;margin:0;}
                h1{font-size:16pt;}
                h2{font-size:13pt;}
                h3{font-size:12pt;}
                h4{font-size:11pt;}
                h5{font-size:11pt;}
                h6{font-style:italic;font-size:11pt;}
                th{background: #efefef;padding: 4px 12px;}
                td{padding: 2px 12px;}
              </style>
            </head>''').strip()
        html_content = textwrap.dedent('''
            <body>
              <h1>QSAR model for $foo</h1>
              <p>The $foo model is a QSAR model for predicting $foo based on a set of
              substances with that property extracted from the ChEMBL database.</p>
              <table cellpadding='0' cellspacing='0'>
                <tbody>
                  <tr>
                    <th colspan='2'>Dataset</th>
                  </tr>
                  <tr>
                    <td>Training set size</td>
                    <td>{train_size}</td>
                  </tr>
                  <tr>
                    <td>Test set size</td>
                    <td>{test_size}</td>
                  </tr>
                  <tr>
                    <td>Minimum number of non hydrogen atoms</td>
                    <td>{min_nonh_atoms}</td>
                  </tr>
                  <tr>
                    <td>Maximum number of non hydrogen atoms</td>
                    <td>{max_nonh_atoms}</td>
                  </tr>
                  <tr>
                    <td>Number of substances removed during filtering</td>
                    <td>{filtered_substances_count}</td>
                  </tr>
                  <tr>
                    <th colspan='2'>Descriptors</th>
                  </tr>
                  <tr>
                    <td>Descriptor type</td>
                    <td>Faulon Signatures</td>
                  </tr>
                  <tr>
                    <td>Faulon signatures heights</td>
                    <td>{sign_height_min}-{sign_height_max}</td>
                  </tr>
                  <tr>
                    <td>Faulon signatures generation running time</td>
                    <td>{sign_gen_runtime}</td>
                  </tr>
                  <tr>
                    <td>Faulon signatures sparse training data set generation running time</td>
                    <td>{sparse_train_gen_runtime}</td>
                  </tr>
                  <tr>
                    <td>Faulon signatures sparse test data set generation running time</td>
                    <td>{sparse_test_gen_runtime}</td>
                  </tr>
                  <tr>
                    <td>Total number of unique Faulon signatures</td>
                    <td>{signatures_count}</td>
                  </tr>
                  <tr>
                    <th colspan='2'>Model</th>
                  </tr>
                  <tr>
                    <td>Modelling method</td>
                    <td>RBF-kernel SVM</td>
                  </tr>
                  <tr>
                    <td>RMSD test set</td>
                    <td>{testset_rmsd}</td>
                  </tr>
                  <tr>
                    <td>Model choice</td>
                    <td>Maximal accuracy, with 5-fold cross-validated accuracy (RMSD) as objective function</td>
                  </tr>
                  <tr>
                    <td>Model validation</td>
                    <td>Accuracy measued on an external test set</td>
                  </tr>
                  <tr>
                    <td>Learning parameters</td>
                    <td>kernel=RBF, c={svm_cost}, gamma={svm_gamma}</td>
                  </tr>
                  <tr>
                    <td colspan='2' ><img src=\'{png_image_path}\' style='width:400px;height:400px;margin:2em 7em;border: 1px solid #ccc;'></td>
                  </tr>
                </tbody>
              </table>
            </body>
            </html>
        '''.format(
            train_size=self.train_size,
            test_size=self.test_size,
            min_nonh_atoms='5',
            max_nonh_atoms='50',
            filtered_substances_count='N/A',
            sign_height_min='0',
            sign_height_max='3',
            sign_gen_runtime=report_info['sample_train_and_test']['runningtime'],
            sparse_train_gen_runtime=report_info['create_sparse_train_dataset']['runningtime'],
            sparse_test_gen_runtime=report_info['create_sparse_train_dataset']['runningtime'],
            signatures_count=report_info['create_sparse_train_dataset']['numsign'],
            testset_rmsd=self.get_svm_rmsd(),
            svm_cost=self.svm_cost,
            svm_gamma=self.svm_gamma,
            png_image_path=self.in_assess_svm_plot().path.split('/')[-1]
        )).strip()

        return html_head + html_content


    # SOME HELPER METHODS
    def tag(self, tagname, content):
        return '<{t}>{c}</{t}>'.format(t=tagname, c=content)


    # WHAT THE TASK DOES
    def run(self):

        # WRITE HTML REPORT
        with open(self.out_html_report().path, 'w') as html_report_file:
            log.info('Writing HTML report to file: ' + self.out_html_report().path)
            html_report_file.write(self.get_html_report_content())


# ====================================================================================================

class CreateElasticNetModel(sl.Task):

    # INPUT TARGETS
    in_traindata = None

    # TASK PARAMETERS
    l1_value = luigi.Parameter()
    lambda_value = luigi.Parameter()
    java_path = luigi.Parameter()

    # DEFINE OUTPUTS
    def out_model(self):
        return sl.TargetInfo(self, self.in_traindata().path + '.model_{l}_{y}'.format(
            l=self.get_value('l1_value'),
            y=self.get_value('lambda_value')
        ))

    def run(self):
        self.ex(['java', '-jar', 'bin/CreateElasticNetModel.jar',
                '-inputfile', self.in_traindata().path,
                '-l1ratio', str(self.get_value('l1_value')),
                '-lambda', str(self.get_value('lambda_value')),
                '-outputfile', self.out_model().path,
                '-silent'])

        #self.ex_local(['mv',
        #         self.in_traindata().path + '.model',
        #         self.in_traindata().path + '.model_{l}_{y}'.format(l=self.get_value('l1_value'),y=self.get_value('lambda_value'))])


# ====================================================================================================

class PredictElasticNetModel(sl.Task):

    # INPUT TARGETS
    in_elasticnet_model = None
    in_testdata = None

    # TASK PARAMETERS
    l1_value = luigi.Parameter()
    lambda_value = luigi.Parameter()
    java_path = luigi.Parameter()

    def out_prediction(self):
        return sl.TargetInfo(self, self.in_elasticnet_model().path + '.pred')

    def run(self):
        self.ex(['java', '-jar', 'bin/PredictElasticNetModel.jar',
                '-modelfile', self.in_elasticnet_model().path,
                '-testset', self.in_testdata().path,
                '-outputfile', self.out_prediction().path,
                '-silent'])

# ====================================================================================================

class EvaluateElasticNetPrediction(sl.Task):
     # INPUT TARGETS
     in_testdata = None
     in_prediction = None

     # TASK PARAMETERS
     l1_value = luigi.Parameter()
     lambda_value = luigi.Parameter()

     # DEFINE OUTPUTS
     def out_evaluation(self):
         return sl.TargetInfo(self,  self.in_prediction().path + '.evaluation' )

     # WHAT THE TASK DOES
     def run(self):
         with gzip.open(self.in_testdata().path) as testset_file, self.in_prediction().open() as prediction_file:
             original_vals = [float(line.split(' ')[0]) for line in testset_file]
             predicted_vals = [float(val.strip('\n')) for val in prediction_file]
         squared = [(pred-orig)**2 for orig, pred in zip(original_vals, predicted_vals)]
         rmsd = math.sqrt( sum(squared) / len(squared) )
         with self.out_evaluation().open('w') as outfile:
             csvwriter = csv.writer(outfile)
             csvwriter.writerow(['rmsd', rmsd])
             csvwriter.writerow(['l1ratio', self.get_value('l1_value')])
             csvwriter.writerow(['lambda', self.get_value('lambda_value')])

# ====================================================================================================

class ElasticNetGridSearch(sl.Task):

    # INPUT TARGETS
    in_traindata = None
    in_testdata = None
    replicate_id = luigi.Parameter()

    # TASK PARAMETERS
    l1_steps = luigi.Parameter()
    lambda_steps = luigi.Parameter()

    def grid_step_generator(self):
        for l1 in ast.literal_eval(self.l1_steps):
            for lambda_value in ast.literal_eval(self.lambda_steps):
                create_elasticnet_model = CreateElasticNetModel(
                        traindata_target = self.traindata_target,
                        l1_value = l1,
                        lambda_value = lambda_value,
                        dataset_name = self.dataset_name,
                        replicate_id = self.replicate_id,
                        accounted_project = self.accounted_project )
                predict_elasticnet_model = PredictElasticNetModel(
                        l1_value = l1,
                        lambda_value = lambda_value,
                        elasticnet_model_target =
                            { 'upstream' : { 'task' : create_elasticnet_model,
                                             'port' : 'model' } },
                        testdata_target = self.testdata_target,
                        dataset_name = self.dataset_name,
                        replicate_id = self.replicate_id,
                        accounted_project = self.accounted_project )
                eval_elasticnet_prediction = EvaluateElasticNetPrediction(
                        l1_value = l1,
                        lambda_value = lambda_value,
                        testdata_target = self.testdata_target,
                        prediction_target =
                            { 'upstream' : { 'task' : predict_elasticnet_model,
                                             'port' : 'prediction' } },
                        dataset_name = self.dataset_name,
                        replicate_id = self.replicate_id,
                        accounted_project = self.accounted_project )
                yield eval_elasticnet_prediction

    def requires(self):
        return [x for x in self.grid_step_generator()]

    def out_optimal_parameters_info(self):
        inpath = self.input()[0]['evaluation'].path
        return sl.TargetInfo(self, inpath +
                              '.gridsearch_l1_' +
                              '_'.join([str(x).replace('.','-') for x in ast.literal_eval(self.l1_steps)]) +
                              '_lambda_' +
                              '_'.join([str(x).replace('.','-') for x in ast.literal_eval(self.lambda_steps)]) +
                              '_optimal_params' )

    def run(self):
        rmsd_infos = []
        for rmsd_task in self.input():
            rmsd_info_obj = ConfigParser()
            rmsd_info_obj.read(rmsd_task['evaluation'].path)
            rmsd_info = dict(rmsd_info_obj.items('evaluation_info'))
            rmsd_infos.append(rmsd_info)

        best_rmsd_info = min(rmsd_infos, key = lambda x: float(x['rmsd']))
        log.info( 'BEST RMSD INFO: ' + str(best_rmsd_info) )

        with self.out_optimal_parameters_info().open('w') as outfile:
            info_obj = ConfigParser()
            info_obj.add_section('optimal_parameters')
            for key in best_rmsd_info.keys():
                info_obj.set('optimal_parameters', key, best_rmsd_info[key])
            info_obj.write(outfile)

# ====================================================================================================

class BuildP2Sites(sl.Task):

    # INPUT TARGETS
    in_signatures = None
    in_sparse_traindata = None
    in_svmmodel = None
    in_assess_svm_log = None

    # TASK PARAMETERS
    dataset_name = luigi.Parameter()
    accounted_project = luigi.Parameter()
    test_size = luigi.Parameter()
    svm_cost = luigi.Parameter()
    svm_gamma = luigi.Parameter()
    java_path = luigi.Parameter()

    def out_plugin_bundle(self):
        return sl.TargetInfo(self, self.temp_folder_path() + '/%s_p2_site.zip' % self.dataset_name)

    def temp_folder_path(self):
        return 'data/' + '/'.join(self.in_svmmodel().path.split('/')[-2:]) + '.plugin_bundle'

    def temp_folder_abspath(self):
        return os.path.abspath(self.in_svmmodel().path + '.plugin_bundle')

    def get_svm_rmsd(self):

        with self.in_assess_svm_log().open() as infile:
            csv_reader = csv.reader(infile)
            for row in csv_reader:
                if row[0] == 'RMSD':
                    svm_rmsd = float(row[1])
        return svm_rmsd

    def get_signatures_count(self):
        signatures_count = 0
        with self.in_signatures().open() as infile:
            for line in infile:
                signatures_count += 1
        return signatures_count

    def get_properties_file_contents(self, signatures_file, svmmodel_file):
        return textwrap.dedent('''
            <?xml version='1.0' encoding='UTF-8'?>
            <?eclipse version='3.4'?>
            <plugin>
               <extension
                    point='net.bioclipse.decisionsupport'>

                <test
                        id='net.bioclipse.mm.{dataset_name}'
                        name='Model predicted {dataset_name}'
                        class='net.bioclipse.ds.libsvm.SignaturesLibSVMPrediction'
                        endpoint='net.bioclipse.mm'
                        informative='false'>

                        <parameter name='isClassification' value='false'/>

                        <parameter name='signatures.min.height' value='1'/>
                        <parameter name='signatures.max.height' value='3'/>
                        <resource name='modelfile' path='{svmmodel_file}'/>
                        <resource name='signaturesfile' path='{signatures_file}'/>
                        <parameter name='Model type' value='QSAR'/>
                        <parameter name='Learning model' value='SVM'/>

                        <parameter name='Model performance' value='{svm_rmsd}'/>
                        <parameter name='Model choice' value='External test set of {test_size} observations'/>
                        <parameter name='Learning parameters' value='kernel=RBF, c={svm_cost}, gamme={svm_gamma}'/>
                        <parameter name='Descriptors' value='Signatures (height 1-3)'/>
                        <parameter name='Observations' value='{test_size}'/>
                        <parameter name='Variables' value='{signatures_count}'/>

                        <parameter name='lowPercentile' value='0'/>
                        <parameter name='highPercentile' value='1'/>
                    </test>
               </extension>
            </plugin>
        '''.format(
                dataset_name=self.dataset_name,
                svmmodel_file=svmmodel_file,
                signatures_file=signatures_file,
                svm_rmsd=self.get_svm_rmsd(),
                test_size=self.test_size,
                svm_cost=self.svm_cost,
                svm_gamma=self.svm_gamma,
                signatures_count=self.get_signatures_count()
            )).strip()

    def run(self):
        temp_folder = self.temp_folder_path()
        temp_folder_abspath = self.temp_folder_abspath()
        model_folder = 'model'
        model_folder_abspath = temp_folder_abspath + '/model'

        # Create temp and model folder
        self.ex_local(['mkdir -p', model_folder_abspath])

        # Copy some files into the newly created model folder
        # (which is in turn inside the newly created temp folder)

        signatures_file = model_folder + '/signatures.txt'
        signatures_file_abspath = model_folder_abspath + '/signatures.txt'
        self.ex_local(['cp',
                 self.in_signatures().path,
                 signatures_file_abspath])

        traindata_file = model_folder + '/sparse_train_datset.csr'
        traindata_file_abspath = model_folder_abspath + '/sparse_train_datset.csr'
        self.ex_local(['cp',
                 self.in_sparse_traindata().path,
                 traindata_file_abspath])

        svmmodel_file = model_folder + '/model.svm'
        svmmodel_file_abspath = model_folder_abspath + '/model.svm'
        self.ex_local(['cp',
                 self.in_svmmodel().path,
                 svmmodel_file_abspath])


        # -----------------------------------------------------------------------
        # PROPERTIES FILE

        properties_file_contents = self.get_properties_file_contents(signatures_file, svmmodel_file)
        with open(temp_folder + '/plugin.xml', 'w') as pluginxml_file:
            pluginxml_file.write(properties_file_contents)

        # Zip the files (Has to happen after writing of plugin.xml, in order to include it)
        cmd = ['cd', temp_folder, ';',
               'zip -r ', 'plugin_bundle.zip', './*']
        self.ex_local(cmd)

        # -----------------------------------------------------------------------
        # ENDPOINT FILE

        # Create Endpoint XML file
        endpoint_xmlfile_content = textwrap.dedent('''
            <?xml version='1.0' encoding='UTF-8'?>
            <?eclipse version='3.4'?>
            <plugin>
               <extension
                     point='net.bioclipse.decisionsupport'>
                     <endpoint
                           id='net.bioclipse.mm'
                           description='Predicted logp based on acd logP'
                           icon='biolock.png'
                           name='Predicted Properties'>
                     </endpoint>
                </extension>
            </plugin>
        ''').strip()
        with open(temp_folder + '/endpoint_bundle.xml','w') as endpoint_xmlfile:
            endpoint_xmlfile.write(endpoint_xmlfile_content)

        # Create Endpoint BND file
        endpoint_bndfile_content = textwrap.dedent('''
            Bundle-Version:1.0.0
            Bundle-SymbolicName: net.bioclipse.mm.endpoint;singleton:=true
            -includeresource: plugin.xml=endpoint_bundle.xml
            -output source.p2/plugins/${bsn}-${Bundle-Version}.jar
        ''').strip()
        with open(temp_folder + '/endpoint_bundle.bnd','w') as endpoint_bndfile:
            endpoint_bndfile.write(endpoint_bndfile_content)

        # Process Endpoint
        self.ex_local(['cd', temp_folder, ';',
                'java', '-jar', 'bin/bnd-2.3.0.jar',
                'endpoint_bundle.bnd'])


        # -----------------------------------------------------------------------
        # PLUGIN FILE

        time_stamp = time.strftime('%Y%m%d%H%M%S')
        bundle_version = '0.0.0.' + time_stamp
        bundle_id = 'net.bioclipse.mm.' + self.dataset_name

        # Create Plugin BND file
        plugin_bndfile_content = textwrap.dedent('''
            Bundle-Version:{bundle_version}
            Bundle-SymbolicName: {bundle_id};singleton:=true
            -includeresource: @plugin_bundle.zip
            -output source.p2/plugins/${{bsn}}-${{Bundle-Version}}.jar
            Require-Bundle: net.bioclipse.ds.libsvm
        '''.format(
            bundle_version = bundle_version,
            bundle_id = bundle_id
        )).strip()
        with open(temp_folder + '/plugin_bundle.bnd','w') as plugin_bndfile:
            plugin_bndfile.write(plugin_bndfile_content)

        # Process
        self.ex_local(['cd', temp_folder, ';',
                 'java', '-jar', 'bin/bnd-2.3.0.jar',
                 'plugin_bundle.bnd'])

        # Create feature file

        features_file_content = textwrap.dedent('''
        <?xml version='1.0' encoding='UTF-8'?>
        <feature id='{bundle_id}' label='{feature_label}' version='{bundle_version}' provider-name='Bioclipse' plugin='{bundle_id}'>
        <description></description>
        <copyright></copyright>
        <license url=''></license>

        <requires>
        <import plugin='net.bioclipse.ds.libsvm' version='2.6.2' match='greaterOrEqual'/>
        </requires>

        <plugin id='{bundle_id}' download-size='0' install-size='0' version='{bundle_version}' unpack='false'/>
        <plugin id='net.bioclipse.mm.endpoint' download-size='0' install-size='0' version='1.0.0' unpack='false'/>

        </feature>
        '''.format(bundle_id = bundle_id,
                   bundle_version = bundle_version,
                   feature_label = self.dataset_name)).strip()

        # Create a folder for features
        features_folder = temp_folder + '/source.p2/features'
        self.ex_local(['mkdir -p', features_folder])

        # Write out the content of the feature.xml file
        with open(temp_folder + '/feature.xml', 'w') as features_file:
            features_file.write(features_file_content)

        # Zip the feature.xml file, into a file in the features folder
        self.ex_local(['pwd; cd', temp_folder, ';',
                 'zip', 'source.p2/features/' + bundle_id + '.jar', 'feature.xml'])

        # -----------------------------------------------------------------------
        # Assemble it all together
        pubcmd = ['cd', temp_folder, ';',
                  '/proj/b2013262/nobackup/eclipse_director/director/director',
                  '-application', 'org.eclipse.equinox.p2.publisher.FeaturesAndBundlesPublisher',
                  '-metadataRepository', 'file:' + temp_folder_abspath + '/site.p2',
                  '-artifactRepository', 'file:' + temp_folder_abspath + '/site.p2',
                  '-metadataRepositoryName', '"MM SVM Model for ' + self.dataset_name + '"',
                  '-source', temp_folder_abspath + '/source.p2',
                  '-publishArtifacts']
        self.ex_local(pubcmd)

        zipcmd = ['cd', temp_folder + '/site.p2;',
                  'zip -r', '../%s_p2_site.zip' % self.dataset_name,
                  './*']
        self.ex_local(zipcmd)

# ====================================================================================================

class PushP2SiteToRemoteHost(sl.Task):
    # INPUT TARGETS
    in_plugin_bundle = None

    # TASK PARAMETERS
    remote_host = luigi.Parameter()
    remote_user = luigi.Parameter()
    remote_base_folder = luigi.Parameter()

    def out_completion_marker(self):
        return sl.TargetInfo(self, self.in_plugin_bundle().path + '.p2site_pushed' )

    def run(self):
        remote_folder = self.remote_base_folder + '/' + self.replicate_id
        remote_command = 'mkdir -p ' + remote_folder

        self.ex_local(['ssh -o PubkeyAuthentication=no',
                 '%s@%s' % (self.remote_user, self.remote_host),
                 '\'' + remote_command + '\''])

        # Copy the p2 site zip file to the remote host via SCP
        self.ex_local(['scp',
                 self.in_plugin_bundle().path,
                 '%s@%s:%s/' % (self.remote_user,
                                self.remote_host,
                                remote_folder)])

        # Write some dummy content to the completion marker
        self.ex_local(['echo',
                 '"p2 site pushed"',
                 '>',
                 self.out_completion_marker().path])

# ====================================================================================================

class BuildP2SiteOnRemoteHost(sl.Task):
    # INPUT TARGETS
    in_pushp2_completion = None
    in_plugin_bundle = None

    # TASK PARAMETERS
    remote_host = luigi.Parameter()
    remote_user = luigi.Parameter()
    remote_folder = luigi.Parameter()
    eclipse_dir = luigi.Parameter()
    comp_repo_bin_path = luigi.Parameter()
    bundle_name = luigi.Parameter()

    def out_completion_marker(self):
        return sl.TargetInfo(self, self.in_pushp2_completion().path + '.p2site_built' )

    def run(self):
        p2_site_zip_filename = self.in_plugin_bundle().path.split('/')[-1]

        # Unzip the site zip file
        remote_command = 'cd {basedir}; unzip {dir}/{zipfile} -d {dir}'.format(
                basedir = self.remote_folder,
                dir = self.replicate_id,
                zipfile = p2_site_zip_filename)

        self.ex_local(['ssh -o PubkeyAuthentication=no',
                 '%s@%s' % (self.remote_user, self.remote_host),
                 '\'' + remote_command + '\''])

        # Build the p2 site previously pushed, on remote host
        #TODO: bundle_zipfile should be replaced with path to unpacked folder (relative to /var/www/armadillo) (no slashes)
        remote_command = 'export ECLIPSE_DIR={eclipse_dir}/;{comp_repo_bin} {repo_folder} --name "{bundle_name}" add {site_folder}'.format(
                eclipse_dir=self.eclipse_dir,
                comp_repo_bin=self.comp_repo_bin_path,
                repo_folder=self.remote_folder,
                bundle_name=self.bundle_name,
                site_folder=self.replicate_id
        )
        self.ex_local(['ssh -o PubkeyAuthentication=no',
                 '%s@%s' % (self.remote_user, self.remote_host),
                 '\'' + remote_command + '\''])

        # Write some dummy content to the completion marker
        self.ex_local(['echo',
                 '"p2 site built"',
                 '>',
                 self.out_completion_marker().path])

# ====================================================================================================

class ExistingDataFiles(luigi.ExternalTask):
    '''External task for getting hand on existing data files'''

    # PARAMETERS
    dataset_name = luigi.Parameter()

    # DEFINE OUTPUTS
    def out_test_neg(self):
        return sl.TargetInfo(self, os.path.join(os.path.abspath('./data'), 'test_neg_' + str(self.dataset_name) + '.data'))
    def out_test_pos(self):
        return sl.TargetInfo(self, os.path.join(os.path.abspath('./data'), 'test_pos_' + str(self.dataset_name) + '.data'))
    def out_training(self):
        return sl.TargetInfo(self, os.path.join(os.path.abspath('./data'), 'training_' + str(self.dataset_name) + '.data'))

# ====================================================================================================

class GenerateFingerprint(sl.Task):
    '''
    Usage of the FingerprintsGenerator Jar file:

    usage: java FingerprintsGenerator
    -fp <String>                           fingerprint name
    -inputfile <file [inputfile.smiles]>   filename for input SMILES file
    -limit <integer>                       Number of lines to read. Handy for
                                           testing.
    -maxatoms <integer>                    Maximum number of non-hydrogen
                                           atoms. [default: 50]
    -minatoms <integer>                    Minimum number of non-hydrogen
                                           atoms. [default: 5]
    -outputfile <file [output.sign]>       filename for generated output file
    -parser <String>                       parser type
    -silent                                Do not output anything during run
    -threads <integer>                     Number of threads. [default:
                                                               number of cores]

    Supported parser modes are:
    1 -- for our ChEMBL datasets (*.data)
    2 -- for our other datasets (*.smi)

    Supported fingerprints are:
    ecfbit
    ecfpcount
    extended
    signbit
    signcount
    '''

    # INPUT TARGETS
    in_dataset = None

    # PARAMETERS
    fingerprint_type = luigi.Parameter()
    java_path = luigi.Parameter()

    # DEFINE OUTPUTS
    def out_fingerprints(self):
        return sl.TargetInfo(self, self.in_dataset().path + '.' + self.fingerprint_type + '.csr')

    def run(self):
        self.ex(['java', '-jar', 'bin/FingerprintsGenerator.jar',
                '-fp', self.fingerprint_type,
                '-inputfile', self.in_dataset().path,
                '-parser', '1',
                '-outputfile', self.out_fingerprints().path])

# ====================================================================================================

class CompactifyFingerprintHashes(sl.Task):
    '''
    Takes a sparse dataset as input and compacts the values before :'s to integers
    counting from 1 and upwards.
    '''

    in_train_fingerprints = None
    in_test_fingerprints = None

    def out_train_fingerprints(self):
        return sl.TargetInfo(self, self.in_train_fingerprints().path + '.compacted')
    def out_test_fingerprints(self):
        return sl.TargetInfo(self, self.in_test_fingerprints().path + '.compacted')

    def run(self):
        inout_filepaths = [(self.in_train_fingerprints().path,
                            self.out_train_fingerprints().path),
                           (self.in_test_fingerprints().path,
                            self.out_test_fingerprints().path)]

        counter = 0
        register = {}

        for infile_path, outfile_path in inout_filepaths:
            with open(infile_path) as infile, open(outfile_path, 'w') as outfile:
                reader = csv.reader(infile, delimiter=' ')
                for row in reader:
                    # Make sure the first columns stays on 1st column
                    newrow = [row[0]]
                    newcols = []
                    for col in row[1:]:
                        if ':' in col:
                            newcol = ''
                            parts = col.split(':')
                            val = int(parts[0])
                            part2 = parts[1]
                            if val in register:
                                newval = register[val]
                            else:
                                counter += 1
                                register[val] = counter
                                newval = counter
                            newcol = str(newval) + ':' + str(part2)
                            newcols.append(newcol)
                        else:
                            newcols.append(col)
                    newcols = sorted(newcols, key=lambda(x): int(x.split(':')[0]))
                    newrow.extend(newcols)
                    outfile.write(' '.join(newrow) + '\n')

# ====================================================================================================

class BCutPreprocess(sl.Task):

    # INPUT TARGETS
    in_signatures = None

    # TASK PARAMETERS
    replicate_id = luigi.Parameter()
    java_path = luigi.Parameter()

    def out_bcut_preprocessed(self):
        return sl.TargetInfo(self, self.in_signatures().path + '.bcut_preproc')
    def out_bcut_preprocess_log(self):
        return sl.TargetInfo(self, self.in_signatures().path + '.bcut_preproc.log')

    def run(self):
        self.ex(['bin/runbcut',
                self.in_signatures().path,
                self.out_bcut_preprocessed().path])

# ====================================================================================================

class BCutSplitTrainTest(sl.Task):

    # TASK PARAMETERS
    train_size = luigi.Parameter()
    test_size = luigi.Parameter()
    replicate_id = luigi.Parameter()

    # INPORTS
    in_bcut_preprocessed = None

    # OUTPORTS
    def out_traindata(self):
        return sl.TargetInfo(self, self.in_bcut_preprocessed().path + '.{tr}_{te}_bcut_train'.format(
                        tr=str(self.train_size),
                        te=str(self.test_size)
                ))
    def out_testdata(self):
        return sl.TargetInfo(self, self.in_bcut_preprocessed().path + '.{tr}_{te}_bcut_test'.format(
                        tr=str(self.train_size),
                        te=str(self.test_size)
                ))

    def run(self):
        self.ex(['/usr/bin/xvfb-run /sw/apps/R/x86_64/3.0.2/bin/Rscript r/pick_bcut.r',
                '--input_file=%s' % self.in_bcut_preprocessed().path,
                '--training_file=%s' % self.out_traindata().path,
                '--test_file=%s' % self.out_testdata().path,
                '--training_size=%s' % self.train_size,
                '--test_size=%s' % self.test_size])

        # Documentation of pick_bcut.r commandline flags:

        '''
        Usage: /proj/b2013262/nobackup/workflow_components/SampleTrainingAndTest/src/sample/trainingAndTest/pick_bcut.r [options]


        Options:
        --input_file=INPUT_FILE
                filename for BCUT data [required]

        --test_file=TEST_FILE
                filename for generated test set

        --training_file=TRAINING_FILE
                filename for generated training set

        -x [1..N, X%, OR REST], --test_size=[1..N, X%, OR REST]
                Size of test set [default 20]

        -y [1..N, X%, OR REST], --training_size=[1..N, X%, OR REST]
                Size of training set [default rest]

        -c NUMBER, --centers=NUMBER
                Number of kmeans centers [default 100]

        -i NUMBER, --iterations=NUMBER
                Number of kmeans iterations [default 10]

        -k [0, 1, 2], --keep_na=[0, 1, 2]
                0=>remove NA; 1=>NA OK in test; 2=>NA OK; [default 0]

        -f, --force
                Overwrite output files if found

        -r, --random_seed
                Set a seed for reproducibly random results

        -s, --silent
                Suppress comments to STDOUT

        -h, --help
                Show this help message and exit
        '''

# ====================================================================================================

class CountLines(sl.SlurmTask):
    ungzip = luigi.BooleanParameter(default=False)

    in_file = None

    def out_linecount(self):
        return sl.TargetInfo(self, self.in_file().path + '.linecnt')

    def run(self):
        if self.ungzip:
            cmd = 'zcat %s | wc -l' % self.in_file().path
        else:
            cmd = 'wc -l %s' % self.in_file().path

        with self.in_file().open() as infile:
            with self.out_linecount().open('w') as outfile:
                stat, out, err = self.ex_local(cmd)
                linecnt = int(out.split(' ')[0])
                outfile.write(str(linecnt))

# ====================================================================================================

class CreateRandomData(sl.SlurmTask):
    size_mb = luigi.IntParameter()
    replicate_id = luigi.Parameter()

    in_basepath = None

    def out_random(self):
        return sl.TargetInfo(self, self.in_basepath().path + '.randombytes')

    def run(self):
        cmd =['dd',
              'if=/dev/urandom',
              'of=%s' % self.out_random().path,
              'bs=1048576',
              'count=%d' % self.size_mb]
        self.ex(cmd)

# ====================================================================================================

class ShuffleLines(sl.SlurmTask):
    in_file = None
    in_randomdata = None

    def out_shuffled(self):
        return sl.TargetInfo(self, self.in_file().path + '.shuf')

    def run(self):
        #with self.in_file().open() as infile:
        #    with self.out_shuffled().open('w') as outfile:
        self.ex(['shuf',
                       '--random-source=%s' % self.in_randomdata().path,
                       self.in_file().path,
                       '>',
                       self.out_shuffled().path])

# ====================================================================================================

class CreateFolds(sl.SlurmTask):

    # TASK PARAMETERS
    folds_count = luigi.IntParameter()
    fold_index = luigi.IntParameter()

    # TARGETS
    in_dataset = None
    in_linecount = None

    def out_testdata(self):
        return sl.TargetInfo(self, self.in_dataset().path + '.fld{0:02}_tst'.format(self.fold_index))

    def out_traindata(self):
        return sl.TargetInfo(self, self.in_dataset().path + '.fld{0:02}_trn'.format(self.fold_index))

    def run(self):
        with self.in_linecount().open() as linecntfile:
            linecnt = int(linecntfile.read())

        linesperfold = int(math.floor(linecnt / self.folds_count))
        tst_start = self.fold_index * linesperfold
        tst_end = (self.fold_index + 1) * linesperfold

        # CREATE TEST FOLD
        self.ex(['awk',
                 '"NR >= %d && NR <= %d { print }"' % (tst_start, tst_end),
                 self.in_dataset().path,
                 '>',
                 self.out_testdata().path])

        # CREATE TRAIN FOLD
        self.ex(['awk',
                 '"NR < %d || NR > %d { print }"' % (tst_start, tst_end),
                 self.in_dataset().path,
                 '>',
                 self.out_traindata().path])

# ================================================================================

class SelectPercentIndexValue(sl.Task):

    # TASK PARAMETERS
    percent_index = luigi.IntParameter()

    # TARGETS
    in_prediction = None

    def out_indexvalue(self):
        return sl.TargetInfo(self, self.in_prediction().path + '.idx{i:d}'.format(i=self.percent_index))

    def run(self):
        with self.in_prediction().open() as infile:
            lines = [float(l) for l in infile.readlines()]
            lines.sort()
            linescnt = len(lines)
            index = int(linescnt * (self.percent_index / 100.0))
            indexval = lines[index]
            with self.out_indexvalue().open('w') as outfile:
                outfile.write('%f\n' % indexval)

# ================================================================================

class MergeOrigAndPredValues(sl.Task):
    # TARGETS
    in_original_dataset = lambda: sl.TargetInfo(None, None)
    in_predicted_dataset = lambda: sl.TargetInfo(None, None)

    def out_merged(self):
        return sl.TargetInfo(self, self.in_original_dataset().path + '.merged')

    def run(self):
        with self.in_original_dataset().open() as origfile:
            with self.in_predicted_dataset().open() as predfile:
                with self.out_merged().open('w') as outfile:
                    for orig, pred in zip(origfile, predfile):
                        outfile.write(orig.split(' ')[0] + ', ' + pred + '\n')

# ================================================================================

class PlotCSV(sl.Task):
    # TARGETS
    in_csv = lambda: sl.TargetInfo(None, None)

    xmin = luigi.Parameter()
    xmax = luigi.Parameter()
    ymin = luigi.Parameter()
    ymax = luigi.Parameter()

    def out_pdf(self):
        return sl.TargetInfo(self, self.in_csv().path + '.pdf')

    def run(self):
        # Create a temporary R script
        rscript = u'''
        ## Parse arguments
        library('argparse')
        p <- ArgumentParser()
        p$add_argument("-i", "--input", type="character",
                       help="Input file in CSV format")
        p$add_argument("-o", "--output", type="character",
                       help="Output file (will be in .pdf format)")
        args <- p$parse_args()

        ## Plot
        if ( args$input != "" && args$output != "" ) {{
          data = read.csv(file=args$input, header = FALSE)
          pdf(file = args$output, width=5, height=5)
          plot(NULL, xlim=c({xmin},{xmax}), ylim=c({ymin},{ymax}), xlab="", ylab="", cex.axis=1.5)
          points(data, cex = .2, pch=16)
          dev.off()
        }} else {{
            print('Either input or output is missing! Use -h to see options!')
            quit(1)
        }}
        '''.format(
                xmin=self.xmin,
                xmax=self.xmax,
                ymin=self.ymin,
                ymax=self.ymax)

        tempscriptpath='.temp-r-script-%s.r' % uuid.uuid4()
        tsf = open(tempscriptpath,'w')
        tsf.write(rscript)
        tsf.close()
        # Execute the R script
        self.ex_local(['xvfb-run',
                       'Rscript',
                       tempscriptpath,
                       '-i',
                       self.in_csv().path,
                       '-o',
                       self.out_pdf().path])
        # Remove the temporary R script
        self.ex_local(['rm',
                       tempscriptpath])

# ================================================================================

class MergedDataReport(sl.Task):
    run_id = luigi.Parameter()

    in_reports = None

    def out_merged_report(self):
        return sl.TargetInfo(self, 'data/' + self.run_id + '_merged_report.csv')

    def run(self):
        merged_rows = []
        for i, inreportfile_targetinfo in enumerate(self.in_reports):
            infile = inreportfile_targetinfo().open()
            for j, line in enumerate(infile):
                if i == 0 and j == 0:
                    merged_rows.append(line) # Append header
                if j > 0:
                    merged_rows.append(line)
        with self.out_merged_report().open('w') as outfile:
            outfile.write(''.join(merged_rows))
