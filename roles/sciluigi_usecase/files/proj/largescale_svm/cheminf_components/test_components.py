from components import *
import os
import time

class TestConcatenate2Files():
    file1_path = '/tmp/luigi_concat2files_file1'
    file2_path = '/tmp/luigi_concat2files_file2'

    file1_content = 'A'*80 + '\n'
    file2_content = 'B'*80 + '\n'

    def setup(self):
        with open(self.file1_path,'w') as file1:
            file1.write(self.file1_content)
        with open(self.file2_path,'w') as file2:
            file2.write(self.file2_content)

        self.concat2files = Concatenate2Files(
                replicate_id='TESTID',
                accounted_project='b2015002',
                file1_target=luigi.LocalTarget(self.file1_path),
                file2_target=luigi.LocalTarget(self.file2_path),
                skip_file1_header=False,
                skip_file2_header=False
        )

    def teardown(self):
        os.remove(self.file1_path)
        os.remove(self.file2_path)
        os.remove(self.concat2files.output()['concatenated_file'].path)

    def test_run(self):
        # Run the task with a luigi worker
        w = luigi.worker.Worker()
        w.add(self.concat2files)
        w.run()
        w.stop()

        with open(self.concat2files.output()['concatenated_file'].path) as concat_file:
            concatenated_content = concat_file.read()

        assert concatenated_content == self.file1_content + self.file2_content

# ================================================================================

class TestCrossValidate():
    indata_file_path = '/tmp/luigi_crossval_indatafile'

    indata_file_content = 'A'*80 + '\n'

    def setup(self):
        with open(self.indata_file_path,'w') as indata_file:
            indata_file.write(self.indata_file_content)

        self.concat2files = CrossValidate(
        )

    def teardown(self):
        os.remove(self.indata_file_path)
        os.remove() # The output file ...

    def test_run(self):
        # Run the task with a luigi worker
        w = luigi.worker.Worker()
        w.add(self.crossvalidate)
        w.run()
        w.stop()
        
        # assert stuff ...
