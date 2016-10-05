from ConfigParser import ConfigParser
import dataset
import glob
import sciluigi as sl
import csv
import re

def main():
    audit_table = parse_audit_files()

    dataframe = []
    glob_pattern = 'data/*ungz.s12*.rmsd'
    rmsd_paths = glob.glob(glob_pattern)
    for rp in rmsd_paths:
        datarow = {}
        ms = re.match('data/(solubility|acd_logd).smi.h1_3.sign.(r[0-9]).([0-9]+)_([0-9]+|rest)_rand_trn.csr.ungz.s12_c([0-9\.]+).(lin|svm)mdl.pred.rmsd', rp)
        m = ms.groups()
        datarow['dataset'] = m[0]
        datarow['replicate'] = m[1]
        datarow['test_size'] = m[2]
        datarow['training_size'] = m[3]
        datarow['cost'] = m[4]
        if m[5] == 'lin':
            datarow['learning_method'] = 'liblinear'
        elif m[5] == 'svm':
            datarow['learning_method'] = 'svmrbf'
        with open(rp) as rf:
            rd = sl.util.recordfile_to_dict(rf)
            datarow['rmsd'] = rd['rmsd']
            datarow['cost'] = rd['cost']
            dataframe.append(datarow)
    with open('rdataframe.csv', 'w') as fh:
        csvwrt = csv.writer(fh)
        csvwrt.writerow(['dataset', 'learning_method', 'training_size', 'replicate', 'rmsd', 'model_creation_time', 'cost'])
        for row in dataframe:

            auditrow = audit_table.find_one(training_size=row['training_size'], test_size=row['test_size'])
            if auditrow is not None:
                model_creation_time = int(dict(auditrow)['slurm_exectime_sec'])
            else:
                model_creation_time = 'N/A'

            outrow = [row['dataset'], row['learning_method'], row['training_size'], row['replicate'], row['rmsd'], model_creation_time, row['cost']]
            print outrow
            csvwrt.writerow(outrow)


def parse_audit_files():
    audit_paths = glob.glob('audit/workflow_mmlinear_started_20151026_17*')
    audit_data = []
    for path in audit_paths:
        with open(path) as fh:
            cp = ConfigParser()
            cp.readfp(fh)
            for sec in cp.sections():
                dat = dict(cp.items(sec))
                if 'train_lin' in dat['instance_name']:
                    ms = re.match('train_lin_trn([0-9]+|rest)_tst([0-9]+)_c([0-9\.]+)', dat['instance_name'])
                    if ms is None:
                        raise Exception('No match in name: ' + dat['instance_name'])
                    m = ms.groups()
                    dat['training_size'] = m[0]
                    dat['test_size'] = m[1]
                    dat['cost'] = m[2]
                    audit_data.append(dat)
    db = dataset.connect('sqlite:///:memory:')
    tbl = db['audit']
    for d in audit_data:
        tbl.insert(d)
    return tbl


if __name__ == '__main__':
    main()
