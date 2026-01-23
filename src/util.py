from .data_preprocess import *
def load_data(file_name):
    if file_name == 'Taiwan':
        data_fea, data_labels, X_train, Y_train, X_test, Y_test = Taiwan()
        return data_fea, data_labels, X_train, Y_train, X_test, Y_test
    elif file_name == 'GMSC':
        data_fea, data_labels, X_train, Y_train, X_test, Y_test = Give_me_some_credit()
        return data_fea, data_labels, X_train, Y_train, X_test, Y_test
    elif file_name == 'LD':
        data_fea, data_labels, X_train, Y_train, X_test, Y_test = Loan_Data()
        return data_fea, data_labels, X_train, Y_train, X_test, Y_test
    elif file_name == 'German':
        data_fea, data_labels, X_train, Y_train, X_test, Y_test = German()
        return data_fea, data_labels, X_train, Y_train, X_test, Y_test
    elif file_name == 'HMEQ':
        data_fea, data_labels, X_train, Y_train, X_test, Y_test = HMEQ()
        return data_fea, data_labels, X_train, Y_train, X_test, Y_test
    elif file_name == 'HC':
        data_fea, data_labels,X_train, Y_train, X_test, Y_test = HomeCredit()
        return data_fea, data_labels, X_train, Y_train, X_test, Y_test
    elif file_name == 'LC':
        data_fea, data_labels, X_train, Y_train, X_test, Y_test = Lendingclub()
        return data_fea, data_labels, X_train, Y_train, X_test, Y_test
    elif file_name == 'PAKDD':
        data_fea, data_labels, X_train, Y_train, X_test, Y_test = PAKDD()
        return data_fea, data_labels, X_train, Y_train, X_test, Y_test
    elif file_name == 'Ant':
        data_fea, data_labels, X_train, Y_train, X_test, Y_test = Ant_data()
        return data_fea, data_labels, X_train, Y_train, X_test, Y_test
    else:
        raise Exception('this dataset is not supported yet')


