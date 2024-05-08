import pyreadstat
import pandas as pd
import numpy as np
from random import shuffle
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.ensemble import RandomForestRegressor as RFR
from scipy.stats import pearsonr, ttest_ind

# Path to your .sav file
sav_file_path = 'Initial_31_centres.sav'


# Read the .sav file

def rep(self):
    return f"The keys of meta are {', '.join(meta.__dict__.keys())}"


pyreadstat._readstat_parser.metadata_container.__repr__ = rep


def create_one_hot(one_hot, df):
    keys = sorted(list(df[one_hot].drop_duplicates().values))

    base = pd.Series([0 for i in keys], index=keys)

    ohs = []
    column = df[one_hot]
    for r in column.index:
        oh = base.copy()
        oh[column[r]] = 1
        ohs.append(oh)
    ohs_df = pd.concat(ohs, axis=1).T
    ohs_df.columns = [f"{one_hot}_{i}" for i in ohs_df.columns]
    return ohs_df


def improve_date(date_col, df):
    new_col = ((df[date_col] - df['MRI_diagnosis']) // 86400)  # seconds in a day
    new_col.name = date_col
    return new_col


# use for one_hots
def process_one_hots(feature, good, correlants, comp, missing):
    values = comp[feature].drop_duplicates().values
    for v in values:
        yes = comp.loc[comp[feature] == v].days_alive
        no = comp.loc[comp[feature] != v].days_alive
        description = pd.concat([yes.describe(), no.describe()], axis=1)
        description.columns = ['yes', 'no']
        result = ttest_ind(yes, no)
        correlants[(feature, v)] = [result.pvalue, missing, 'one_hot']
        if result.pvalue < 0.05:
            good.append((feature, v))
        return good, correlants


def process_bool(feature, good, correlants, comp, missing):
    yes = comp.loc[comp[feature] == 1.0].days_alive
    no = comp.loc[comp[feature] == 0.0].days_alive
    result = ttest_ind(yes, no)
    correlants[feature] = [result.pvalue, missing, 'bool']
    if result.pvalue < 0.05:
        good.append(feature)
    return good, correlants


# use for chunked or continuous
def process_other(feature, good, correlants, comp, missing):
    if feature in dates:
        comp[feature] = improve_date(feature, comp)

    result = pearsonr(comp[feature].values, comp['days_alive'].values)
    correlants[feature] = [result.pvalue, missing, 'ordinal']
    if result.pvalue < 0.05:
        good.append(feature)
    return good, correlants


def train_test_validate_split(X, y, train=0.8, test=0.1, validate=0.1):
    index = list(X.index)
    shuffle(index)
    num_test = int(test * len(index))
    num_train = int(train * len(index))
    num_validate = len(index) - num_test - num_train
    test_index = index[: num_test]
    train_index = index[num_test: num_train + num_test]
    validate_index = index[num_train + num_test:]
    X_test, y_test = X.loc[test_index], y[test_index]
    X_train, y_train = X.loc[train_index], y[train_index]
    X_validate, y_validate = X.loc[validate_index], y[validate_index]
    return X_train, y_train, X_test, y_test, X_validate, y_validate


# There are a number of labels that are given in the meta data
# that are not present in the df. Talk to Stephen about them. These are:
other_labels = sorted(['Age at diagnosis', 'Performance status at diagnosis',
                       'Contrast enhancement on MRI', 'MGMT promoter methylation',
                       'G3 marker',
                       'G4 marker',
                       'Histological grade',
                       'Molecular features of glioblastoma',
                       'Extent of surgery',
                       'Upfront radiotherapy',
                       'Type of radiotherapy',
                       'Planning target volume, cc',
                       'Concurrent temozolomide',
                       'Adjuvant temozolomide',
                       'Concurrent + >5 cycles adjuvant',
                       'concurrent chemo + conventional fractionation',
                       'Progression free survival, days',
                       'Overall survival, days',
                       'Time from MRI to surgery',
                       'Time from surgery to radiotherapy',
                       'Time from MRI to radiotherapy',
                       'TERT only molecular GBM',
                       'OS from initial MRI, days',
                       'IDHwt = 1 (FILTER)'])

features = ['Age',
            'Gender',
            'MRI_diagnosis',
            'Presenting_seizures',
            'Presenting_motor',
            'Presenting_speech',
            'Presenting_cognition',
            'Presenting_behaviour',
            'Presenting_vision',
            'Presentin_headache',
            'Presenting_sensory',
            'Presenting_GCS',
            'Presenting_incidental',
            'Presenting_other',
            'Steroids',
            'AED',
            'PS',
            'Tumour_site',
            'Lobe_Frontal',
            'Lobe_parietal',
            'Lobe_temporal',
            'Lobe_occipital',
            'Lobe_cerebellum',
            'Lobe_brainstem',
            'Multifocal',
            'Contrast',
            'Histological_diagnosis',
            'IDHwt',
            'IDH_IHC',
            'IDH_NGS',
            'IDH_methylation',
            'MGMT',
            'MGMT_percentage',
            'MGMT_summary',
            'Histo_none',
            'Histo_atypia',
            'Histo_mitoses',
            'Histo_necrosis',
            'Histo_mvp',
            'Grade',
            'Mol_none',
            'Mol_EGFR',
            'Mol_7_10',
            'Mol_TERT',
            'Mol_CDKN2A',
            'Mol_CDKN2B',
            'Mol_not_tested',
            'Molecular',
            'Mol_GBM',
            'Surgery',
            'Surgery_extent',
            'Surgery_percentage',
            'Surgery_imaging',
            'Surgery_imaging_type',
            'Radiotherapy',
            'RT_MRI',
            'RT_imaging',
            'REP',
            'RT_type',
            'PTV',
            'Concurrent_TMZ',
            'Concurrent_completed',
            'Adjuvant_TMZ',
            'Adjuvant_cycles',
            'Adjuvant_stop',
            'Full_chemo',
            'Stupp',
            'Perry',
            'TTF',
            'Surveillance1_date',
            'Surveillance1',
            'Surveillance1_outcome',
            'Surveillance2_date',
            'Surveillance2',
            'Surveillance2_outcome',
            'Indeterminate',
            'Indeterminate_date',
            'Indeterminate_structural',
            'Indeterminate_spectroscopy',
            'Indeterminate_perfusion',
            'Indeterminate_advanced_other',
            'Indeterminate_biopsy',
            'Indeterminate_other',
            'Indeterminate_repeat_date',
            'Indeterminate_repeat_findings',
            'Indeterminate_final_findings',
            'Treatment_effects',
            'Treatment_effects_12_weeks',
            'Progression1',
            'Progression1_date',
            'Treatment2',
            'Treatment2_type',
            'Progression2',
            'Progression2_date',
            'PFS1',
            'TTSurgery',
            'TTRt',
            'Delay',
            'TERT_Mol',
            'filter_$']

dates = ['MRI_diagnosis',
         'Histological_diagnosis',
         'RT_imaging',
         'Surgery',
         'Surgery_imaging',
         'Surveillance1_date',
         'Surveillance2_date',
         'Indeterminate_date',
         'Indeterminate_repeat_date',
         'Progression1_date',
         'Progression2_date',
         'Follow_Up',
         'Death_date']
chunked = ['MGMT_summary', 'Molecular',
           'Surgery_extent',
           'RT_type',
           'Adjuvant_stop',
           'Surveillance1',
           'Surveillance1_outcome',
           'Surveillance2',
           'Surveillance2_outcome',
           'Indeterminate_repeat_findings',
           'Indeterminate_final_findings',
           'Treatment_effects', 'Grade'
           ]
one_hots = ['Centre', 'Tumour_site', 'Treatment2_type', 'TERT_Mol']
bools = ['Age_65',
         'Age_70',
         'Age_80',
         'Gender',
         'Presenting_seizures',
         'Presenting_motor',
         'Presenting_speech',
         'Presenting_cognition',
         'Presenting_behaviour',
         'Presenting_vision',
         'Presentin_headache',
         'Presenting_sensory',
         'Presenting_GCS',
         'Presenting_incidental',
         'Presenting_other',
         'Radiotherapy',
         'Steroids',
         'AED',
         'Tumour_site',
         'Lobe_Frontal',
         'Lobe_parietal',
         'Lobe_temporal',
         'Lobe_occipital',
         'Lobe_cerebellum',
         'Lobe_brainstem',
         'Multifocal',
         'Contrast',
         'IDHwt',
         'IDH_IHC',
         'IDH_NGS',
         'IDH_methylation',
         'MGMT',
         'Histo_none',
         'Histo_atypia',
         'Histo_mitoses',
         'Histo_necrosis',
         'Histo_mvp',
         'Mol_none',
         'Mol_EGFR',
         'Mol_7_10',
         'Mol_TERT',
         'Mol_CDKN2A',
         'Mol_CDKN2B',
         'Mol_not_tested',
         'Mol_GBM',
         'RT_MRI',
         'REP',
         'Concurrent_TMZ',
         'Concurrent_completed',
         'Adjuvant_TMZ',
         'Full_chemo',
         'Stupp',
         'Surgery_imaging_type',
         'Perry',
         'TTF',
         'Indeterminate',
         'Indeterminate_structural',
         'Indeterminate_spectroscopy',
         'Indeterminate_perfusion',
         'Indeterminate_advanced_other',
         'Indeterminate_biopsy',
         'Indeterminate_other',
         'Treatment_effects_12_weeks',
         'Progression1',
         'Treatment2',
         'Progression2',
         'Alive',
         'filter_$']


def get_good_features(df, missing, pvalue):
    df_basic = df[features]
    correlants = {}
    good = []
    for feature in df_basic.columns:
        if feature != 'MRI_diagnosis':
            comp = pd.concat([df[feature], df['MRI_diagnosis'], y], axis=1)
            missing = comp.isna().sum()[feature]
            comp = comp.dropna()
            if feature in bools:
                good, correlants = process_bool(feature, good, correlants, comp, missing)
            elif feature in one_hots:
                good, correlants = process_one_hots(feature, good, correlants, comp, missing)
            else:
                good, correlants = process_other(feature, good, correlants, comp, missing)

    correlant_df = pd.DataFrame(correlants.values())
    correlant_df.index = correlants.keys()
    correlant_df.columns = ['pvalue', 'number_missing', 'category']
    correlant_df.sort_values(by='number_missing', ascending=False)
    cdf = correlant_df.loc[(correlant_df.pvalue < pvalue) & (correlant_df.number_missing < pvalue)]
    good_features = list(cdf.index)
    return cdf, good_features


def improve_na(df):
    for c in df.columns:
        df[c] = df[c].fillna(df[c].median())
    return df


def make_X_y(df, dropna, pvalue, missing):
    df.dropna(subset='MRI_diagnosis', inplace=True)
    y = improve_date('Death_date', df)
    y = y.fillna(y.max())
    y.name = 'days_alive'
    # Must be a better assumption to make here. Ask Stephen what.

    cdf, good_features = get_good_features(df, pvalue, missing)
    df_good = pd.concat([df[good_features + ['MRI_diagnosis']], y], axis=1)
    # drop the samples for features that are shown to be strongly correlated
    # and where the number of missing is less than one hundred
    if dropna:
        df_vgood = df_good.dropna()
    else:
        df_vgood = improve_na(df_good)
    vg_features = []
    for feature in df_vgood.columns:
        if feature in dates:
            vg_features.append(improve_date(feature, df_vgood))
        # add in way to deal with one-hots!!
        else:
            vg_features.append(df_vgood[feature])

    vg = pd.concat(vg_features, axis=1)
    vg.drop('MRI_diagnosis', axis=1, inplace=True)

    X = vg.iloc[:, :-1]
    y = vg.iloc[:, -1]
    return X, y, cdf


# improve to deal with one-hots properly !!!!!!!
def find_cutoffs():
    df, meta = pyreadstat.read_sav(sav_file_path)

    pvalue_range = np.arange(0.01, 0.051, 0.01)
    missing_range = np.arange(100, 210, 10)
    for pvalue in pvalue_range:
        for missing in missing_range:
            for dropna in [True, False]:

                X, y, cdf = make_X_y(df, pvalue=pvalue, missing=missing, dropna=dropna)
                scores = []
                for i in range(20):
                    X_train, y_train, X_test, y_test, X_validate, y_validate = train_test_validate_split(X, y)
                    rfr = RFR()
                    rfr.fit(X_train, y_train)
                    scores.append(rfr.score(X_test, y_test))
                mean_score = np.array(scores).mean()
                mean_scores[(pvalue, missing, dropna)] = mean_score
                print(pvalue, missing, dropna, mean_score)
    return mean_scores