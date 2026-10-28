"""Part 2: the fixed cleaning rules, taken from the BRFSS 2015 codebook.

The rules depend only on the codebook, so they learn nothing from the data and
clean train and test alike. The reasons behind them are in
`docs/eda_findings.md`, and `notebooks/02_cleaning.ipynb` checks them on the data.
Steps that learn from the data (one-hot encoding, imputation, scaling) are
part-3 transforms.
"""

import numpy as np

from p1.data import Dataset, load_raw, save_dataset
from p1.paths import CLEAN_DATA

NAN = np.nan

# Dropped: about the survey, not the respondent (record ids, interview dates,
# sampling strata and weights).
SURVEY_DESIGN = [
    "SEQNO",
    "_PSU",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_LLCPWT",
    "_CLLCPWT",
    "_DUALCOR",
]
# Dropped: exact copies of CTELNUM1 and _CRACE1.
EXACT_COPIES = ["CELLFON2", "_CPRACE"]
# Dropped: answers that pack a unit into the number (ALCDAY5 is 1xx days per
# week or 2xx days per month). The CDC converted each of them into the kept
# variables named in the comment.
MIXED_UNITS = [
    "WEIGHT2",  # WTKG3
    "HEIGHT3",  # HTIN4, HTM4
    "ALCDAY5",  # DROCDY3_, DRNKANY5
    "FRUITJU1",  # FTJUDA1_
    "FRUIT1",  # FRUTDA1_
    "FVBEANS",  # BEANDAY_
    "FVGREEN",  # GRENDAY_
    "FVORANG",  # ORNGDAY_
    "VEGETAB1",  # VEGEDA1_
    "EXEROFT1",  # PAFREQ1_
    "EXEROFT2",  # PAFREQ2_
    "EXERHMM1",  # PADUR1_
    "EXERHMM2",  # PADUR2_
    "STRENGTH",  # STRFREQ_
]
# Every other feature is kept, however sparse: HAREHAB1 (99.8% NaN) is answered
# only by positives.
DROPPED = SURVEY_DESIGN + EXACT_COPIES + MIXED_UNITS

# Numeric answers: special code -> value. "None" becomes 0, "don't know" and
# "refused" become NaN; all other values are already quantities.
CODES = {
    "PHYSHLTH": {88: 0, 77: NAN, 99: NAN},  # days in the past 30
    "MENTHLTH": {88: 0, 77: NAN, 99: NAN},
    "POORHLTH": {88: 0, 77: NAN, 99: NAN},
    "CHILDREN": {88: 0, 99: NAN},
    "AVEDRNK2": {77: NAN, 99: NAN},  # drinks per drinking day
    "DRNK3GE5": {88: 0, 77: NAN, 99: NAN},  # binge occasions in the past 30 days
    "MAXDRNKS": {77: NAN, 99: NAN},
    "JOINPAIN": {77: NAN, 99: NAN},  # 0-10 scale
    "DIABAGE2": {98: NAN, 99: NAN},  # age when diagnosed, 97 = 97 or older
    "SCNTWRK1": {98: 0, 97: NAN, 99: NAN},  # hours of work per week, 77 is valid
    "SCNTLWK1": {98: 0, 97: NAN, 99: NAN},
    "DOCTDIAB": {88: 0, 77: NAN, 99: NAN},  # times in the past 12 months
    "CHKHEMO3": {88: 0, 77: NAN, 98: NAN, 99: NAN},  # 98 = never heard of the test
    "FEETCHK": {88: 0, 77: NAN, 99: NAN},
    "ASTHMAGE": {97: 10, 98: NAN, 99: NAN},  # age when diagnosed, 97 = 10 or younger
    "ASERVIST": {88: 0, 98: NAN, 99: NAN},  # visits in the past 12 months
    "ASDRVIST": {88: 0, 98: NAN, 99: NAN},
    "ASRCHKUP": {88: 0, 98: NAN, 99: NAN},
    "ASACTLIM": {888: 0, 777: NAN, 999: NAN},  # days in the past 12 months
    # Computed by the CDC. The release divided most of them by their implied
    # decimals, codes included (MAXVO2_ 99900 -> 999).
    "DROCDY3_": {900: NAN},
    "_DRNKWEK": {99900: NAN},
    "MAXVO2_": {999: NAN},
    "FC60_": {999: NAN},
    "PAFREQ1_": {99: NAN},
    "PAFREQ2_": {99: NAN},
    "STRFREQ_": {99: NAN},
}

# Physical activity per week: times and minutes, for the first and the second
# activity. The CDC leaves them blank when there is no such activity.
FIRST_ACTIVITY_PER_WEEK = ["PAFREQ1_", "_MINAC11", "PAMIN11_", "PAVIG11_"]
SECOND_ACTIVITY_PER_WEEK = ["PAFREQ2_", "_MINAC21", "PAMIN21_", "PAVIG21_"]
TOTAL_ACTIVITY_PER_WEEK = ["PA1MIN_", "PA1VIGM_"]

# The clean features that are quantities. All other kept features are
# categorical codes, to be one-hot encoded.
NUMERIC = [
    # answers, after the code rules
    "PHYSHLTH",
    "MENTHLTH",
    "POORHLTH",
    "CHILDREN",
    "AVEDRNK2",
    "DRNK3GE5",
    "MAXDRNKS",
    "JOINPAIN",
    "DIABAGE2",
    "SCNTWRK1",
    "SCNTLWK1",
    "DOCTDIAB",
    "CHKHEMO3",
    "FEETCHK",
    "ASTHMAGE",
    "ASERVIST",
    "ASDRVIST",
    "ASRCHKUP",
    "ASACTLIM",
    # answers converted to one unit by `clean`
    "BLDSUGAR",  # times per year
    "FEETCHK2",  # times per year
    "LONGWTCH",  # years
    "FLSHTMY2",  # months before the interview
    "HIVTSTD3",  # months before the interview
    # computed by the CDC
    "_AGE80",
    "HTIN4",
    "HTM4",
    "WTKG3",
    "_BMI5",
    "DROCDY3_",
    "_DRNKWEK",
    "FTJUDA1_",
    "FRUTDA1_",
    "BEANDAY_",
    "GRENDAY_",
    "ORNGDAY_",
    "VEGEDA1_",
    "_FRUTSUM",
    "_VEGESUM",
    "METVL11_",
    "METVL21_",
    "MAXVO2_",
    "FC60_",
    "PADUR1_",
    "PADUR2_",
    "PAFREQ1_",
    "PAFREQ2_",
    "_MINAC11",
    "_MINAC21",
    "STRFREQ_",
    "PAMIN11_",
    "PAMIN21_",
    "PA1MIN_",
    "PAVIG11_",
    "PAVIG21_",
    "PA1VIGM_",
]
# The numeric features that count times, drinks, visits, servings or minutes.
# A few respondents report huge values (skewness 3 to 67 on the training data),
# so part 3 can log-transform them.
HEAVY_TAILED = [
    "CHILDREN",
    "AVEDRNK2",
    "DRNK3GE5",
    "MAXDRNKS",
    "DOCTDIAB",
    "CHKHEMO3",
    "FEETCHK",
    "ASERVIST",
    "ASDRVIST",
    "ASRCHKUP",
    "ASACTLIM",
    "BLDSUGAR",
    "FEETCHK2",
    "_DRNKWEK",
    "FTJUDA1_",
    "FRUTDA1_",
    "BEANDAY_",
    "GRENDAY_",
    "ORNGDAY_",
    "VEGEDA1_",
    "_FRUTSUM",
    "_VEGESUM",
    "PAFREQ1_",
    "PAFREQ2_",
    "_MINAC11",
    "_MINAC21",
    "STRFREQ_",
    "PAMIN11_",
    "PAMIN21_",
    "PA1MIN_",
    "PAVIG11_",
    "PAVIG21_",
    "PA1VIGM_",
]


def clean(x: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Apply the fixed cleaning rules to a feature matrix.

    The rules come from the BRFSS codebook and learn nothing from the data, so
    train and test are cleaned the same way. Steps that learn from the data are
    transforms in `p1/transforms.py`.

    1. Drop the `DROPPED` columns.
    2. Numeric answers: replace the special codes of `CODES`, and convert the
       answers that mix units, or are dates, to one unit.
    3. Fill the numeric questions the survey skipped because their answer is
       implied, such as 0 drinks for people who did not drink.
    The categorical features keep their codes: "don't know" carries signal, and
    one-hot encoding gives every code its own column.

    Args:
        x: Raw features, shape (N, D). Not modified.
        feature_names: Name of each column of `x`.

    Returns:
        The clean features and the names of their columns.
    """
    raw = {name: x[:, j] for j, name in enumerate(feature_names)}
    names = [name for name in feature_names if name not in DROPPED]
    x_clean = x[:, [feature_names.index(name) for name in names]]  # a copy
    col = {name: x_clean[:, j] for j, name in enumerate(names)}  # views to edit

    for name, codes in CODES.items():
        for code, value in codes.items():
            col[name][raw[name] == code] = value
    # The release left these two in hundredths, unlike the other CDC variables.
    col["DROCDY3_"] /= 100  # drink-occasions per day
    col["_DRNKWEK"] /= 100  # drinks per week

    col["BLDSUGAR"][:] = _times_per_year(raw["BLDSUGAR"])
    col["FEETCHK2"][:] = _times_per_year(raw["FEETCHK2"])  # 555 (no feet) -> NaN
    col["LONGWTCH"][:] = _years(raw["LONGWTCH"], age=raw["_AGE80"])
    for name in ["FLSHTMY2", "HIVTSTD3"]:
        col[name][:] = _months_before(raw[name], raw["IYEAR"], raw["IMONTH"])

    # The codebook skips these questions in these cases; the data confirms that
    # they are then always blank.
    no_bad_day = (raw["PHYSHLTH"] == 88) & (raw["MENTHLTH"] == 88)
    col["POORHLTH"][no_bad_day] = 0
    for name in ["AVEDRNK2", "DRNK3GE5", "MAXDRNKS"]:
        col[name][raw["ALCDAY5"] == 888] = 0  # no drink in the past 30 days
    activity = FIRST_ACTIVITY_PER_WEEK + SECOND_ACTIVITY_PER_WEEK
    for name in activity + TOTAL_ACTIVITY_PER_WEEK:
        col[name][raw["EXERANY2"] == 2] = 0  # no exercise in the past 30 days
    for name in SECOND_ACTIVITY_PER_WEEK:
        col[name][raw["EXRACT21"] == 88] = 0  # no second activity
    return x_clean, names


def _split_unit(code: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read codes 1xx to 4xx as xx days, weeks, months or years.

    Returns:
        The count xx, and how many of its unit fit in a year (365, 52, 12 or 1;
        NaN for other codes).
    """
    unit = code // 100
    per_year = np.select([unit == u for u in (1, 2, 3, 4)], [365, 52, 12, 1], NAN)
    return code % 100, per_year


def _times_per_year(code: np.ndarray) -> np.ndarray:
    """Convert a frequency coded 1xx to 4xx (xx times per day to per year).

    888 (never) becomes 0; "don't know", "refused" and other codes become NaN.
    """
    count, per_year = _split_unit(code)
    return np.where(code == 888, 0, count * per_year)


def _years(code: np.ndarray, age: np.ndarray) -> np.ndarray:
    """Convert a duration coded 1xx to 4xx (xx days to years) to years.

    555 (all my life) becomes `age`; "don't know" and "refused" become NaN.
    """
    count, per_year = _split_unit(code)
    return np.where(code == 555, age, count / per_year)


def _months_before(date: np.ndarray, year: np.ndarray, month: np.ndarray) -> np.ndarray:
    """Count the months from a date coded MMYYYY to the interview.

    An unknown month (77YYYY) is taken as the middle of the year, and a date
    later in the interview year counts as 0 months. "Don't know" (777777),
    "refused" (999999) and years before 1985 (coded "don't know" by the survey)
    or after the interview become NaN.

    Args:
        date: Dates coded MMYYYY, with NaN for missing answers.
        year: Year of the interview.
        month: Month of the interview.
    """
    date_month, date_year = date // 10000, date % 10000
    date_month = np.where(date_month == 77, 6.5, date_month)
    months = 12 * (year - date_year) + month - date_month
    valid = (date_month <= 12) & (date_year >= 1985) & (date_year <= year)
    return np.where(valid, np.maximum(months, 0), NAN)


def build_clean_dataset() -> None:
    """Clean the raw train and test data and save it to `data/processed/clean.npz`."""
    raw = load_raw()
    x_train, names = clean(raw.x_train, raw.feature_names)
    x_test, _ = clean(raw.x_test, raw.feature_names)
    clean_ds = Dataset(
        x_train=x_train,
        y_train=raw.y_train,
        x_test=x_test,
        train_ids=raw.train_ids,
        test_ids=raw.test_ids,
        feature_names=names,
    )
    save_dataset(clean_ds, CLEAN_DATA)
