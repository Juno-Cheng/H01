#!/usr/bin/env python3

"""
Do a local practice grading.
The score you recieve here is not an actual score,
but gives you an idea on how prepared you are to submit to the autograder.
"""

import os
import sys

# TODO: Find another way to test the plots, since the GitHub CI fails on this import.
# import matplotlib.axes._subplots
import numpy
import pandas

import cse40.question
import cse40.assignment
import cse40.style
import cse40.utils

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(THIS_DIR, 'synthetic_covid_data.csv')

class T0A(cse40.question.Question):
    def score_question(self, submission, synthetic_data):
        result = submission.select_column(synthetic_data, 'titer')
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, pandas.Series)):
            self.fail("Answer must be a column.")
            return

        self.full_credit()

class T0B(cse40.question.Question):
    def score_question(self, submission, synthetic_data):
        result = submission.filter_rows(synthetic_data, 'titer', 32)
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, pandas.DataFrame)):
            self.fail("Answer must be a DataFrame.")
            return

        self.full_credit()

class T0C(cse40.question.Question):
    def score_question(self, submission, synthetic_data):
        result = submission.add_column(pandas.DataFrame(), 'test', [])
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, pandas.DataFrame)):
            self.fail("Answer must be a DataFrame.")
            return

        self.full_credit()

class T0D(cse40.question.Question):
    def score_question(self, submission, synthetic_data):
        result = submission.drop_column(synthetic_data.copy(), 'titer')
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, pandas.DataFrame)):
            self.fail("Answer must be a DataFrame.")
            return

        self.full_credit()

class T0E(cse40.question.Question):
    def score_question(self, submission, synthetic_data):
        result = submission.concat_frames(pandas.DataFrame(), pandas.DataFrame())
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, pandas.DataFrame)):
            self.fail("Answer must be a DataFrame.")
            return

        self.full_credit()

class T1A(cse40.question.Question):
    def score_question(self, submission, synthetic_data):
        result = submission.count_infected(synthetic_data)
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, (int, numpy.integer))):
            self.fail("Answer must be an integer.")
            return

        self.full_credit()

class T1B(cse40.question.Question):
    def score_question(self, submission, synthetic_data):
        result = submission.count_symptomatic(synthetic_data)
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, (int, numpy.integer))):
            self.fail("Answer must be an integer.")
            return

        self.full_credit()

class T1C(cse40.question.Question):
    def score_question(self, submission, synthetic_data):
        result = submission.mean_days(synthetic_data)
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, (float, numpy.float64, numpy.float32))):
            self.fail("Answer must be a float.")
            return

        self.full_credit()

class T2A(cse40.question.Question):
    def score_question(self, submission, synthetic_data):
        result = submission.fraction_infected(synthetic_data)
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, (float, numpy.float64, numpy.float32))):
            self.fail("Answer must be a float.")
            return

        self.full_credit()

class T2B(cse40.question.Question):
    def score_question(self, submission, synthetic_data):
        result = submission.fraction_symptomatic(synthetic_data)
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, (float, numpy.float64, numpy.float32))):
            self.fail("Answer must be a float.")
            return

        self.full_credit()

class T2C(cse40.question.Question):
    def score_question(self, submission, synthetic_data):
        result = submission.count_special_uninfected(synthetic_data)
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, (int, numpy.integer))):
            self.fail("Answer must be an integer.")
            return

        self.full_credit()

class T2D(cse40.question.Question):
    def score_question(self, submission, synthetic_data):
        result = submission.fraction_isoantigenic(synthetic_data)
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, (float, numpy.float64, numpy.float32))):
            self.fail("Answer must be a float.")
            return

        self.full_credit()

class T3A(cse40.question.Question):
    def score_question(self, submission, synthetic_data):
        result = submission.add_isoantigenic_column(synthetic_data)
        if (self.check_not_implemented(result)):
            return

        if ('isoantigenic' not in result):
            self.fail("Isoantigenic column is missing.")
            return

        self.full_credit()

class T4A(cse40.question.Question):
    def score_question(self, submission, synthetic_data):
        result = submission.prep_scatter(synthetic_data, 'days_before_symptoms', 'titer', 'X', 'Y')
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, pandas.DataFrame)):
            self.fail("Answer must be a DataFrame.")
            return

        self.full_credit()

class T5A(cse40.question.Question):
    def score_question(self, submission, synthetic_data):
        predictions = [1, 1, 0, 0]
        labels = [1, 0, 1, 0]

        result = submission.rmse(predictions, labels)
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, (float, numpy.float64, numpy.float32))):
            self.fail("Answer must be a float.")
            return

        self.full_credit()

def grade(path):
    submission = cse40.utils.prepare_submission(path)
    additional_data = {
        'synthetic_data': pandas.read_csv(DATA_PATH, index_col = 'id')
    }

    questions = [
        T0A("Task 0.A (select_column)", 1),
        T0B("Task 0.B (filter_rows)", 1),
        T0C("Task 0.C (add_column)", 1),
        T0D("Task 0.D (drop_column)", 1),
        T0E("Task 0.E (concat_frames)", 1),
        T1A("Task 1.A (count_infected)", 1),
        T1B("Task 1.B (count_symptomatic)", 1),
        T1C("Task 1.C (mean_days)", 1),
        T2A("Task 2.A (fraction_infected)", 1),
        T2B("Task 2.B (fraction_symptomatic)", 1),
        T2C("Task 2.C (count_special_uninfected)", 1),
        T2D("Task 2.D (fraction_isoantigenic)", 1),
        T3A("Task 3.A (add_isoantigenic_column)", 1),
        T4A("Task 4.A (prep_scatter)", 1),
        T5A("Task 5.A (rmse)", 1),
        cse40.style.Style(path, max_points = 5),
    ]

    assignment = cse40.assignment.Assignment('Practice Grading for Hands-On 1', questions)
    assignment.grade(submission, additional_data = additional_data)

    return assignment

def main(path):
    assignment = grade(path)
    print(assignment.report())

def _load_args(args):
    exe = args.pop(0)
    if (len(args) != 1 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args})):
        print("USAGE: python3 %s <submission path (.py or .ipynb)>" % (exe), file = sys.stderr)
        sys.exit(1)

    path = os.path.abspath(args.pop(0))

    return path

if (__name__ == '__main__'):
    main(_load_args(list(sys.argv)))
