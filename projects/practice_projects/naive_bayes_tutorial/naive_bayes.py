import operator
from functools import reduce


class Person(object):

    def __init__(self, p_person, p_freedom, p_immigration, p_environment):
        self.p_person, self.p_freedom, self.p_immigration, self.p_environment =\
            p_person, p_freedom, p_immigration, p_environment

    def p_sum_words_given_person(self, words):
        """
        return for example P(F, I|J)
        @words: list of words, e.g., ['freedom', 'immigration']
        """
        p_words = [self.__getattribute__('p_' + attr) for attr in words]
        return reduce(operator.mul, [*p_words, self.p_person])

    def p_person_given_words(self, words, p_sum_words):
        """
        return for example P(J|F, I)
        @words: list of words, e.g., ['freedom', 'immigration']
        """
        p_sum_words_given_person = self.p_sum_words_given_person(words)
        return p_sum_words_given_person / p_sum_words

    @staticmethod
    def p_sum_words(persons, words):
        return sum([p.p_sum_words_given_person(words) for p in persons])


def main():
    Jill = Person(0.5, 0.1, 0.1, 0.8)
    Gary = Person(0.5, 0.7, 0.2, 0.1)
    words = ['freedom', 'immigration']
    p_fi = Person.p_sum_words([Jill, Gary], words)
    p_j_fi = Jill.p_person_given_words(['freedom', 'immigration'], p_fi)
    p_g_fi = Gary.p_person_given_words(['freedom', 'immigration'], p_fi)

    print(p_j_fi)
    print(p_g_fi)
    print(p_j_fi + p_g_fi)


if __name__ == '__main__':
    main()
