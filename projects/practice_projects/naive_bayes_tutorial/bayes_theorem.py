def possibility_of_getting_positive_result(p_d, sensitivity,
                                           p_not_d, specifictivity):
    return (p_d * sensitivity) + (p_not_d * (1 - specifictivity))


def possibility_of_Diabetes_given_pos_result(p_pos_diabetes, p_diabetes,
                                             p_pos):
    return p_pos_diabetes * p_diabetes / p_pos


def possibility_of_no_diabetes_given_pos_result(p_pos_no_diabetes, p_no_diabetes,
                                                p_pos):
    return p_pos_no_diabetes * p_no_diabetes / p_pos


def main():
    # P(D)
    p_diabetes = 0.01

    # P(~D)
    p_no_diabetes = 0.99

    # P(pos|D)
    p_pos_diabetes = 0.9

    # P(neg|~D)
    p_neg_no_diabetes = 0.9

    # P(pos)
    p_pos = possibility_of_getting_positive_result(p_diabetes, p_pos_diabetes,
                                                   p_no_diabetes, p_neg_no_diabetes)

    # P(D|pos)
    p_diabetes_pos = possibility_of_Diabetes_given_pos_result(p_pos_diabetes,
                                                              p_diabetes, p_pos)

    # P(~D|pos)
    p_pos_no_diabetes = 1 - p_neg_no_diabetes
    p_no_diabetes_pos = possibility_of_no_diabetes_given_pos_result(p_pos_no_diabetes,
                                                                    p_no_diabetes,
                                                                    p_pos)

    print(p_diabetes_pos)
    print(p_no_diabetes_pos)
    print(p_diabetes_pos + p_no_diabetes_pos)


if __name__ == '__main__':
    main()
