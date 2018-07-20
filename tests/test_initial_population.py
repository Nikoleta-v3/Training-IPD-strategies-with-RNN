import sequence_sensei as ss

def test_initial_population():
    size_of_population = 20
    initial_population = ss.get_initial_population(size_of_population)

    assert len(initial_population) == 40
    assert initial_population[-1] == [0 for _ in range(200)]
    assert initial_population[-2] == [1 for _ in range(200)]
