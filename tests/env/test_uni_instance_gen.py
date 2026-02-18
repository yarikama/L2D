import numpy as np

from l2d.env.uni_instance_gen import (
    _shuffle_rows,
    generate_uniform_machines_assignment_and_stores_to_file,
    generate_uniform_times_and_machines_assignment,
)


class TestShuffleRows:
    def test_shape_preserved(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = _shuffle_rows(arr)
        assert result.shape == arr.shape

    def test_elements_preserved(self):
        """Each row should contain the same elements, just reordered."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = _shuffle_rows(arr)
        for i in range(arr.shape[0]):
            assert sorted(result[i]) == sorted(arr[i])

    def test_rows_independent(self):
        """Rows should be shuffled independently (not all the same permutation)."""
        np.random.seed(0)
        arr = np.tile(np.arange(10), (100, 1))
        result = _shuffle_rows(arr)
        # With 100 rows of 10 elements, it's virtually impossible
        # for all rows to end up with the same permutation
        assert not np.all(result == result[0])

    def test_single_element_rows(self):
        arr = np.array([[5], [3]])
        result = _shuffle_rows(arr)
        np.testing.assert_array_equal(result, arr)


class TestGenerateUniformTimesAndMachinesAssignment:
    def test_output_type(self):
        from l2d.env.types import JSSPInstance
        instance = generate_uniform_times_and_machines_assignment(3, 4, 1, 99)
        assert isinstance(instance, JSSPInstance)

    def test_shapes(self):
        instance = generate_uniform_times_and_machines_assignment(3, 4, 1, 99)
        assert instance.times.shape == (3, 4)
        assert instance.machines.shape == (3, 4)

    def test_processing_times_in_range(self):
        low, high = 10, 50
        instance = generate_uniform_times_and_machines_assignment(5, 5, low, high)
        assert np.all(instance.times >= low)
        assert np.all(instance.times < high)  # randint upper bound is exclusive

    def test_machine_assignments_are_permutations(self):
        """Each row of machines should be a permutation of [1, ..., num_machines]."""
        num_machines = 5
        instance = generate_uniform_times_and_machines_assignment(10, num_machines, 1, 99)
        expected = set(range(1, num_machines + 1))
        for row in instance.machines:
            assert set(row) == expected

    def test_reproducible_with_seed(self):
        np.random.seed(42)
        a = generate_uniform_times_and_machines_assignment(3, 3, 1, 99)
        np.random.seed(42)
        b = generate_uniform_times_and_machines_assignment(3, 3, 1, 99)
        np.testing.assert_array_equal(a.times, b.times)
        np.testing.assert_array_equal(a.machines, b.machines)


class TestGenerateUniformAndStoreToFile:
    def test_file_created(self, tmp_path):
        generate_uniform_machines_assignment_and_stores_to_file(
            2, 3, 1, 99, file_path=tmp_path, seed=42,
        )
        expected_file = tmp_path / "generatedData2_3_Seed42.npy"
        assert expected_file.exists()

    def test_file_content_matches(self, tmp_path):
        instance = generate_uniform_machines_assignment_and_stores_to_file(
            2, 3, 1, 99, file_path=tmp_path, seed=42,
        )
        loaded = np.load(tmp_path / "generatedData2_3_Seed42.npy", allow_pickle=True)
        np.testing.assert_array_equal(loaded[0], instance.times)
        np.testing.assert_array_equal(loaded[1], instance.machines)

    def test_return_type(self, tmp_path):
        from l2d.env.types import JSSPInstance
        instance = generate_uniform_machines_assignment_and_stores_to_file(
            2, 3, 1, 99, file_path=tmp_path, seed=42,
        )
        assert isinstance(instance, JSSPInstance)
