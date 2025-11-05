from math import floor

# region Summary
"""
Following are some utilities for tile coding from R. Sutton.
To make each file self-contained, they were copied them from http://incompleteideas.net/tiles/tiles3.py-remove with some naming convention changes.
"""
# endregion Summary

class IHT:
    # region Summary
    """
    Index Hash Table - a structure to handle collisions
    """
    # endregion Summary

    # region Constructor

    def __init__(self, size_val):
        self.size = size_val
        self.overfull_count = 0
        self.dictionary = {}

    # endregion Constructor

    # region Functions

    def count(self):
        return len(self.dictionary)

    def full(self):
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfull_count == 0: print('IHT full, starting to allow collisions')
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count

    # endregion Functions


# region Functions

def hash_coords(coordinates, m, read_only=False):
    # region Summary
    """
    Hash coordinates.
    :param coordinates: Coordinates
    :param m: Either an IHT of a given size, or an integer "size" (range of the indices from 0)
    :param read_only: Read-only?
    :return: Hash coordinates
    """
    # endregion Summary

    # region Body

    if isinstance(m, IHT):
        return m.get_index(tuple(coordinates), read_only)

    if isinstance(m, int):
        return hash(tuple(coordinates)) % m

    if m is None:
        return coordinates

    # endregion Body

def tiles(iht_or_size, num_tilings, floats, ints=None, read_only=False):
    # region Summary
    """
    Maps floating and integer variables to a list of tiles
    :param iht_or_size: Either an IHT of a given size, or an integer "size" (range of the indices from 0)
    :param num_tilings: Should be a power of 2. To make the offsetting work properly,
                        it should also be greater than or equal to 4 times the number of floats.
    :param floats: The float variables will be gridded at unit intervals,
                   so generalization will be by approximately 1 in each direction,
                   and any scaling will have to be done externally before calling tiles.
    :param ints: Integer variables
    :param read_only: Read-only?
    :return: Num-tilings tile indices corresponding to the floats and ints
    """
    # endregion Summary

    # region Body

    if ints is None:
        ints = []

    q_floats = [floor(f * num_tilings) for f in floats]

    tiles = []

    for tiling in range(num_tilings):
        tilingX2 = tiling * 2

        coords = [tiling]

        b = tiling

        for q in q_floats:
            coords.append((q + b) // num_tilings)

            b += tilingX2

        coords.extend(ints)

        tiles.append(hash_coords(coords, iht_or_size, read_only))

    return tiles

    # endregion Body

# endregion Functions
