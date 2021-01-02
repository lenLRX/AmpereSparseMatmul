import numpy as np
from numpy.core.fromnumeric import nonzero

M = 16
N = 8
K = 16

dtype = 'float16'


def gen_4in2_metadata():
    metadata = np.random.permutation(np.arange(4))[:2].astype('uint8')
    metadata.sort()
    return metadata


def make_sparse_metadata(row, col):
    metadata_col = col // 4
    return [[gen_4in2_metadata() for _ in range(metadata_col)]
            for _ in range(row)]


def set_zero_by_metadata(mat, metadata):
    row, col = mat.shape
    non_zero_mat = np.zeros((row, col), dtype=dtype)
    for row_index in range(row):
        metadata_row = metadata[row_index]
        for i, metadata_block in enumerate(metadata_row):
            offset = i * 4
            for idx in metadata_block:
                non_zero_mat[row_index,  offset +
                             idx] = mat[row_index, offset + idx]
    return non_zero_mat


def make_sparse_mat(row, col):
    mat = np.random.rand(row, col).astype(dtype)
    metadata = make_sparse_metadata(row, col)
    mat = set_zero_by_metadata(mat, metadata)
    return mat, metadata


def compress_sparse_mat(mat, metadata):
    row, col = mat.shape
    non_zero_mat = np.zeros((row, col//2), dtype=dtype)
    for row_index in range(row):
        metadata_row = metadata[row_index]
        for i, metadata_block in enumerate(metadata_row):
            offset = i * 4
            for ii, idx in enumerate(metadata_block):
                non_zero_mat[row_index,  i * 2 +
                             ii] = mat[row_index, offset + idx]
    return non_zero_mat


def metadata_to_binary(metadata):
    row = len(metadata)
    col = len(metadata[0]) * 2
    size = row * col // 16
    half_row = row // 2
    bin_meta = np.zeros((size,), dtype='uint32')
    for row_id in range(half_row):
        bit_offset = 0
        first_half_row = np.concatenate(metadata[row_id])
        second_half_row = np.concatenate(metadata[row_id + half_row])
        whole_row = np.concatenate([first_half_row, second_half_row])
        for idx in whole_row:
            bin_meta[row_id] |= idx << bit_offset
            bit_offset += 2
    return bin_meta


def make_dense_mat(row, col):
    return np.random.rand(row, col).astype(dtype)


if __name__ == '__main__':
    mat_a, metadata = make_sparse_mat(M, K)
    compressed_mat_a = compress_sparse_mat(mat_a, metadata)
    bin_meta = metadata_to_binary(metadata)
    mat_b = make_dense_mat(K, N)
    mat_c = make_dense_mat(M, N)
    mat_d = np.matmul(mat_a, mat_b) + mat_c

    compressed_mat_a.tofile('a.bin')
    bin_meta.tofile('metadata.bin')
    mat_b.tofile('b.bin')
    mat_c.tofile('c.bin')
    mat_d.tofile('d.bin')
