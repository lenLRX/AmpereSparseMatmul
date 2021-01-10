import numpy as np
from numpy.core.fromnumeric import nonzero
import argparse


def gen_4in2_metadata():
    metadata = np.random.permutation(np.arange(4))[:2].astype('uint8')
    metadata.sort()
    return metadata


def make_sparse_metadata(row, col):
    metadata_col = col // 4
    return [[gen_4in2_metadata() for _ in range(metadata_col)]
            for _ in range(row)]


def set_zero_by_metadata(mat, metadata, dtype):
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


def make_sparse_mat(row, col, dtype):
    mat = np.random.rand(row, col).astype(dtype)
    metadata = make_sparse_metadata(row, col)
    mat = set_zero_by_metadata(mat, metadata, dtype)
    return mat, metadata


def compress_sparse_mat(mat, metadata, dtype):
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
    half_row_num = row // 2
    half_col_num = col // 2 // 2
    bin_meta = np.zeros((size,), dtype='uint32')
    for row_id in range(half_row_num):
        for sub_col in range(2):
            bit_offset = 0
            first_half_row = np.concatenate(metadata[row_id][sub_col * half_col_num: (sub_col + 1) * half_col_num])
            second_half_row = np.concatenate(metadata[row_id + half_row_num][sub_col * half_col_num: (sub_col + 1) * half_col_num])
            whole_row = np.concatenate([first_half_row, second_half_row])
            for idx in whole_row:
                bin_meta[row_id * 2 + sub_col] |= idx << bit_offset
                bit_offset += 2
    return bin_meta


def make_dense_mat(row, col, dtype):
    return np.random.rand(row, col).astype(dtype)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, required=True)
    parser.add_argument('-m', type=int, required=True)
    parser.add_argument('-k', type=int, required=True)
    parser.add_argument('--dtype', default='float16', required=False)
    args = parser.parse_args()
    dtype = args.dtype
    N = args.n
    M = args.m
    K = args.k
    mat_a, metadata = make_sparse_mat(M, K, dtype)
    compressed_mat_a = compress_sparse_mat(mat_a, metadata, dtype)
    bin_meta = metadata_to_binary(metadata)
    mat_b = make_dense_mat(K, N, dtype)
    #mat_c = make_dense_mat(M, N, dtype)
    mat_c = np.zeros((M, N), dtype)
    mat_d = np.matmul(mat_a, mat_b) + mat_c

    compressed_mat_a.tofile('a.bin')
    bin_meta.tofile('metadata.bin')
    print(bin_meta)
    mat_b.tofile('b.bin')
    mat_c.tofile('c.bin')
    mat_d.tofile('d.bin')
