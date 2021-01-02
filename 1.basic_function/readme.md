# demo of Ampere GPU's sparse matmul 

## run the code
1. generate test data
```
python3 sparse_mmad_data.py
```
2. compile test code
```
nvcc -arch sm_80 sparse_mmad.cu
```
3. run the test program
```
./a.out
```
4. check result
```
python3 check_result.py
```