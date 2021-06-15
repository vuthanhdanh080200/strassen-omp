# strassen-omp

cú pháp để chạy: 

b1: g++ main.cpp -o main -fopenmp  
b2: ./main size threshold threads mode  
trong đó mode là giải thuật sẽ chạy  
mode = 0 là sử dụng giải thuật strassen  
mode = 1 là sử dụng giải thuật biến thể của strassen  
