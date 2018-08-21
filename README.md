used the following code to sample 0.1% of the data 
`cat train.csv | awk 'BEGIN {srand()} !/^$/ { if (rand() <= 0.001 || FNR==1) print $0}' > sample.csv`
