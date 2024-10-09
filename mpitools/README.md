# MPICAT is CAT(1) for MPI environment.
Program read data from stdin on rank 0 and forward it to stdout on all ranks.
Data forwarder from rank 0 to others via MPI_Bcast, which makes it independent
on number of nodes.
This is very basic implementation with single read/bcast/write loop.

# Build
```
make 
```
# Why

MPICAT solves data distribution scalability problem in HPC environmet, data distributed via fast MPI
interconnect from rank 0 to all ranks. 

# Examples
## Fetch data once, distribute to all

Given cluster of N nodes, we want to fetch some blob from slow internet to each node.
Naive approach will be to fetch data from each host which result in download multiplication.
Lets download blob only once on rank 0 and distribute it to all others with MPICAT via fast interconnect.
```
 curl "$URL/data.bin" | mpirun -N 1 --hostfile hosts.txt bash -c './mpicat > data.bin'
```

## Fetch container from rank 0 and distribute to all others.
Given large docker image which saved in OCI format, for example:
```
## Fetch image once, on rank 0
docker pull nvcr.io/nvidia/pytorch:23.10-py3
docker docker save nvcr.io/nvidia/pytorch:23.10-py3 | zstdmt -o img.tar.zst
# Distribute it to all mpi ranks
cat img.tar.zst | mpirun -N 1 --hostfile hosts.txt bash -c './mpicat | zstdcat | docker load'

```
## Fetch OCI image from S3 bucket once, and distribute it to all ranks, Image will be downloaded only once.
```
 aws s3 cp s3://my_bucket/docker-img.tar.zst - | mpirun -N 1 --hostfile hosts.txt bash -c './mpicat | zstdcat | docker load'
```

## Validate data correctness
```
dd if=/dev/urandom bs=1M count=10 | mpirun -n 8  bash -c ' ./mpicat | sha1sum'
```
