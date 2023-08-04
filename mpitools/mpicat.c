/*
 * CAT(1) for MPI environment.
 * Program read data from stdin on rank 0 and forward it to stdout on all ranks.
 * Data forwarder from rank 0 to others via MPI_Bcast, which makes it independed
 * on number of nodes.
 * This is very basic implementation with single read/bcast/write loop.
 * 
 * Typical use case is to distribute some blob to mpi workers
 * Examples:
 *
 * Fetch data from rank 0, and distribute it to each host
 *  curl "$URL/data.bin" | mpirun -N 1 --hostfile h.txt | mpicat > data.bin
 *
 * Fetch docker image ,which saved as "docker save $IMAGE | zstdmt -o img.tar.zst"
 * and load it to all hosts. Image will be downloaded only once.
 *
 *  curl "$URL" | mpirun -N 1 --hostfile hosts.txt bash -c './mpicat | zstdcat | docker load'
 *
 * Validate correctness:
 *     dd if=/dev/urandom | mpirun -n 8  bash -c ' ./mpicat | sha1sum'
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>

struct msg {
	int len;
	char data[];
};

int do_read(struct msg *msg, int count, int fd)
{
	int rc;
	msg->len = 0;
	while(count) {
		rc = read(fd, msg->data + msg->len, count);
		if (rc <= 0)
			break;
		count -= rc;
		msg->len += rc;
	}
	return msg->len ? msg->len : rc;
}

int main(int argc, char** argv) {
	int world_rank;
	int buf_sz = 0x1000000; //1MB
	int msg_sz;
	int inbox = 0;
	long gen_count = 0;
	struct msg* msg;

	if (argc == 2)
		buf_sz = atoi(argv[1]);

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	msg_sz = (sizeof(struct msg) + buf_sz);
	msg = (struct msg *)malloc(msg_sz * 2);
	assert(msg != NULL);


	double time = -MPI_Wtime();
	if (!world_rank)
		do_read(&msg[inbox^1], buf_sz, 0);
	inbox ^= 1;
	while(1) {
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast((char *)&msg[inbox], msg_sz, MPI_BYTE, 0, MPI_COMM_WORLD);
		
		//if (!world_rank)
		//	fprintf(stderr, "rank:%d gen:%d count:%d\n", world_rank, gen_count, msg[inbox].len);

		if (msg[inbox].len <= 0) {
			break;
		}
		write(1, msg[inbox].data, msg[inbox].len);
		if (!world_rank)
			do_read(&msg[inbox^1], buf_sz, 0);

		inbox ^= 1;
		gen_count++;
	}		
	MPI_Barrier(MPI_COMM_WORLD);
	time += MPI_Wtime();

	if (!world_rank)
		fprintf(stderr, "total time  = %lf\n", time);

	free(msg);
	return MPI_Finalize();
}
