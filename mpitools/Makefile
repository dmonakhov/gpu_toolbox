CC      = mpicc
CFLAGS+ = -W -Wall

SRCS=mpicat.c
OBJS=$(SRCS:.c=.o)
DEPS=$(OBJS:.o=.d)

.PHONY: all clean
all: mpicat

mpicat: mpicat.o

-include $(DEPS)  # this makes magic happen

clean:
	rm -f mpicat
	rm -f $(DEPS) $(OBJS)
