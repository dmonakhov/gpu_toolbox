CC      ?= gcc
CFLAGS  ?= -O2 -Wall
BIN     := bin

all: $(BIN)/nv_intr_dump $(BIN)/nv_bar0_recorder $(BIN)/nv_intr_monitor

$(BIN):
	mkdir -p $(BIN)

$(BIN)/nv_intr_dump: tools/nv_intr_dump.c | $(BIN)
	$(CC) $(CFLAGS) -o $@ $<

$(BIN)/nv_bar0_recorder: tools/nv_bar0_recorder.c | $(BIN)
	$(CC) $(CFLAGS) -pthread -o $@ $<

$(BIN)/nv_intr_monitor: cmd/nv_intr_monitor/main.go go.mod | $(BIN)
	go build -o $@ ./cmd/nv_intr_monitor

image:
	docker build -f deploy/Dockerfile -t nv-intr-monitor .

clean:
	rm -rf $(BIN)

.PHONY: all image clean
