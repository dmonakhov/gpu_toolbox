CC      := gcc
CFLAGS  := -O2 -Wall -Wextra
TARGET  := hv_freeze_detect
PREFIX  := /usr/local

.PHONY: all clean install uninstall

all: $(TARGET)

$(TARGET): hv_freeze_detect.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

install: $(TARGET)
	install -Dm0755 $(TARGET) $(PREFIX)/bin/$(TARGET)

uninstall:
	rm -f $(PREFIX)/bin/$(TARGET)
