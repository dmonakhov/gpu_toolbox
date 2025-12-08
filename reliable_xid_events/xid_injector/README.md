# XID Fault Injector

A simple CUDA program to trigger GPU XID errors for testing monitoring systems.

## WARNING

This tool intentionally causes GPU errors! Use with caution:
- May require GPU reset after use
- XID 43 can hang the system
- Run on test/dev machines only

## Build

```bash
make
```

Requires NVIDIA CUDA toolkit (`nvcc`).

## Usage

```bash
./xid_inject <xid_type>
```

### Available XID Types

| Type    | XID | Description                        | Risk              |
|---------|-----|------------------------------------|-------------------|
| `31`    | 31  | GPU memory page fault (NULL deref) | Low - recoverable |
| `31oob` | 31  | Out-of-bounds memory access        | Low - recoverable |
| `13`    | 13  | Graphics exception (trap)          | Medium            |
| `43`    | 43  | GPU timeout (infinite loop)        | High - may hang   |

## Testing with XID Tracer

Terminal 1 - Start the eBPF tracer:
```bash
cd ../xid_tracer
sudo ./xid_tracer
```

Terminal 2 - Trigger an XID:
```bash
./xid_inject 31
```

Terminal 1 should show:
```
[XID ERROR] Code: 31
  PCI Device: 0000:XX:XX
  Message: GPU memory page fault...
```

