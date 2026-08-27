# Build: docker build -f deploy/Dockerfile -t nv-intr-monitor .
FROM golang:1.22 AS build
WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
RUN CGO_ENABLED=0 go build -o /nv_intr_monitor ./cmd/nv_intr_monitor

FROM gcr.io/distroless/static-debian12
COPY --from=build /nv_intr_monitor /nv_intr_monitor
ENTRYPOINT ["/nv_intr_monitor"]
