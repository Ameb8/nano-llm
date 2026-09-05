#!/usr/bin/env sh
set -eu
if [ "$#" -ne 2 ]; then
    echo "usage: $0 <x86_64|aarch64> <binary>" >&2
    exit 2
fi
architecture=$1
binary=$2
case "$architecture" in
    x86_64) elf_machine='x86-64'; host_machine='x86_64'; qemu='' ;;
    aarch64) elf_machine='aarch64'; host_machine='aarch64'; qemu='qemu-aarch64-static' ;;
    *) echo "unsupported architecture: $architecture" >&2; exit 2 ;;
esac
test -x "$binary"
file_output=$(file "$binary")
printf '%s\n' "$file_output" | grep -Eiq "ELF 64-bit.*$elf_machine"
printf '%s\n' "$file_output" | grep -Eiq 'statically linked|static-pie linked'
if [ "$(uname -m)" = "$host_machine" ]; then
    "$binary" --help >/dev/null
elif command -v "$qemu" >/dev/null 2>&1; then
    "$qemu" "$binary" --help >/dev/null
else
    echo "artifact inspection passed; CLI smoke test requires a compatible runner" >&2
fi
