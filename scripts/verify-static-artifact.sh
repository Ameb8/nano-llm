#!/usr/bin/env sh
set -eu
if [ "$#" -ne 2 ]; then
    echo "usage: $0 <x86_64|aarch64> <binary>" >&2
    exit 2
fi
architecture=$1
binary=$2
case "$architecture" in
    x86_64) elf_machine='x86-64'; host_machine='x86_64'; qemu='qemu-x86_64-static qemu-x86_64' ;;
    aarch64) elf_machine='aarch64'; host_machine='aarch64'; qemu='qemu-aarch64-static qemu-aarch64' ;;
    *) echo "unsupported architecture: $architecture" >&2; exit 2 ;;
esac
test -x "$binary"
file_output=$(file "$binary")
printf '%s\n' "$file_output" | grep -Eiq "ELF 64-bit.*$elf_machine"
printf '%s\n' "$file_output" | grep -Eiq 'statically linked|static-pie linked'
# A static-pie may have a dynamic section, but it must not need an ELF
# interpreter or a shared object at runtime.  `file` alone is not enough to
# distinguish that from a dynamically linked executable.
! readelf -l "$binary" | grep -q 'Requesting program interpreter'
! readelf -d "$binary" 2>/dev/null | grep -q '(NEEDED)'
if [ "$(uname -m)" = "$host_machine" ]; then
    "$binary" --help >/dev/null
else
    runner=''
    direct_execution=0
    if "$binary" --help >/dev/null 2>&1; then
        direct_execution=1
    else
        for candidate in $qemu; do
            if command -v "$candidate" >/dev/null 2>&1; then
                runner=$candidate
                break
            fi
        done
    fi
    if [ "$direct_execution" = 1 ]; then
        : # The host has transparent binfmt_misc execution for this architecture.
    elif [ -n "$runner" ]; then
        "$runner" "$binary" --help >/dev/null
    elif [ "${NANO_LLM_REQUIRE_EXECUTION:-0}" = 1 ]; then
        echo "artifact inspection passed, but no compatible runner is available for $architecture" >&2
        exit 1
    else
        echo "artifact inspection passed; CLI smoke test requires a compatible runner" >&2
    fi
fi
