#!/usr/bin/env bash
# Interactive upstream installer; the user reviews and accepts its license.
set -eu
installer="$1"
target="$2"
if [ "$(uname -m)" != x86_64 ]; then
    echo 'The downloaded installer requires x86_64 Linux.' >&2
    exit 1
fi
work=$(mktemp -d /tmp/nao-meshes.XXXXXX)
cp "$installer" "$work/installer.run"
chmod u+x "$work/installer.run"
"$work/installer.run" --mode text --prefix "$target"
test -f "$target/meshes/V40/HeadYaw.dae"
test -f "$target/meshes/V40/HeadYaw_0.10.stl"
echo 'Installer finished. Returning to Windows for complete URDF validation.'
