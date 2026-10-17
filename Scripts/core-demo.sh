#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

host_target="$(rustc +stable -vV | sed -n 's/^host: //p')"
cargo +stable test --manifest-path crates/legday-core/Cargo.toml --target "$host_target"
cargo +stable run --manifest-path crates/legday-core/Cargo.toml --target "$host_target" \
    --example window_demo
