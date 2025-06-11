{
  system ? builtins.currentSystem,
  pkgs ? import <nixpkgs> {
    inherit system;
    overlays = [
      (import (builtins.fetchTarball "https://github.com/oxalica/rust-overlay/archive/master.tar.gz"))
    ];
  },
}:
pkgs.mkShell {
  packages = [
    (pkgs.rust-bin.nightly.latest.default.override {
      extensions = [
        "rust-src"
        "rust-analyzer"
        "miri"
      ];
    })
    pkgs.cargo-nextest
  ];
}
