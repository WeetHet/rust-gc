{
  system ? builtins.currentSystem,
  pkgs ? import <nixpkgs> {
    inherit system;
    overlays = [ (import <rust-overlay>) ];
  },
}:
pkgs.mkShell {
  packages = [
    (pkgs.rust-bin.selectLatestNightlyWith (
      toolchain:
      toolchain.default.override {
        extensions = [
          "rust-src"
          "rust-analyzer"
          "miri"
        ];
      }
    ))
    pkgs.cargo-nextest
  ];
  env.MIRIFLAGS = "-Zmiri-disable-isolation -Zmiri-env-forward=RUST_BACKTRACE";
  env.RUST_BACKTRACE = 1;
}
