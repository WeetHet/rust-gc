{
  system ? builtins.currentSystem,
  sources ? import ./npins,
  pkgs ? import sources.nixpkgs {
    inherit system;
    overlays = [ (import sources.rust-overlay) ];
  },
}:
let
  rust-toolchain = pkgs.rust-bin.selectLatestNightlyWith (
    toolchain:
    toolchain.default.override {
      extensions = [
        "rust-src"
        "rust-analyzer"
        "miri"
      ];
    }
  );
  package = pkgs.callPackage ./package.nix { };
  rustPlatform = pkgs.makeRustPlatform {
    cargo = rust-toolchain;
    rustc = rust-toolchain;
  };
  debugPackage = package.override {
    inherit rustPlatform;
    buildType = "debug";
  };
  stdlibDeps = rustPlatform.importCargoLock {
    lockFile = "${rust-toolchain}/lib/rustlib/src/rust/library/Cargo.lock";
  };
in
package
// {
  miriTest = debugPackage.overrideAttrs (old: {
    nativeCheckInputs = (old.nativeCheckInputs or [ ]) ++ [ pkgs.writableTmpDirAsHomeHook ];

    postCheck = "cargo miri test --offline";
    cargoDeps = pkgs.symlinkJoin {
      inherit (old.cargoDeps) name;
      paths = [
        old.cargoDeps
        stdlibDeps
      ];
    };
    env.MIRIFLAGS = "-Zmiri-disable-isolation -Zmiri-env-forward=RUST_BACKTRACE";
    env.RUST_BACKTRACE = 1;
  });
  shell = pkgs.mkShell {
    packages = [ rust-toolchain ];
    env.MIRIFLAGS = "-Zmiri-disable-isolation -Zmiri-env-forward=RUST_BACKTRACE";
    env.RUST_BACKTRACE = 1;
  };
}
