{
  rustPlatform,
  buildType ? "release",
}:
rustPlatform.buildRustPackage {
  inherit buildType;

  pname = "rust-gc-example";
  version = "0.1.0";

  src = ./.;

  cargoLock.lockFile = ./Cargo.lock;
}
