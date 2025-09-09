{
  outputs =
    { ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forEachSystem =
        f:
        builtins.listToAttrs (
          builtins.map (system: {
            name = system;
            value = f system;
          }) systems
        );
      withDefault = f: forEachSystem (system: f (import ./default.nix { inherit system; }));
    in
    {
      devShells = withDefault (default: {
        default = default.shell;
      });

      packages = withDefault (default: {
        inherit default;
      });

      checks = withDefault (default: {
        inherit (default) miriTest;
      });
    };
}
